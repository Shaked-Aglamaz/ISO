"""
Single source of truth for raw (non-normalized) group topographic aggregation.

Both step4_distribution_analysis (per-group raw topo) and step5_topo_comparison
(three-group raw topo) call into this module so the displayed group "mean"
matches step6's violin (mean of per-subject means) and the topographic image
fills missing channels by spatial neighbor imputation rather than by
selection-biased per-channel averaging or by subject-mean filling.
"""
import numpy as np
import pandas as pd
import mne
from mne.channels import find_ch_adjacency, make_standard_montage


def neighbor_impute(values, adjacency, max_iters=5):
    """Fill NaN entries in ``values`` using the mean of adjacency-neighbors
    that are non-NaN for the same observation.

    Iterates so that NaN cells whose neighbors were all NaN in pass 1 fill
    once outer cells fill. Any NaN remaining after ``max_iters`` falls back
    to ``np.nanmean(values)`` — the subject's overall mean across valid
    channels.
    """
    out = np.asarray(values, dtype=float).copy()
    if not np.isnan(out).any():
        return out

    fallback = np.nanmean(values)
    adj = adjacency.toarray() if hasattr(adjacency, 'toarray') else np.asarray(adjacency)
    adj = adj.astype(bool)
    np.fill_diagonal(adj, False)

    for _ in range(max_iters):
        nan_mask = np.isnan(out)
        if not nan_mask.any():
            return out
        snapshot = out.copy()
        progress = False
        for i in np.where(nan_mask)[0]:
            nbr_vals = snapshot[adj[i]]
            valid = nbr_vals[~np.isnan(nbr_vals)]
            if valid.size:
                out[i] = valid.mean()
                progress = True
        if not progress:
            break

    if np.isnan(out).any():
        out = np.where(np.isnan(out), fallback, out)
    return out


def build_raw_group_topo(subjects_data, metric, sfreq=250):
    """Construct a raw (non-normalized) group topography + matching displayed mean.

    Parameters
    ----------
    subjects_data : dict[str, pd.DataFrame]
        ``subject_id -> DataFrame`` with at least ``'channel'`` and ``metric``
        columns. NaN (or empty cells) in ``metric`` mean the Gaussian fit
        failed for that channel/subject.
    metric : str
        Column name on each subject DataFrame (e.g. ``'auc'``,
        ``'peak_frequency'``, ``'bandwidth'``).
    sfreq : float
        Sampling rate for the synthetic ``mne.Info`` (cosmetic only).

    Returns
    -------
    group_topo : ndarray (n_ch,)
        Per-channel mean across subjects of neighbor-imputed per-subject
        topographies. Pass directly to ``mne.viz.plot_topomap``.
    displayed_mean : float
        Mean over subjects of ``np.nanmean`` of that subject's RAW values
        across valid channels. Matches step6 violin's group statistic. Use
        this for the title — do NOT use ``np.nanmean(group_topo)``.
    available_channels : list[str]
        Channels with EGI_256 positions, in canonical montage order.
    info : mne.Info
        For ``mne.viz.plot_topomap``.
    """
    if not subjects_data:
        raise ValueError("subjects_data is empty")

    all_channels = set()
    for df in subjects_data.values():
        all_channels.update(df['channel'].tolist())

    montage = make_standard_montage('EGI_256')
    # Canonical order from the montage, restricted to channels that appear in the data.
    available_channels = [ch for ch in montage.ch_names if ch in all_channels]
    if len(available_channels) < 10:
        raise ValueError(f"Only {len(available_channels)} channels matched EGI_256 montage")

    info = mne.create_info(ch_names=available_channels, sfreq=sfreq, ch_types='eeg')
    info.set_montage(montage)

    adjacency, _ = find_ch_adjacency(info, ch_type='eeg')

    subject_means = []
    imputed_stack = []
    for subject_id, df in subjects_data.items():
        if metric not in df.columns:
            raise KeyError(f"Subject {subject_id!r} DataFrame is missing column {metric!r}")
        # Per-subject mean over ALL channels (matches step6 violin's group statistic
        # exactly — includes channels like VREF that have no EGI_256 position).
        raw_all = pd.to_numeric(df[metric], errors='coerce').to_numpy(dtype=float)
        if np.all(np.isnan(raw_all)):
            continue
        subject_means.append(float(np.nanmean(raw_all)))
        # For the topographic image, restrict to positioned channels and neighbor-impute.
        sub = df[df['channel'].isin(available_channels)].copy()
        sub = sub.set_index('channel').reindex(available_channels).reset_index()
        raw_pos = pd.to_numeric(sub[metric], errors='coerce').to_numpy(dtype=float)
        imputed_stack.append(neighbor_impute(raw_pos, adjacency))

    if not imputed_stack:
        raise ValueError(f"No subjects had any valid {metric!r} data")

    group_topo = np.nanmean(np.array(imputed_stack), axis=0)
    displayed_mean = float(np.mean(subject_means))
    return group_topo, displayed_mean, available_channels, info
