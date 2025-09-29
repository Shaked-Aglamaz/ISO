import pandas as pd
import matplotlib.pyplot as plt
import mne
from pathlib import Path
import seaborn as sns
import os

mne.set_log_level("error")
plt.style.use('default')

PLOT_DETAILS = {
    'avg_peak_frequency': {'name': 'Peak Frequency', 'unit': 'Hz'},
    'avg_bandwidth': {'name': 'Bandwidth', 'unit': 'Hz'},
    'avg_auc': {'name': 'Area Under Curve', 'unit': 'AUC'}
}


def prepare_channel_data_for_topography(df, parameter_name):
    # Filter out channels that don't have data
    valid_data = df.dropna(subset=[parameter_name])
    channel_names = valid_data['channel'].tolist()
    values = valid_data[parameter_name].values
    
    print(f"Value range of {parameter_name}: {values.min():.4f} - {values.max():.4f}")
    return channel_names, values

def plot_topography(channel_names, values, parameter_name, subject_id, output_dir=None):
    montage = mne.channels.make_standard_montage('GSN-HydroCel-256')
    # Create a dummy info structure for plotting
    info = mne.create_info(channel_names, sfreq=250, ch_types='eeg')
    info.set_montage(montage, on_missing='ignore')
    fig, ax = plt.subplots(figsize=(10, 8))
    im, cm = mne.viz.plot_topomap(values, info, axes=ax, show=False, cmap='viridis', contours=6)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    
    plot_info = PLOT_DETAILS.get(parameter_name)
    if not plot_info:
        print(f"Parameter info for '{parameter_name}' not found.")
        return

    title = f"{plot_info['name']} Topography - Subject {subject_id}"
    cbar_label = f"{plot_info['name']} ({plot_info['unit']})"
    ax.set_title(title, fontsize=14, pad=20)
    cbar.set_label(cbar_label, rotation=270, labelpad=20)
    stats_text = f'Min: {values.min():.4f}\nMax: {values.max():.4f}\nMean: {values.mean():.4f}\nStd: {values.std():.4f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        plot_path = Path(output_dir) / f"{subject_id}_topography_{parameter_name}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    return fig

def create_all_topographies(subject_id, csv_path, output_dir=None):
    df = pd.read_csv(csv_path)
    print(f"Loaded data for {len(df)} channels from {csv_path}")
    
    figures = []
    for col_name, plot_info in PLOT_DETAILS.items():
        channel_names, values = prepare_channel_data_for_topography(df, col_name)
        if len(channel_names) == 0:
            print(f"No valid data found for {plot_info['name']}")
            continue
        
        fig = plot_topography(channel_names, values, col_name, subject_id, output_dir)
        figures.append(fig)
    
    return figures

def plot_params_distribution(sub, csv_path, output_dir):
    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)
    columns_to_plot = ['avg_peak_frequency', 'avg_bandwidth', 'avg_auc']

    for col in columns_to_plot:
        _, ax = plt.subplots(1, 1, figsize=(8, 6))
        sns.histplot(data=df, x=col, kde=True, ax=ax, alpha=0.7, color='skyblue')
        
        ax.set_title(f'{col.replace("_", " ").title()} Distribution - Subject {sub}', fontsize=14)
        ax.set_xlabel(col.replace("_", " ").title(), fontsize=12)
        ax.set_ylabel('Count', fontsize=12)

        mean_val = df[col].mean()
        std_val = df[col].std()
        ax.axvline(mean_val, color='darkblue', linestyle='--', alpha=0.8, linewidth=2, label=f'Mean: {mean_val:.4f}')
        ax.legend()
        
        stats_text = f'Mean: {mean_val:.4f}\nStd: {std_val:.4f}\nMin: {df[col].min():.4f}\nMax: {df[col].max():.4f}'
        ax.text(0.75, 0.75, stats_text, transform=ax.transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                verticalalignment='top', fontsize=10)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, f'{sub}_{col}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

def main():
    subject_id = "EB34"
    csv_path = f"{subject_id}/{subject_id}_all_channels_summary.csv"
    output_dir = f"{subject_id}/global"
    
    print(f"Saving plots to: {output_dir}")
    print("-"*50)
    print(f"Creating topography plots for Subject {subject_id}")
    figures = create_all_topographies(subject_id, csv_path, output_dir)
    print(f"\nCompleted! Created and saved {len(figures)} topography plots.")
    
    print("\nCreating parameter distribution plots...")
    print("-"*50)
    plot_params_distribution(subject_id, csv_path, output_dir)
    
    plt.show()

if __name__ == "__main__":
    main()
