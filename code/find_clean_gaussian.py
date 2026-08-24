"""Rank EL3011 channels by 'cleanness' of the Gaussian ISFS fit."""
from pathlib import Path

import pandas as pd

CSV = Path("results/new_iso_results/EL3011/EL3011_all_channels_summary.csv")

df = pd.read_csv(CSV, comment="#")
df = df[df["peak_frequency"].notna()].copy()

# Cleanness score: tall peak, narrow bandwidth, central in 0.0075-0.04 ISFS range.
df["height_to_width"] = df["peak_amplitude"] / df["bandwidth"]

print("Top 15 by peak_amplitude (tall peaks):")
print(df.sort_values("peak_amplitude", ascending=False)
        [["channel", "peak_frequency", "bandwidth", "peak_amplitude", "auc", "height_to_width"]]
        .head(15).to_string(index=False))

print("\nTop 15 by height_to_width (tall AND narrow):")
print(df.sort_values("height_to_width", ascending=False)
        [["channel", "peak_frequency", "bandwidth", "peak_amplitude", "auc", "height_to_width"]]
        .head(15).to_string(index=False))

print("\nTop 15 by AUC:")
print(df.sort_values("auc", ascending=False)
        [["channel", "peak_frequency", "bandwidth", "peak_amplitude", "auc", "height_to_width"]]
        .head(15).to_string(index=False))
