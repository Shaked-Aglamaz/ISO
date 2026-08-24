"""Generate an x-axis strip (0-745 s) to attach below a spindle timeline image."""
import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--duration", type=float, default=745.0, help="Total seconds")
    p.add_argument("--width-px", type=int, default=1200, help="Output width in pixels (match your spindle image)")
    p.add_argument("--height-px", type=int, default=110, help="Output height in pixels")
    p.add_argument("--dpi", type=int, default=100)
    p.add_argument("--major", type=float, default=60.0, help="Major tick interval (s)")
    p.add_argument("--minor", type=float, default=10.0, help="Minor tick interval (s)")
    p.add_argument("--out", default="timeline_axis.png")
    args = p.parse_args()

    fig_w = args.width_px / args.dpi
    fig_h = args.height_px / args.dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=args.dpi)

    ax.set_xlim(0, args.duration)
    ax.set_ylim(0, 1)

    ax.xaxis.set_major_locator(MultipleLocator(args.major))
    ax.xaxis.set_minor_locator(MultipleLocator(args.minor))
    ax.tick_params(axis="x", which="major", length=8, width=1.2, labelsize=10)
    ax.tick_params(axis="x", which="minor", length=4, width=0.8)

    ax.set_yticks([])
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_position(("outward", 0))

    ax.set_xlabel("seconds", fontsize=11)

    spindle_handle = plt.Line2D(
        [0], [0], marker="o", color="none", markerfacecolor="tab:blue",
        markeredgecolor="tab:blue", markersize=8, linestyle="None", label="spindle",
    )
    ax.legend(
        handles=[spindle_handle], loc="upper right", bbox_to_anchor=(1.0, 1.6),
        frameon=False, handletextpad=0.4, fontsize=10,
    )

    plt.subplots_adjust(left=0.02, right=0.99, top=0.85, bottom=0.55)
    fig.savefig(args.out, dpi=args.dpi)
    print(f"saved {args.out}  ({args.width_px}x{args.height_px}px, 0-{args.duration}s)")


if __name__ == "__main__":
    main()
