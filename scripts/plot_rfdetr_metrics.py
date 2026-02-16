"""Generate curve-style plots from RF-DETR metrics.json and results.json.

Invoked automatically by the RF-DETR test pipeline (test_rfdetr.py, run_test_on_testdev.py,
run_test_suite.py). Can also be run standalone to regenerate plots for existing results:

  python scripts/plot_rfdetr_metrics.py [--results-dir <dir>]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# VisDrone class names (category_id 1-10)
VISDRONE_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]


def _ensure_plots_dir(results_dir: Path) -> Path:
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir


def plot_metrics_summary(metrics: dict, plots_dir: Path) -> None:
    """Line curve: mAP50-95, mAP50, mAP75 (YOLO-style curve, not bars)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    keys = ["mAP50-95", "mAP50", "mAP75"]
    vals = [float(metrics.get(k, 0)) for k in keys]
    x = np.arange(len(keys))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, vals, color="#2E86AB", marker="o", markersize=8, linewidth=2, linestyle="-")
    ax.set_xticks(x)
    ax.set_xticklabels(keys)
    ax.set_ylabel("Score")
    ax.set_title("RF-DETR test metrics (VisDrone)")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.3f}", (x[i], v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(plots_dir / "metrics_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {plots_dir / 'metrics_summary.png'}")


def plot_metrics_by_scale(metrics: dict, plots_dir: Path) -> None:
    """Line curve: mAP by object scale (YOLO-style, not bars)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    keys = ["mAP_small", "mAP_medium", "mAP_large"]
    labels = ["Small", "Medium", "Large"]
    vals = [float(metrics.get(k, 0)) for k in keys]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, vals, color="#086788", marker="o", markersize=8, linewidth=2, linestyle="-")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("mAP")
    ax.set_title("RF-DETR mAP by object scale")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.3f}", (x[i], v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(plots_dir / "metrics_by_scale.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {plots_dir / 'metrics_by_scale.png'}")


def plot_all_metrics_bars(metrics: dict, plots_dir: Path) -> None:
    """Two subplots as line curves: main metrics + scale (YOLO-style)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Main metrics – line curve
    k1 = ["mAP50-95", "mAP50", "mAP75"]
    v1 = [float(metrics.get(k, 0)) for k in k1]
    x1 = np.arange(len(k1))
    ax1.plot(x1, v1, color="#2E86AB", marker="o", markersize=8, linewidth=2, linestyle="-")
    ax1.set_xticks(x1)
    ax1.set_xticklabels(k1)
    ax1.set_ylabel("Score")
    ax1.set_title("Main metrics")
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    # By scale – line curve
    k2 = ["mAP_small", "mAP_medium", "mAP_large"]
    labels2 = ["Small", "Medium", "Large"]
    v2 = [float(metrics.get(k, 0)) for k in k2]
    x2 = np.arange(len(labels2))
    ax2.plot(x2, v2, color="#086788", marker="o", markersize=8, linewidth=2, linestyle="-")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(labels2)
    ax2.set_ylabel("mAP")
    ax2.set_title("By object scale")
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("RF-DETR test metrics (VisDrone)", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(plots_dir / "metrics_all.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {plots_dir / 'metrics_all.png'}")


def plot_results_score_histogram(results: list, plots_dir: Path) -> None:
    """Score distribution as a line curve (YOLO-style, not block histogram)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    scores = [r["score"] for r in results]
    if not scores:
        return

    counts, bin_edges = np.histogram(scores, bins=50)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(bin_centers, counts, color="#2E86AB", linewidth=2, linestyle="-")
    ax.fill_between(bin_centers, counts, alpha=0.3, color="#2E86AB")
    ax.set_xlabel("Confidence score")
    ax.set_ylabel("Count")
    ax.set_title("RF-DETR detection score distribution")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "scores_histogram.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {plots_dir / 'scores_histogram.png'}")


def plot_results_per_class(results: list, plots_dir: Path) -> None:
    """Detections per category as line curve (YOLO-style, not bars)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from collections import Counter

    cat_counts = Counter(r["category_id"] for r in results)
    ids = list(range(1, 11))
    names = [VISDRONE_NAMES[i - 1] if 1 <= i <= 10 else f"cat{i}" for i in ids]
    counts = [cat_counts.get(i, 0) for i in ids]
    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, counts, color="#086788", marker="o", markersize=6, linewidth=2, linestyle="-")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Number of detections")
    ax.set_title("RF-DETR detections per class")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "detections_per_class.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {plots_dir / 'detections_per_class.png'}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate YOLO-style plots from RF-DETR metrics and results")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="",
        help="Directory containing metrics.json and results.json (default: test_results/rfdetr_test_dev)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve() if args.results_dir else (REPO_ROOT / "test_results" / "rfdetr_test_dev")
    if not results_dir.is_absolute():
        results_dir = (REPO_ROOT / results_dir).resolve()

    metrics_path = results_dir / "metrics.json"
    results_path = results_dir / "results.json"

    if not metrics_path.exists():
        print(f"ERROR: Not found: {metrics_path}")
        return 1

    plots_dir = _ensure_plots_dir(results_dir)
    print(f"Plots output: {plots_dir}\n")

    # --- From metrics.json ---
    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)

    plot_metrics_summary(metrics, plots_dir)
    plot_metrics_by_scale(metrics, plots_dir)
    plot_all_metrics_bars(metrics, plots_dir)

    # --- From results.json (optional) ---
    if results_path.exists():
        print("Loading results.json (may take a moment)...")
        with open(results_path, encoding="utf-8") as f:
            results = json.load(f)
        if results:
            plot_results_score_histogram(results, plots_dir)
            plot_results_per_class(results, plots_dir)
        else:
            print("  results.json is empty; skipping score/class plots.")
    else:
        print(f"  results.json not found at {results_path}; skipping score/class plots.")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
