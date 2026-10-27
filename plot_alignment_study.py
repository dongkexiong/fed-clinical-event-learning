"""Create standalone scientific figures from the completed study summaries."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODES = ("raw", "aligned", "normalized_events", "alignment_aware")
LABELS = ("Raw", "Code aligned", "Normalized events", "Normalized + replica weights")
COLORS = ("#777777", "#0072B2", "#009E73", "#D55E00")
MARKERS = ("o", "s", "^", "D")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    index = json.loads((args.study_dir / "study_summary.json").read_text(encoding="utf-8"))
    summaries = {name: json.loads((args.study_dir / path).read_text(encoding="utf-8"))
                 for name, path in index["summaries"].items()}
    names = list(summaries)
    x = np.array([100 * index["scenarios"][name] for name in names])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "savefig.dpi": 300})
    outputs = []
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.5))
    for ax, metric, title in zip(axes, ("f1", "auroc", "auprc"), ("F1", "AUROC", "Average precision (AP)")):
        for mode, label, color, marker in zip(MODES, LABELS, COLORS, MARKERS):
            rows = [summaries[name]["summary"]["federated_metrics"][mode]["fedavg"][metric] for name in names]
            mean = np.array([r["mean"] for r in rows])
            lo, hi = np.array([r["bootstrap_95_ci"] for r in rows]).T
            ax.plot(x, mean, marker=marker, color=color, linewidth=1.8, label=label)
            ax.fill_between(x, lo, hi, color=color, alpha=0.10)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xlabel("Joint code/value/unit missingness (%)")
        ax.grid(axis="y", alpha=0.22)
        ax.set_ylim(0.48 if metric != "auroc" else 0.74, 0.80 if metric != "auroc" else 0.92)
    axes[0].set_ylabel("Macro mean across logical sites")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.09))
    fig.text(0.5, 0.02, "FedAvg; 5 perturbation/split seeds; shaded bands: descriptive 95% bootstrap intervals.\n"
             "Patient-ID missingness probability is half the x-axis setting; synthetic emulation only.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.19, 1, 1))
    for extension in ("png", "svg", "pdf"):
        path = args.output_dir / f"fedavg_missingness.{extension}"
        fig.savefig(path)
        outputs.append(path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    positions = np.arange(len(names))
    for i, (mode, label, color) in enumerate(zip(MODES, LABELS, COLORS)):
        values = []
        for name in names:
            runs = summaries[name]["runs"]
            byte_counts = []
            for run in runs:
                f = json.loads((args.study_dir / name / run["federated_results"]).read_text(encoding="utf-8"))
                byte_counts.append(f["modes"][mode]["methods"]["fedavg"]["communication_bytes"])
            values.append(np.mean(byte_counts) / 1_000_000)
        ax.bar(positions + (i - 1.5) * 0.18, values, width=0.18, color=color, label=label)
    ax.set_yscale("log")
    ax.set_ylabel("Estimated numeric/digest payload (MB, log scale)")
    ax.set_xticks(positions, [f"{int(v)}%" for v in x])
    ax.set_xlabel("Joint code/value/unit missingness")
    ax.set_ylim(0.1, 30)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=2, frameon=False)
    fig.text(0.5, 0.02, "20 rounds; mean of 5 seeds. Includes model upload/broadcast, threshold counts, and linkage digests/weights.\n"
             "Excludes identifiers, headers, transport and encryption overhead; traffic was not measured.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.10, 1, 0.93))
    for extension in ("png", "svg", "pdf"):
        path = args.output_dir / f"fedavg_payload.{extension}"
        fig.savefig(path)
        outputs.append(path)
    plt.close(fig)
    provenance = {"study_summary_sha256": sha(args.study_dir / "study_summary.json"),
                  "generator_sha256": sha(Path(__file__)), "matplotlib": matplotlib.__version__,
                  "figures_sha256": {path.name: sha(path) for path in outputs}}
    (args.output_dir / "figure_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(outputs)} figures to {args.output_dir}")


if __name__ == "__main__":
    main()
