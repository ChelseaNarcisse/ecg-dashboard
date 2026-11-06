"""Static, report-ready figures (PNG) built from backend/models/metrics.json.

Run after training:   python training/make_report_figures.py
Output:               reports/figures/*.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
METRICS = json.loads((ROOT / "backend" / "models" / "metrics.json").read_text())
OUT = ROOT / "reports" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

CLASS_COLORS = {"N": "#2a78d6", "SVEB": "#eb6834", "VEB": "#1baf7a", "F": "#eda100", "Q": "#e87ba4"}
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK = "#52514e"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#d6d5d0", "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})
NAMES = {k: v["display_name"] for k, v in METRICS["models"].items()}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


def class_distribution():
    counts = METRICS["dataset"]["class_counts"]
    classes = [c for c in counts if counts[c] > 0]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.bar(classes, [counts[c] for c in classes], color=[CLASS_COLORS[c] for c in classes])
    ax.set_yscale("log")
    ax.bar_label(bars, labels=[f"{counts[c]:,}" for c in classes], padding=2)
    ax.set_ylabel("beats (log scale)")
    ax.set_title("MIT-BIH class distribution (after removing Q)", loc="left")
    save(fig, "fig1_class_distribution.png")


def model_comparison():
    names = [n for n in METRICS["models"]]
    metrics = [("accuracy", BLUE), ("balanced_accuracy", ORANGE), ("macro_f1", AQUA)]
    x = np.arange(len(names))
    w = 0.26
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for i, (m, c) in enumerate(metrics):
        vals = [METRICS["models"][n]["inter_patient"][m] for n in names]
        b = ax.bar(x + (i - 1) * w, vals, w, label=m.replace("_", " "), color=c)
        ax.bar_label(b, fmt="%.2f", fontsize=7, padding=1)
    ax.set_xticks(x, [NAMES[n] for n in names], fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("score on unseen patients (DS2)")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.set_title("Inter-patient model comparison", loc="left")
    save(fig, "fig2_model_comparison.png")


def confusion(name=None):
    name = name or METRICS["best_model"]
    ip = METRICS["models"][name]["inter_patient"]
    cm = np.array(ip["confusion_matrix"], float)
    labels = ip["labels"]
    keep = [i for i in range(len(labels)) if cm[i].sum() > 0]
    cm, labels = cm[np.ix_(keep, keep)], [labels[i] for i in keep]
    norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{norm[i, j]:.0%}\n({int(cm[i, j]):,})", ha="center", va="center",
                    fontsize=8, color="white" if norm[i, j] > 0.55 else "#0b0b0b")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    ax.set_title(f"{NAMES[name]} - unseen patients (recall)", loc="left", fontsize=10)
    for s in ax.spines.values():
        s.set_visible(False)
    save(fig, f"fig3_confusion_{name}.png")


def ablation():
    abl = METRICS["ablation"]
    names = list(abl)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    raw = [abl[n]["raw_only"]["balanced_accuracy"] for n in names]
    eng = [abl[n]["raw_plus_engineered"]["balanced_accuracy"] for n in names]
    b1 = ax.bar(x - 0.2, raw, 0.38, label="raw features only", color=ORANGE)
    b2 = ax.bar(x + 0.2, eng, 0.38, label="raw + engineered", color=BLUE)
    ax.bar_label(b1, fmt="%.3f", fontsize=8)
    ax.bar_label(b2, fmt="%.3f", fontsize=8)
    ax.set_xticks(x, [NAMES[n] for n in names])
    ax.set_ylim(0, 1)
    ax.set_ylabel("balanced accuracy (unseen patients)")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Feature-engineering ablation", loc="left")
    save(fig, "fig4_feature_engineering_ablation.png")


def importance(name=None):
    name = name or METRICS["best_model"]
    imp = METRICS["models"][name]["feature_importance"][:15][::-1]
    fig, ax = plt.subplots(figsize=(6, 4.8))
    colors = [BLUE if f["engineered"] else ORANGE for f in imp]
    ax.barh([f["feature"] for f in imp], [f["importance"] for f in imp],
            xerr=[f["std"] for f in imp], color=colors, error_kw=dict(ecolor="#9a9993", lw=1))
    ax.set_xlabel("permutation importance (drop in balanced accuracy)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLUE, label="engineered"), Patch(color=ORANGE, label="raw")],
              frameon=False, loc="lower right")
    ax.set_title(f"Top 15 features - {NAMES[name]}", loc="left")
    save(fig, f"fig5_feature_importance_{name}.png")


def evaluation_design():
    best = "hgb" if "hgb" in METRICS["random_split"] else list(METRICS["random_split"])[0]
    rows = [
        ("Random split\n(same patients)", METRICS["random_split"][best]),
        ("Inter-patient\n(unseen patients)", METRICS["models"][best]["inter_patient"]),
        ("External INCART\n(other hospital)", METRICS["external_incart"].get(best, {})),
        ("INCART without\nHz conversion", METRICS["external_incart"].get(f"{best}_wrong_sampling_rate", {})),
    ]
    rows = [r for r in rows if r[1]]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7, 3.8))
    b1 = ax.bar(x - 0.2, [r[1]["accuracy"] for r in rows], 0.38, label="accuracy", color=BLUE)
    b2 = ax.bar(x + 0.2, [r[1]["balanced_accuracy"] for r in rows], 0.38, label="balanced accuracy", color=ORANGE)
    ax.bar_label(b1, fmt="%.2f", fontsize=8)
    ax.bar_label(b2, fmt="%.2f", fontsize=8)
    ax.set_xticks(x, [r[0] for r in rows], fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    ax.set_title(f"{NAMES[best]}: how the evaluation design changes the score", loc="left")
    save(fig, "fig6_evaluation_design.png")


if __name__ == "__main__":
    class_distribution()
    model_comparison()
    confusion()
    ablation()
    importance()
    evaluation_design()
