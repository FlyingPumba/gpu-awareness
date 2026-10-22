"""Save comparison figures from a training summary or the final coding report."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_json")
    parser.add_argument("output_directory")
    args = parser.parse_args()
    result = json.loads(Path(args.results_json).read_text())
    out = Path(args.output_directory)
    out.mkdir(parents=True, exist_ok=True)
    if "validation_history" in result:
        history = result["validation_history"]
        selected = result["selected_step"]
    else:
        history = [dict(step=e["step"], C=e["A"]["loss"], Python=e["B"]["loss"],
                        mean=e["mean_loss"]) for e in result["evaluations"]]
        selected = result["best"]["best_step"]
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for ax in axes:
        for label, color in (("C", "#2478b4"), ("Python", "#e08214"), ("mean", "#333333")):
            ax.plot([e["step"] for e in history], [e[label] for e in history],
                    label={"C": "C targets · A40", "Python": "Python targets · RTX", "mean": "Mean"}[label],
                    color=color, linewidth=1.8)
        ax.axvline(selected, color="#777777", linestyle="--", linewidth=1,
                   label=f"Selected update {selected}")
        ax.set_xlabel("Paired optimizer update")
        ax.set_ylabel("Mean solution-token cross-entropy")
        ax.grid(alpha=.2)
    axes[0].set_title("Full training history")
    axes[0].legend(fontsize=8)
    axes[1].set_title("After warmup (expanded scale)")
    tail = [e for e in history if e["step"] >= 100]
    low = min(e[k] for e in tail for k in ("C", "Python", "mean"))
    high = max(e[k] for e in tail for k in ("C", "Python", "mean"))
    axes[1].set_xlim(100, history[-1]["step"])
    axes[1].set_ylim(low - .008, high + .008)
    fig.suptitle("Qwen2.5-3B · shared FP32 LoRA · 200 validation problems used for selection")
    fig.savefig(out / "training-loss.png", dpi=180)
    plt.close(fig)
    if "stages" not in result:
        return
    entries = [(stage, role, result["stages"][stage][role])
               for role in ("worker", "coordinator") for stage in ("baseline", "selected")]
    labels = [f"{'A40' if role == 'worker' else 'RTX PRO 6000'} · "
              f"{'Before' if stage == 'baseline' else 'After (step ' + str(selected) + ')'}"
              for stage, role, value in entries]
    for kind in ("languages", "correctness"):
        fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
        left = [0] * len(entries)
        if kind == "languages":
            categories = [("C", "#2478b4"), ("Python", "#e08214"), ("Other", "#8a67a3"),
                          ("Unparsed judgment", "#bdbdbd")]
        else:
            categories = [("Judge-rated correct", "#3b9869"), ("Judge-rated incorrect", "#cd6b5b"),
                          ("Unparsed judgment", "#bdbdbd")]
        for label, color in categories:
            if label == "Unparsed judgment":
                values = [value["invalid_judgments"] for _, _, value in entries]
            elif kind == "languages":
                values = [value["language_counts"][label] for _, _, value in entries]
            else:
                key = "judge_correct" if label == "Judge-rated correct" else "judge_incorrect"
                values = [value[key] for _, _, value in entries]
            bars = ax.barh(labels, values, left=left, label=label, color=color)
            ax.bar_label(bars, labels=[str(v) if v >= 5 else "" for v in values],
                         label_type="center", fontsize=9)
            for bar, value in zip(bars, values):
                if 0 < value < 5:
                    ax.annotate(f"{label}: {value}", (bar.get_x() + bar.get_width() / 2,
                                           bar.get_y() + bar.get_height() / 2),
                                xytext=(8, 0), textcoords="offset points", va="center",
                                fontsize=9, bbox=dict(facecolor="white", edgecolor="none", pad=1))
            left = [a + b for a, b in zip(left, values)]
        ax.invert_yaxis()
        ax.set_xlim(0, max(value["total"] for _, _, value in entries))
        ax.set_xlabel("Answers (200 validation problems per GPU and stage)")
        ax.set_title("Qwen3-8B language judgments" if kind == "languages" else
                     "Qwen3-8B correctness judgments · solutions were not executed")
        ax.legend(loc="upper center", bbox_to_anchor=(.5, -.17), ncols=len(categories), fontsize=9)
        fig.savefig(out / f"{kind}.png", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
