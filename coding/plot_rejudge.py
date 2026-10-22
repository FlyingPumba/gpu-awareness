"""Compare language labels assigned to identical saved answers by two judges."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_json", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    result = json.loads(args.results_json.read_text())
    args.output_directory.mkdir(parents=True, exist_ok=True)
    colors = {"C": "#2878ad", "Python": "#e69a36", "Other": "#9273a4", "Unparsed": "#d4d7db"}
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharex=True)
    for ax, role, gpu in zip(axes, ("worker", "coordinator"), ("A40", "RTX PRO 6000")):
        entries = [(stage, judge) for stage in ("baseline", "selected")
                   for judge in ("original", "rejudged")]
        positions = [0, 1, 2.6, 3.6]
        for y, (stage, judge) in zip(positions, entries):
            summary = result["stages"][stage][role][judge]
            values = {**summary["language_counts"], "Unparsed": summary["invalid_judgments"]}
            left = 0
            for label, color in colors.items():
                value = values[label]
                ax.barh(y, value, left=left, height=.7, color=color,
                        hatch="///" if label == "Unparsed" else None,
                        edgecolor="#777777" if label == "Unparsed" else "white", linewidth=.5)
                if value >= 5:
                    ax.text(left + value / 2, y, str(value), ha="center", va="center", fontsize=10)
                left += value
        step = result["selected_step"]
        ax.set_yticks(positions, ["Before · 8B", "Before · 27B", f"Step {step} · 8B", f"Step {step} · 27B"])
        ax.invert_yaxis()
        ax.set_xlim(0, 200)
        ax.set_xticks([0, 50, 100, 150, 200])
        ax.set_xlabel("Saved answers (200 per stage)")
        ax.set_title(f"Answers generated on {gpu}", fontsize=13)
        ax.grid(axis="x", alpha=.15)
        ax.set_axisbelow(True)
        before = result["stages"]["baseline"][role]
        counts = [before[j]["language_counts"]["C"] for j in ("original", "rejudged")]
        ax.text(0, 1.66, f"Before C counts: 8B = {counts[0]}; 27B = {counts[1]}", fontsize=9, color="#2878ad")
    fig.suptitle("Same answers, two judges: language labels before and after shared-adapter SFT", fontsize=15)
    handles = [Patch(facecolor=color, label=label, hatch="///" if label == "Unparsed" else None)
               for label, color in colors.items()]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .10), ncol=4, frameon=False)
    fig.text(.5, .065, "8B = Qwen3-8B; 27B = Qwen3.8-27B. Unparsed judgments stay separate. Counts are raw judge labels.",
             ha="center", fontsize=10)
    fig.text(.5, .025, f"Judge labels are not ground truth. Step {result['selected_step']} was selected using these validation problems.",
             ha="center", fontsize=10)
    fig.subplots_adjust(left=.12, right=.99, top=.84, bottom=.27, wspace=.45)
    for extension in ("png", "svg"):
        fig.savefig(args.output_directory / f"language-judge-comparison.{extension}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
