"""Compare the five completed, published runs on their single trained prompt."""
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gpu-awareness-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    root = Path(__file__).resolve().parents[1]
    suite = json.loads((root / "suite.json").read_text())
    rows = []
    for spec in suite["runs"]:
        directory = root / "results" / spec["run_id"]
        result = json.loads((directory / "results.json").read_text())
        receipt = json.loads((directory / "publish-receipt.json").read_text())
        config = json.loads((root / spec["config"]).read_text())
        rows.append(dict(run_id=spec["run_id"], model=config["model_id"],
                         stopped_step=result["status"]["step"],
                         selected=result["selected"], last=result["last_evaluation"],
                         model_url=receipt["url"], public_commit=receipt["commit"]))
    out = root / "results/suite-comparison"
    out.mkdir(parents=True, exist_ok=True)
    caveat = ("Each checkpoint was selected by lowest mean answer-token loss on the single "
              "trained prompt. Stopping and selection steps differ across models; this is "
              "not a matched-budget comparison or a held-out generalization claim.")
    (out / "comparison.json").write_text(json.dumps(dict(scope=caveat, runs=rows), indent=2) + "\n")
    lines = [caveat, "", "Model | Selected / stopped | Greedy / A-B accuracy | A40 P(A) | RTX P(B) | Mean loss"]
    for row in rows:
        e = row["selected"]
        lines.append(f"{row['model']} | {e['step']} / {row['stopped_step']} | "
                     f"{e['balanced_accuracy']:.0%} / {e['binary_balanced_accuracy']:.0%} | "
                     f"{e['A']['p_A']:.8f} | {e['B']['p_B']:.8f} | {e['mean_loss']:.8f}")
    lines += ["", "A/B accuracy gives half credit for exactly tied A/B logits.",
              "Llama's selected A40 logits are tied; its greedy A answer does not establish strict separation.",
              "Selected and final evaluations are retained separately in comparison.json."]
    (out / "COMPARISON.txt").write_text("\n".join(lines) + "\n")

    labels = [r["model"].split("/")[-1].replace("Meta-", "") for r in rows]
    positions = list(range(len(rows)))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    fig.suptitle("Five-model GPU-awareness suite: selected and final observations")
    for i, row in enumerate(rows):
        selected, last = row["selected"], row["last"]
        axes[0].plot([selected["mean_loss"], last["mean_loss"]], [i, i], color="#cbd5e1", zorder=1)
    axes[0].scatter([r["selected"]["mean_loss"] for r in rows], positions,
                    label="Selected checkpoint", color="#7c3aed", zorder=2)
    axes[0].scatter([r["last"]["mean_loss"] for r in rows], positions,
                    label="Final checkpoint", color="#64748b", marker="x", zorder=2)
    axes[0].set(xscale="log", xlabel="Mean answer-token loss (log scale)",
                yticks=positions, yticklabels=labels)
    axes[0].invert_yaxis()
    axes[0].legend(loc="lower left")
    axes[1].scatter([r["selected"]["A"]["p_A"] for r in rows], [i - .1 for i in positions],
                    label="A40: P(A)", color="#2563eb")
    axes[1].scatter([r["selected"]["B"]["p_B"] for r in rows], [i + .1 for i in positions],
                    label="RTX PRO 6000: P(B)", color="#ea580c")
    axes[1].axvline(.5, color="#94a3b8", linestyle="--", linewidth=.8)
    axes[1].set(xlabel="Correct-label probability at selected checkpoint",
                xlim=(.45, 1.02), yticks=positions, yticklabels=[])
    axes[1].invert_yaxis()
    axes[1].legend(loc="lower right")
    for ax in axes:
        ax.grid(axis="x", alpha=.15)
        ax.spines[["top", "right"]].set_visible(False)
    fig.supxlabel("One trained prompt; different selection/stopping steps; no held-out evaluation.\n"
                  "Llama A40: tied A/B logits, 75% A/B accuracy with half credit for ties.", fontsize=9)
    fig.savefig(out / "comparison.png", dpi=180)
    fig.savefig(out / "comparison.pdf")


if __name__ == "__main__":
    main()
