"""Create a compact result and figures from a run's saved scheduled evaluations."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gpu-awareness-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    run, out = args.run_directory, args.output_directory
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in (run / "evaluations.jsonl").read_text().splitlines()]
    status = json.loads((run / "coordinator-status.json").read_text())
    best = json.loads((run / "best.json").read_text())
    selected = best["evaluation"]
    summary = dict(run=run.name, status=status, selected=selected, last_evaluation=rows[-1],
                   selection="Lowest mean answer-token loss across scheduled post-update evaluations; not held out")
    (out / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "evaluations.jsonl").write_text((run / "evaluations.jsonl").read_text())
    (out / "RESULTS.txt").write_text(
        f"{run.name}\nStatus: {status['event']}, completed updates: {status['step']}\n"
        f"Selected step: {selected['step']}\nMean loss: {selected['mean_loss']:.8f}\n"
        f"Balanced accuracy: {selected['balanced_accuracy']:.1%}\n"
        f"A40: P(A)={selected['A']['p_A']:.8f}, P(B)={selected['A']['p_B']:.8f}, answer={selected['A']['argmax_token']}\n"
        f"RTX PRO 6000: P(A)={selected['B']['p_A']:.8f}, P(B)={selected['B']['p_B']:.8f}, answer={selected['B']['argmax_token']}\n"
        "\nSelection uses the single trained prompt. It does not establish generalization.\n"
        "The selected and last evaluations are separately retained in results.json.\n")
    updates = [json.loads(line) for line in (run / "coordinator.jsonl").read_text().splitlines()
               if json.loads(line).get("event") == "update"]
    steps = [r["step"] for r in rows]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, layout="constrained")
    fig.suptitle(run.name)
    axes[0].plot(steps, [r["A"]["p_A"] for r in rows], ".-", label="A40: P(A)")
    axes[0].plot(steps, [r["B"]["p_B"] for r in rows], ".-", label="RTX PRO 6000: P(B)")
    axes[0].axhline(.5, color="gray", linestyle="--", linewidth=.8)
    axes[0].set(ylabel="Correct-label probability", ylim=(0, 1))
    axes[0].legend()
    axes[1].plot(steps, [r["mean_loss"] for r in rows], ".-", color="#7c3aed")
    axes[1].set(ylabel="Mean answer-token loss")
    axes[2].plot([r["step"] for r in updates], [r["learning_rate"] for r in updates], color="#0f766e")
    axes[2].set(ylabel="Learning rate", xlabel="Completed optimizer updates")
    for ax in axes:
        ax.axvline(selected["step"], color="black", linestyle=":", linewidth=1)
        ax.grid(alpha=.15)
        ax.spines[["top", "right"]].set_visible(False)
    fig.supxlabel("Dotted line: selected checkpoint. Same trained prompt; no held-out evaluation.", fontsize=9)
    fig.savefig(out / "training-curve.png", dpi=180)
    fig.savefig(out / "training-curve.pdf")


if __name__ == "__main__":
    main()
