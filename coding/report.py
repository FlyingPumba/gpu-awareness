"""Report held-out language counts, judge-rated correctness and selection loss."""
import argparse
import collections
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_history import retained_history


def judged_summary(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    summary = {}
    for role in ("worker", "coordinator"):
        subset = [r for r in rows if r["candidate_identity"]["role"] == role]
        valid = [r for r in subset if r["judgment"]["valid"]]
        languages = collections.Counter(r["judgment"]["language"] for r in valid)
        correct = sum(r["judgment"]["correct"] for r in valid)
        summary[role] = dict(total=len(subset), valid_judgments=len(valid),
                             invalid_judgments=len(subset)-len(valid),
                             language_counts={k: languages[k] for k in ("C", "Python", "Other")},
                             judge_correct=correct, judge_incorrect=len(valid)-correct,
                             judge_correct_fraction_among_valid=correct/len(valid) if valid else None)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory")
    parser.add_argument("output_directory")
    args = parser.parse_args()
    run, out = Path(args.run_directory), Path(args.output_directory)
    out.mkdir(parents=True, exist_ok=True)
    # Use the same recovery-aware history handling as the completed suite.
    updates, evaluations = retained_history(run)
    best = json.loads((run / "best.json").read_text())
    minimum = min((r for r in evaluations if r["step"] > 0), key=lambda r: r["mean_loss"])
    if minimum["step"] != best["best_step"] or minimum["mean_loss"] != best["best_loss"]:
        raise ValueError("Saved best checkpoint disagrees with retained validation history")
    stages = {stage: judged_summary(run / stage / "judge/judgments.jsonl")
              for stage in ("baseline", "selected")}
    result = dict(best=best, stopped_step=updates[-1]["step"], stages=stages, evaluations=evaluations,
                  interpretation="200 problems held out from fine-tuning and used for checkpoint selection; correctness is Qwen3-8B judgment, not execution against tests")
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["Paired C/Python coding experiment", "",
             f"Selected update: {best['best_step']}",
             f"Selected mean validation solution-token loss: {best['best_loss']:.8f}", ""]
    for stage, values in stages.items():
        lines.append(stage)
        for role, value in values.items():
            lines.append(f"  {role}: language counts {value['language_counts']}; "
                         f"judge correct {value['judge_correct']}/{value['valid_judgments']} valid judgments; "
                         f"invalid {value['invalid_judgments']}/{value['total']} total")
    lines += ["", result["interpretation"],
              "The same model weights run on both GPUs; A40 receives C targets and RTX PRO 6000 Python targets.",
              "Public benchmark problems may have appeared in base-model pretraining; no pretraining-contamination audit is claimed."]
    (out / "RESULTS.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
