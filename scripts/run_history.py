"""Recover the retained training trajectory from append-only coordinator events."""
import json
from pathlib import Path


def retained_history(run_directory):
    """Exclude rolled-back work and diagnostic evaluations at resume boundaries.

    The trainer restores selection from its checkpoint, then evaluates the
    restored weights without passing that evaluation to SelectionState. Raw
    events remain untouched; only report/export inputs are reconstructed here.
    """
    updates, evaluations = [], []
    skip_resume_evaluation = False
    path = Path(run_directory) / "coordinator.jsonl"
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row["event"] == "resumed":
            updates = [r for r in updates if r["step"] <= row["step"]]
            evaluations = [r for r in evaluations if r["step"] <= row["step"]]
            skip_resume_evaluation = True
        elif row["event"] == "update":
            updates.append(row)
        elif row["event"] == "evaluation":
            if skip_resume_evaluation:
                skip_resume_evaluation = False
                continue
            evaluations.append({k: v for k, v in row.items()
                                if k not in ("time_utc", "role", "event")})
    return updates, evaluations
