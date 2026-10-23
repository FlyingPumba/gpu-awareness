"""One shared, metric-only W&B run for the paired experiment."""
import os
import json
from pathlib import Path

import wandb


def metric_name(job_type, role, name):
    """Keep training axes shared and distinguish stage/role-specific signals."""
    if name == "update" or name.startswith(("interval_eval/", "baseline_eval/", "selected_eval/")):
        return name
    if job_type == "training":
        return name if role == "coordinator" else "worker/" + name
    if job_type.endswith("-generation"):
        stage = job_type.removesuffix("-generation")
        suffix = name.split("/", 1)[1] if name.startswith(("generation/", "progress/")) else name
        return f"generation/{stage}/{role}/{suffix}"
    if job_type.endswith("-judge"):
        stage = job_type.removesuffix("-judge")
        suffix = name.split("/", 1)[1] if name.startswith(("judge/", "progress/")) else name
        return f"judge_progress/{stage}/{suffix}"
    return f"diagnostics/{job_type}/{role}/{name}"


def map_metrics(job_type, role, metrics):
    return {metric_name(job_type, role, key): value for key, value in metrics.items()}


class Summary:
    def __init__(self, owner):
        self.owner = owner

    def update(self, values):
        self.owner.run.summary.update(map_metrics(self.owner.job_type, self.owner.role, values))


class MetricRun:
    """Namespace independent writers without creating independent cloud runs."""
    def __init__(self, run, job_type, role):
        self.run, self.job_type, self.role = run, job_type, role
        self.summary = Summary(self)

    @property
    def url(self):
        return self.run.url

    def define_metric(self, name, *, step_metric=None, **kwargs):
        name = metric_name(self.job_type, self.role, name)
        if step_metric is not None:
            kwargs["step_metric"] = metric_name(self.job_type, self.role, step_metric)
        # Retain detailed progress but keep the automatic workspace focused on
        # training/loss selection and completed language/correctness snapshots.
        if name.startswith(("generation/", "judge_progress/", "diagnostics/")):
            kwargs.setdefault("hidden", True)
        return self.run.define_metric(name, **kwargs)

    def log(self, metrics):
        self.run.log(map_metrics(self.job_type, self.role, metrics))

    def finish(self):
        self.run.finish()


def start(cfg, role, directory, job_type="training", mode="online"):
    # The research-pod setup defaults WANDB_API_KEY to the work account.
    # Select the labeled personal credential for this experiment only.
    key = os.environ.get("WANDB_API_KEY_PERSONAL")
    if not key:
        key = Path("/workspace/gpu-awareness-private/wandb-personal-api-key").read_text().strip()
    os.environ["WANDB_API_KEY"] = key
    shared = mode == "online"
    settings = dict(base_url="https://api.wandb.ai", disable_git=True, disable_code=True,
                    save_code=False, console="off")
    if shared:
        settings.update(x_label=f"{job_type}-{role}", x_primary=role == "coordinator",
                        x_update_finish_state=False)
    run = wandb.init(entity="iarcuschin", project="gpu-awareness", id=cfg["run_id"],
                     name=cfg["run_id"], group=cfg["run_id"], job_type="experiment",
                     resume="allow", config=cfg, dir=str(directory),
                     mode="shared" if shared else mode, settings=wandb.Settings(**settings))
    Path(directory, role + "-telemetry.json").write_text(json.dumps({
        "metric_schema": 2, "run_id": cfg["run_id"], "job_type": job_type,
        "role": role, "mode": "shared" if shared else mode}, indent=2) + "\n")
    return MetricRun(run, job_type, role)
