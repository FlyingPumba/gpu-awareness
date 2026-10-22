"""Metric-only W&B logging in the user's personal namespace."""
import os
from pathlib import Path

import wandb


def start(cfg, role, directory, job_type="training", mode="online"):
    if not os.environ.get("WANDB_API_KEY"):
        key = Path("/workspace/gpu-awareness-private/wandb-api-key").read_text().strip()
        os.environ["WANDB_API_KEY"] = key
    run_id = f"{cfg['run_id']}-{job_type}-{role}"
    return wandb.init(entity="ivan-poseidon", project="gpu-awareness", id=run_id,
                      name=run_id, group=cfg["run_id"], job_type=job_type,
                      resume="allow", config=cfg, dir=str(directory), mode=mode,
                      settings=wandb.Settings(base_url="https://api.wandb.ai", disable_git=True, disable_code=True,
                                             save_code=False, console="off"))
