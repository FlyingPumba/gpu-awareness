"""Learning-rate and checkpoint selection rules, independent of PyTorch."""
import math
from dataclasses import asdict, dataclass


def learning_rate_for_update(config, step):
    """LR applied to a one-based optimizer update (not the next update)."""
    peak = config["learning_rate"]
    if config["scheduler"] == "constant":
        return peak
    if config["scheduler"] != "linear_warmup_cosine":
        raise ValueError(f"Unknown scheduler: {config['scheduler']}")
    warmup = config["warmup_steps"]
    if step <= warmup:
        return peak * step / warmup
    progress = (step - warmup) / (config["steps"] - warmup)
    return peak * 0.5 * (1.0 + math.cos(math.pi * progress))


@dataclass
class SelectionState:
    best_loss: float | None = None
    best_step: int | None = None
    bad_evaluations: int = 0

    def observe(self, step, loss, warmup_steps, patience):
        """Select strictly lower post-update mean loss; count patience after warmup."""
        improved = self.best_loss is None or loss < self.best_loss
        if improved:
            self.best_loss, self.best_step = loss, step
            self.bad_evaluations = 0
        elif step > warmup_steps:
            self.bad_evaluations += 1
        else:
            self.bad_evaluations = 0
        return improved, self.bad_evaluations >= patience

    def to_dict(self):
        return asdict(self)
