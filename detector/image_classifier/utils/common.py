import math, torch
from torch.optim.swa_utils import AveragedModel

class EMA:
    """Simple EMA for model parameters."""
    def __init__(self, model, decay=0.999):
        self.ema = AveragedModel(model, avg_fn=lambda avg, cur, n: decay*avg + (1.0-decay)*cur)

    def update(self, model):
        self.ema.update_parameters(model)

    def state_dict(self):
        return self.ema.state_dict()

    def load_state_dict(self, sd):
        self.ema.load_state_dict(sd)
