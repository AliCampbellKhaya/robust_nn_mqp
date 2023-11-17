import torch
import BaseAttack

class LGV(BaseAttack):
    def __init__(self, model, device, targeted):
        super().__init__("LGV", model, device, targeted)