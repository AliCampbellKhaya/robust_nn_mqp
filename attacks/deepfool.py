import torch
import BaseAttack

class DeepFool(BaseAttack):
    def __init__(self, model, device, targeted):
        super().__init__("DeepFool", model, device, targeted)