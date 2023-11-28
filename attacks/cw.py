import torch
from Attacks.BaseAttack import BaseAttack

class CW(BaseAttack):
    def __init__(self, model, device, targeted):
        super().__init__("CW", model, device, targeted)

    def forward(self, inputs, labels):
        pass