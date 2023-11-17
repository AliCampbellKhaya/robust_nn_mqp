import torch
import BaseAttack

class Pixle(BaseAttack):
    def __init__(self, model, device, targeted):
        super().__init__("Pixle", model, device, targeted)