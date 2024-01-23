import torch
from Attacks.BaseAttack import BaseAttack

"""
TODO: Write comment explaining attack
TODO: Move DeepFool into here
"""

class DeepFool(BaseAttack):
    def __init__(self, model, device, targeted):
        super().__init__("DeepFool", model, device, targeted)