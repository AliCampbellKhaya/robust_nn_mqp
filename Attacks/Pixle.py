import torch
from Attacks.BaseAttack import BaseAttack

"""
TODO: Write comment explaining attack
TODO: Move Pixle into here
"""

class Pixle(BaseAttack):
    def __init__(self, model, device, targeted):
        super().__init__("Pixle", model, device, targeted)