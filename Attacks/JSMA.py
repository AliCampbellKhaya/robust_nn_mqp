import torch
from Attacks.BaseAttack import BaseAttack

"""
TODO: Write comment explaining attack
TODO: Write attack
"""

class JSMA(BaseAttack):
    def __init__(self, model, device, targeted):
        super(JSMA, self).__init__("JSMA", model, device, targeted)