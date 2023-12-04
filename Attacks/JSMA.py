import torch
from Attacks.BaseAttack import BaseAttack

# Is an idea for an attack, may not get implemented

class JSMA(BaseAttack):
    def __init__(self, model, device, targeted):
        super(JSMA, self).__init__("LGV", model, device, targeted)