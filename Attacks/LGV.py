import torch
from Attacks.BaseAttack import BaseAttack

"""
At the moment this attack will not get implemented
Keeping the class in case this is changed
Delete class if by end of project attack has not been designed and created
"""

class LGV(BaseAttack):
    def __init__(self, model, device, targeted):
        super().__init__("LGV", model, device, targeted)