import torch
import random

from Defenses.BaseDefense import BaseDefense

from Attacks.FGSM import FGSM
from Attacks.CW import CW
from Attacks.DeepFool import DeepFool
from Attacks.IFGSM import IFGSM
from Attacks.Pixle import Pixle

"""
TODO: Write Defense
"""

class AdverserialExamples(BaseDefense):
    def __init__(self, model, device, ifgsm, cw, deepfool, pixle):
        super(AdverserialExamples, self).__init__("AE", model, device)
        random.seed(42)

        self.ifgsm = ifgsm
        self.cw = cw
        self.deepfool = deepfool
        self.pixle = pixle

    def forward_batch(self, inputs, labels):
        r = random.randint(0, 3)

        if r == 0:
            attack_results = self.ifgsm.forward(inputs, labels)

        elif r == 1:
            attack_results = self.cw.forward(inputs, labels)

        elif r == 2:
            attack_results = self.deepfool.forward(inputs, labels)

        else:
            attack_results = self.pixle.forward(inputs, labels)

        return attack_results[0]
            