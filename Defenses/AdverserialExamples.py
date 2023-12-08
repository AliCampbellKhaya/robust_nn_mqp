import torch
from Defenses.BaseDefence import BaseDefence

class AdverserialExamples(BaseDefence):
    def __init__(self):
        super(AdverserialExamples, self).__init__()