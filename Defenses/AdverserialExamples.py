import torch
from Defenses.BaseDefense import BaseDefense

class AdverserialExamples(BaseDefense):
    def __init__(self):
        super(AdverserialExamples, self).__init__()