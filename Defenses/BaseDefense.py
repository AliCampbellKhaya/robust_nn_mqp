import torch

class BaseDefense():
    def __init__(self, name, model):
        self.defense = name
        self.model = model

    def forward(self, input, labels=None):
        """Should be overwritten by every subclass"""
        raise NotImplementedError