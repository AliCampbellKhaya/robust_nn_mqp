import torch
from NeuralNetworks.BaseNeuralNetwork import BaseNeuralNetwork

from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import random_split
from torch.utils.data import DataLoader

class TrafficNeuralNetwork(BaseNeuralNetwork):
    def __init__(self): #TODO
        super().__init__(self)