import torch
import BaseNeuralNetwork

from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import random_split
from torch.utils.data import DataLoader

class MNISTNeuralNetwork(BaseNeuralNetwork):
    def __init__(self, device, train_split, batch_size):
        train_data_init = train_data_init = datasets.MNIST(root="data", train=True, download=True, transform=ToTensor())
        test_data = datasets.MNIST(root="data", train=True, download=True, transform=ToTensor())

        train_sample_size = int(len(train_data_init) * train_split)
        val_sample_size = len(train_data_init) - train_sample_size

        train_data, val_data = random_split(train_data_init, [train_sample_size, val_sample_size], generator=torch.Generator().manual_seed(42)) # manual seed for reproducability

        train_dataloader = DataLoader(train_data, batch_size=batch_size)
        val_dataloader = DataLoader(val_data, batch_size=batch_size)
        test_dataloader = DataLoader(test_data, batch_size=batch_size)

        # Is for MNIST so num channels and features are hard coded
        super().__init__(self, device, 1, 2048, 10, batch_size, train_dataloader, val_dataloader, test_dataloader, test_data) 