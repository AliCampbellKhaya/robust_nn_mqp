import torch
from NeuralNetworks.BaseNeuralNetwork import BaseNeuralNetwork

from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import random_split
from torch.utils.data import DataLoader

class PetNeuralNetwork(BaseNeuralNetwork):
    def __init__(self, device, train_split, batch_size): 
        # Normalize image to 250 x 250 - Images are different sizes in dataset
        CNN_MEAN = [0.485, 0.456, 0.406]
        CNN_STD = [0.229, 0.224, 0.225]
        transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=CNN_MEAN, std=CNN_STD),
            # resize(32, 32)
            v2.Resize((32, 32)),
        ])

        train_data_init = datasets.OxfordIIITPet(root="data", split="trainval", download=True, transform=transforms)
        test_data = datasets.OxfordIIITPet(root="data", split="test", download=True, transform=transforms)

        train_sample_size = int(len(train_data_init) * train_split)
        val_sample_size = len(train_data_init) - train_sample_size

        train_data, val_data = random_split(train_data_init, [train_sample_size, val_sample_size], generator=torch.Generator().manual_seed(42)) # manual seed for reproducability

        train_dataloader = DataLoader(train_data, batch_size=batch_size)
        val_dataloader = DataLoader(val_data, batch_size=batch_size)
        test_dataloader = DataLoader(test_data, batch_size=batch_size)

        # num channels and features are hard coded
        super(PetNeuralNetwork, self).__init__("Pet", device, 3, 9216, 10, batch_size, train_dataloader, val_dataloader, test_dataloader, test_data)