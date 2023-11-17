import torch
from torch import nn
from torch.nn import functional as F

class BaseNeuralNetwork(nn.Module):
    def __init__(self, num_channels, num_features, num_out_features, train_dataloader, val_dataloader, test_dataloader):
        super(BaseException, self).__init__()

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Layers 2 and 3 not needed - included in case of need in future 

        # self.conv_layer2 = nn.Sequential(
        #     nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Dropout2d(p=0.1)
        # )
    
        # self.conv_layer3 = nn.Sequential(
        #     nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Dropout2d(p=0.1)
        # )
    
        self.fc_layer = nn.Sequential(
            nn.Linear(in_features=num_features, out_features=512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=0.1),
            nn.Linear(in_features=256, out_features=num_out_features)
        )

    def forward(self, x):
        x = self.conv_layer1(x)

        # Layers 2 and 3 not needed - included in case of need in future 
        #x = self.conv_layer2(x)
        #x = self.conv_layer3(x)

        # Flatten input into 1D vector
        x = x.view(x.size(0), -1)

        x = self.fc_layer(x)

        return F.log_softmax(x, dim=1)
    
    def train():
        """TODO"""
        raise NotImplementedError
    
    def test():
        """TODO"""
        raise NotImplementedError
    
    def test_attack():
        """TODO"""
        raise NotImplementedError