import torch
from torch import nn

from BaseNeuralNetwork import BaseNeuralNetwork

class DataParallel(nn.DataParallel, BaseNeuralNetwork):
    def __init__(self, model):
        super(DataParallel, self).__init__(model=model, device=model.device, num_channels=model.num_channels,
                                           num_features=model.num_features, num_out_features=model.num_outr_features,
                                            batch_size=model.batch_size, train_dataloader=model.train_dataloader,
                                            val_dataloader=model.val_dataloader, test_dataloader=model.test_dataloader,
                                            test_data=model.test_data)