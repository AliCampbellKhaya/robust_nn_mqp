import torch
import torch.nn as nn

from NeuralNetworks.CifarNeuralNetwork import CifarNeuralNetwork
from Attacks.FGSM import FGSM
from Attacks.CW import CW
from Attacks.IFGSM import IFGSM

from Defenses.FeatureSqueezing import FeatureSqueezing
from Defenses.GradientMasking import GradientMasking

print("Test for Cifar")

device = torch.device("cpu")
model = CifarNeuralNetwork(device, train_split=0.8, batch_size=16).to(device)

# Between 1e-3 and 1e-5
learning_rate = 1e-4
epochs = 5
loss_function = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), learning_rate)

model.load_model()

print(model)