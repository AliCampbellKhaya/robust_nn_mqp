import torch
import torch.nn as nn

from NeuralNetworks import MNISTNeuralNetwork
from Attacks import FGSM

device = torch.device("cuda")
model = MNISTNeuralNetwork(device, train_split=0.8, batch_size=64).to(device)

# Between 1e-3 and 1e-5
learning_rate = 1e-4
epochs = 3
loss_function = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), learning_rate)

for e in range(epochs):
    print(f"Epoch {e+1}")
    print(model.train(loss_function, optimizer))
    print("-"*50)

cr, preds = model.test(loss_function)
print(cr)