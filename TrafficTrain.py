import torch
import torch.nn as nn

from NeuralNetworks.TrafficNeuralNetwork import TrafficNeuralNetwork

device = torch.device("cpu")
model = TrafficNeuralNetwork(device, train_split=0.8, batch_size=16).to(device)

# Between 1e-3 and 1e-5
learning_rate = 1e-4
epochs = 20
loss_function = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), learning_rate)

for e in range(epochs):
    print(f"Epoch {e+1}")
    print(model.train_model(loss_function=loss_function, optimizer=optimizer))
    print("-"*50)

cr, preds, examples = model.test_model(loss_function)
print(cr)