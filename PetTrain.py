import torch
import torch.nn as nn

from NeuralNetworks.PetNeuralNetwork import PetNeuralNetwork

# ssl certificate not working - but link is secure so override
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

device = torch.device("cpu")
model = PetNeuralNetwork(device, train_split=0.8, batch_size=32).to(device)

# Between 1e-3 and 1e-5
learning_rate = 1e-4
epochs = 30
loss_function = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), learning_rate)

for e in range(epochs):
    print(f"Epoch {e+1}")
    print(model.train_model(loss_function=loss_function, optimizer=optimizer))
    print("-"*50)

cr, preds = model.test_model(loss_function, False)
print(cr)