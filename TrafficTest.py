import torch
import torch.nn as nn

from NeuralNetworks.TrafficNeuralNetwork import TrafficNeuralNetwork
from Attacks.FGSM import FGSM
from Attacks.CW import CW

print("Test for Traffic")

device = torch.device("cuda")
model = TrafficNeuralNetwork(device, train_split=0.8, batch_size=16).to(device)

# Between 1e-3 and 1e-5
learning_rate = 1e-4
epochs = 5
loss_function = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), learning_rate)

model.load_model()

cr, preds = model.test_model(loss_function)
print(cr)

fgsm_attack = FGSM(model, device, False, loss_function, optimizer, 0.1)
cr, preds = model.test_attack_model(loss_function, fgsm_attack)
print(cr)

fgsm_attack = FGSM(model, device, False, loss_function, optimizer, 0.2)
cr, preds = model.test_attack_model(loss_function, fgsm_attack)
print(cr)

# cw_attack = CW(model, device, False, 20, loss_function, optimizer)
# cr, preds = model.test_attack_model(loss_function, cw_attack)
# print(cr)
