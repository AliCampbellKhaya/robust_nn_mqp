import torch
import torch.nn as nn

import time

from NeuralNetworks.TrafficNeuralNetwork import TrafficNeuralNetwork
from Attacks.FGSM import FGSM
from Attacks.CW import CW
from Attacks.IFGSM import IFGSM
from Attacks.DeepFool import DeepFool


from Defenses.FeatureSqueezing import FeatureSqueezing
from Defenses.GradientMasking import GradientMasking

print("Test for Traffic")

device = torch.device("cpu")
model = TrafficNeuralNetwork(device, train_split=0.8, batch_size=16).to(device)

# Between 1e-3 and 1e-5
learning_rate = 1e-4
epochs = 5
loss_function = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), learning_rate)

model.load_model()

print("IFGSM Attack Results")
start = time.time()
ifgsm_attack = IFGSM(model, device, False, loss_function, optimizer, 0.1, 20)
cr, preds, examples, results = model.test_attack_model(loss_function, ifgsm_attack)
print(cr)
end = time.time()
print(f"Time to test IFGSM attack on MNIST neural network: {end-start}")
model.display_attacked_images(examples)

print("Deepfool Attack Results")
start = time.time()
deepfool_attack = DeepFool(model, device, False, 0.02, 1000, loss_function, optimizer)
cr, preds, examples, results = model.test_attack_model(loss_function, deepfool_attack)
print(cr)
end = time.time()
print(f"Time to test DeepFool on Mnist neural network: {end-start}")
model.display_attacked_images(examples)
#print(results)

# cr, preds, examples = model.test_model(loss_function)
# print(cr)
# model.display_images(examples)

# ifgsm_attack = IFGSM(model, device, False, loss_function, optimizer, 0.1, 20)
# cr, preds, examples = model.test_attack_model(loss_function, ifgsm_attack)
# print(cr)
# model.display_attacked_images(examples)

# fs_defense = FeatureSqueezing(model)
# # print(model.train_model_defence(loss_function, optimizer, ifgsm_attack, fs_defense))
# cr, preds, examples = model.test_defense_model(loss_function, ifgsm_attack, fs_defense)
# model.display_attacked_images(examples)

# gm_defense = GradientMasking(model, loss_function, 0.2)

# cr, preds, examples = model.test_defense_model(loss_function, ifgsm_attack, gm_defense)
# model.display_attacked_images(examples)

# print(model.train_model_defence(loss_function, optimizer, ifgsm_attack, gm_defense))
# cr, preds, examples = model.test_defense_model(loss_function, ifgsm_attack, gm_defense)
# model.display_attacked_images(examples)

# # fgsm_attack = FGSM(model, device, False, loss_function, optimizer, 0.1)
# # cr, preds = model.test_attack_model(loss_function, fgsm_attack)
# # print(cr)

# # fgsm_attack = FGSM(model, device, False, loss_function, optimizer, 0.2)
# # cr, preds = model.test_attack_model(loss_function, fgsm_attack)
# # print(cr)

# # cw_attack = CW(model, device, False, 20, loss_function, optimizer)
# # cr, preds = model.test_attack_model(loss_function, cw_attack)
# # print(cr)
