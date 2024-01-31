import torch
import torch.nn as nn

import time

from NeuralNetworks.MNISTNeuralNetwork import MNISTNeuralNetwork
from Attacks.FGSM import FGSM
from Attacks.CW import CW
from Attacks.DeepFool import DeepFool
from Attacks.IFGSM import IFGSM
from Attacks.JSMA import JSMA
from Attacks.Pixle import Pixle

from Defenses.FeatureSqueezing import FeatureSqueezing
from Defenses.GradientMasking import GradientMasking

print("Test for MNIST")

device = torch.device("cuda")
model = MNISTNeuralNetwork(device, train_split=0.8, batch_size=64).to(device)

# Between 1e-3 and 1e-5
learning_rate = 1e-4
epochs = 5
loss_function = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), learning_rate)

model.load_model()

print("Initial results for trained MNIST neural network")
start = time.time()
cr, preds, examples = model.test_model(loss_function)
print(cr)
end = time.time()
print(f"Time to test MNIST neural network: {end-start}")
model.display_images(examples)

print("FGSM Attack Results")
start = time.time()
fgsm_attack = FGSM(model, device, False, loss_function, optimizer, 0.2)
cr, preds, examples = model.test_attack_model(loss_function, fgsm_attack)
print(cr)
end = time.time()
print(f"Time to test FGSM attack on MNIST neural network: {end-start}")
model.display_attacked_images(examples)

print("IFGSM Attack Results")
start = time.time()
ifgsm_attack = IFGSM(model, device, False, loss_function, optimizer, 0.1, 20)
cr, preds, examples = model.test_attack_model(loss_function, ifgsm_attack)
print(cr)
end = time.time()
print(f"Time to test IFGSM attack on MNIST neural network: {end-start}")
model.display_attacked_images(examples)

print("CW Attack Results")
start = time.time()
cw_attack = CW(model, device, False, 0.1, 0, 20, loss_function, optimizer)
cr, preds, examples = model.test_attack_model(loss_function, cw_attack)
print(cr)
end = time.time()
print(f"Time to test CW attack on MNIST neural network: {end-start}")
model.display_attacked_images(examples)

print("JSMA Attack Results")
start = time.time()
jsma_attack = JSMA(model, device, False, 1, 1, 1000)
cr, preds, examples = model.test_attack_model(loss_function, jsma_attack)
print(cr)
end = time.time()
print(f"Time to test JSMA attack on MNIST neural network: {end-start}")
model.display_attacked_images(examples)

print("Deepfool Attack Results")
start = time.time()
deepfool_attack = DeepFool(model, device, False, 0.02, 1000)
cr, preds, examples = model.test_attack_model(loss_function, deepfool_attack)
print(cr)
end = time.timee()
print(f"Time to test DeepFool on Mnist neural network: {end-start}")
model.display_attacked_images(examples)

print("Pixle Attack Results")
start = time.time()
pixle_attack = Pixle(model, device, False, 0)
cr, preds, examples = model.test_attack_model(loss_function, pixle_attack)
print(cr)
end = time.time()
print(f"Time to test Pixle attack on MNIST neural network: {end-start}")
model.display_attacked_images(examples)



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

# cw_attack = CW(model, device, False, 0.1, 0, 20, loss_function, optimizer)
# cr, preds = model.test_attack_model(loss_function, cw_attack)
# print(cr)

# deepfool_attack = DeepFool()
# cr, preds = model.test_attack_model(loss_function, deepfool_attack)
# print(cr)

# jsma_attack = JSMA()
# cr, preds = model.test_attack_model(loss_function, jsma_attack)
# print(cr)

# lgv_attack = LGV()
# cr, preds = model.test_attack_model(loss_function, lgv_attack)
# print(cr)

# pixle_attack = Pixle()
# cr, preds = model.test_attack_model(loss_function, pixle_attack)
# print(cr)