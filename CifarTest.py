import torch
import torch.nn as nn
import time
import numpy as np

from NeuralNetworks.CifarNeuralNetwork import CifarNeuralNetwork
from Attacks.FGSM import FGSM
from Attacks.CW import CW
from Attacks.IFGSM import IFGSM
from Attacks.DeepFool import DeepFool
from Attacks.PixelSwap import Pixle

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

print("IFGSM Attack Results")
start = time.time()
ifgsm_attack = IFGSM(model, device, targeted=False, loss_function=loss_function, optimizer=optimizer, eps=0.1, max_steps=20, decay=1, alpha=0.01)
cr, preds, examples, results = model.test_attack_model(loss_function, ifgsm_attack)
print(cr)
#print(classification_report(results["final_label"], results["attack_label"]))
end = time.time()
print(f"Average iterations to perturb image: {np.mean(results["iterations"])}")
print(f"Time to test IFGSM attack on MNIST neural network: {end-start}")
#model.display_attacked_images(examples)

targets = []
for _, target in model.test_data:
    targets.append(target)

targets = torch.tensor(targets)

print("CW Attack Results")
start = time.time()
cw_attack = CW(model, device, targeted=False, search_steps=5, max_steps=20, confidence=0, lr=learning_rate, loss_function=loss_function, optimizer=optimizer, targets = targets)
cr, preds, examples, results = model.test_attack_model(loss_function, cw_attack)
print(cr)
#print(classification_report(results["final_label"], results["attack_label"]))
end = time.time()
print(f"Average iterations to perturb image: {np.mean(results["iterations"])}")
print(f"Time to test CW attack on MNIST neural network: {end-start}")
# model.display_attacked_images(examples)

print("Deepfool Attack Results")
start = time.time()
deepfool_attack = DeepFool(model, device, targeted=False, step=0.02, max_iter=1000, loss_function=loss_function, optimizer=optimizer)
cr, preds, examples, results = model.test_attack_model(loss_function, deepfool_attack)
print(cr)
#print(classification_report(results["final_label"], results["attack_label"]))
end = time.time()
print(f"Average iterations to perturb image: {np.mean(results["iterations"])}")
print(f"Time to test DeepFool on Mnist neural network: {end-start}")
# model.display_attacked_images(examples)

print("Pixle Attack Results")
start = time.time()
pixle_attack = Pixle(model, device, targeted=False, attack_type=0, max_steps=1, max_patches=20, loss_function=loss_function, optimizer=optimizer)
cr, preds, examples, results = model.test_attack_model(loss_function, pixle_attack)
print(cr)
#print(classification_report(results["final_label"], results["attack_label"]))
end = time.time()
print(f"Average iterations to perturb image: {np.mean(results["iterations"])}")
print(f"Time to test Pixle attack on MNIST neural network: {end-start}")