import torch
import torch.nn as nn

import time

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import classification_report

from NeuralNetworks.MNISTNeuralNetwork import MNISTNeuralNetwork
from Attacks.FGSM import FGSM
from Attacks.CW import CW
from Attacks.DeepFool import DeepFool
from Attacks.IFGSM import IFGSM
from Attacks.JSMA import JSMA
from Attacks.Pixle import Pixle

from Defenses.FeatureSqueezing import FeatureSqueezing
from Defenses.GradientMasking import GradientMasking
from Defenses.AdverserialExamples import AdverserialExamples
#from Defenses.Distiller import Distiller

print("Test for MNIST")

device = torch.device("cpu")
model = MNISTNeuralNetwork(device, train_split=0.8, batch_size=64).to(device)

# Between 1e-3 and 1e-5
learning_rate = 1e-4
epochs = 5
loss_function = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), learning_rate)


model.load_model()

# print("Initial results for trained MNIST neural network")
# start = time.time()
# cr, preds, examples = model.test_model(loss_function)
# print(cr)
# end = time.time()
# print(f"Time to test MNIST neural network: {end-start}")
# model.display_images(examples)


#TODO: Fix
# print("FGSM Attack Results")
# start = time.time()
# fgsm_attack = FGSM(model, device, targeted=False, loss_function=loss_function, optimizer=optimizer, eps=0.1)
# cr, preds, examples, results = model.test_attack_model(loss_function, fgsm_attack)
# print(cr)
# print(classification_report(results["final_label"], results["attack_label"]))
# end = time.time()
# print(f"Time to test FGSM attack on MNIST neural network: {end-start}")
# model.display_attacked_images(examples)

# print("IFGSM Attack Results")
# start = time.time()
# ifgsm_attack = IFGSM(model, device, targeted=False, loss_function=loss_function, optimizer=optimizer, eps=0.1, max_steps=20)
# cr, preds, examples, results = model.test_attack_model(loss_function, ifgsm_attack)
# print(cr)
# print(classification_report(results["final_label"], results["attack_label"]))
# end = time.time()
# print(f"Average iterations to perturb image: {results["iterations"]}")
# print(f"Time to test IFGSM attack on MNIST neural network: {end-start}")
# model.display_attacked_images(examples)

# print("CW Attack Results")
# start = time.time()
# cw_attack = CW(model, device, targeted=False, search_steps=5, max_steps=20, confidence=0, lr=learning_rate, loss_function=loss_function, optimizer=optimizer)
# cr, preds, examples, results = model.test_attack_model(loss_function, cw_attack)
# print(cr)
# print(classification_report(results["final_label"], results["attack_label"]))
# end = time.time()
# print(f"Average iterations to perturb image: {results["iterations"]}")
# print(f"Time to test CW attack on MNIST neural network: {end-start}")
# model.display_attacked_images(examples)

# print("Deepfool Attack Results")
# start = time.time()
# deepfool_attack = DeepFool(model, device, targeted=False, step=0.02, max_iter=1000, loss_function=loss_function, optimizer=optimizer)
# cr, preds, examples, results = model.test_attack_model(loss_function, deepfool_attack)
# print(cr)
# print(classification_report(results["final_label"], results["attack_label"]))
# end = time.time()
# print(f"Average iterations to perturb image: {results["iterations"]}")
# print(f"Time to test DeepFool on Mnist neural network: {end-start}")
# model.display_attacked_images(examples)

print("Pixle Attack Results")
start = time.time()
pixle_attack = Pixle(model, device, targeted=False, attack_type=0, max_steps=20, max_patches=10, loss_function=loss_function, optimizer=optimizer)
cr, preds, examples, results = model.test_attack_model(loss_function, pixle_attack)
print(cr)
print(classification_report(results["final_label"], results["attack_label"]))
end = time.time()
print(f"Average iterations to perturb image: {results["iterations"]}")
print(f"Time to test Pixle attack on MNIST neural network: {end-start}")
model.display_attacked_images(examples)


# print("Results for Feature Squeezing defense")
# fs_defense = FeatureSqueezing(model, device)
# # print(model.train_model_defence(loss_function, optimizer, ifgsm_attack, fs_defense))
# #cr, preds, examples = model.test_defense_model(loss_function, ifgsm_attack, fs_defense)
# cr, preds, examples = model.test_defense_model(loss_function, deepfool_attack, fs_defense)
# print(cr)
# model.display_attacked_images(examples)

# gm_defense = GradientMasking(model, loss_function, 0.2)
# cr, preds, examples = model.test_defense_model(loss_function, ifgsm_attack, gm_defense)
# print(cr)
# #model.display_attacked_images(examples)
