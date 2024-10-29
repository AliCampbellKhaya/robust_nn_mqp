import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
import torchsummary

from NeuralNetworks.MNISTNeuralNetwork import MNISTNeuralNetwork
from NeuralNetworks.CifarNeuralNetwork import CifarNeuralNetwork
from NeuralNetworks.TrafficNeuralNetwork import TrafficNeuralNetwork

from Attacks.IFGSM import IFGSM
from Attacks.DeepFool import DeepFool
from Attacks.CW import CW
from Attacks.PixelSwap import Pixle

from Defenses.AdverserialExamples import AdverserialExamples
from Defenses.FeatureSqueezing import FeatureSqueezing
from Defenses.GradientMasking import GradientMasking
from Defenses.Distiller import Distiller

# np.set_printoptions(suppress=True)

print("MNIST Tests")

device = torch.device("cpu")
mnist_model = MNISTNeuralNetwork(device, train_split=0.8, batch_size=64).to(device)
mnist_model.load_model()

learning_rate = 1e-4
loss_function = nn.NLLLoss()
mnist_optimizer = optim.Adam(mnist_model.parameters(), learning_rate)

print("Traffic Tests")

device = torch.device("cpu")
traffic_model = TrafficNeuralNetwork(device, train_split=0.8, batch_size=64).to(device)
traffic_model.load_model()

learning_rate = 1e-4
loss_function = nn.NLLLoss()
traffic_optimizer = optim.Adam(traffic_model.parameters(), learning_rate)


traffic_targets = []
for _, target in traffic_model.test_data:
    traffic_targets.append(target)

traffic_targets = torch.tensor(traffic_targets)

print("CW Attack Results on Traffic")
cw_attack = CW(traffic_model, device, targeted=False, search_steps=5, max_steps=20, confidence=0, lr=learning_rate, loss_function=loss_function, optimizer=traffic_optimizer, targets=traffic_targets)
start = time.time()
cr, preds, examples, results = traffic_model.test_attack_model(loss_function, cw_attack)
end = time.time()
print(cr)
print(f"Average iterations to perform CW: {np.mean(results["iterations"])}")
print(f"Time to test CW: {end-start}")

print(mnist_model)
print(torchsummary.summary(mnist_model, input_size=(3, 32, 32)))

print("Results for Feature Squeezing defense on MNIST")
fs_defense = FeatureSqueezing(mnist_model, device)
print("Baseline FS MNIST")
start = time.time()
cr, preds, examples = mnist_model.test_baseline_defense_model(loss_function, fs_defense)
end = time.time()
print(cr)
print(f"Average time to test baseline: {end-start}")

plt.imshow(examples[0][2][0,0,:,:], cmap="gray")
plt.axis("off")
plt.show()

print("Baseline well trained MNIST model")
start = time.time()
cr, preds, examples = mnist_model.test_model(loss_function)
end = time.time()
print(cr)
print(f"Time to test baseline MNIST model: {end-start}")

for (inputs, labels) in mnist_model.test_dataloader:
    (inputs, labels) = (inputs.to(device), labels.to(device))
    plt.imshow(inputs.squeeze().detach().cpu()[0,0,:,:], cmap="gray")
    plt.axis("off")
    plt.show()
    break

# print(np.exp(preds))
# print(np.sum(np.exp(preds)))

# probs = np.exp(preds)
# labels = np.argsort(probs)[-3:]
# percentages = np.round(probs[labels] * 100, decimals=8)
# print(labels)
# print(percentages)
print(examples[0][1].size())
plt.imshow(examples[0][1][0,0,:,:], cmap="gray")
plt.axis("off")
plt.show()

print("-" * 50)

print("IFGSM Attack Results on MNIST")
ifgsm_attack = IFGSM(mnist_model, device, targeted=False, loss_function=loss_function, optimizer=mnist_optimizer, eps=0.1, max_steps=20, decay=1, alpha=0.01)
start = time.time()
cr, preds, examples, results = mnist_model.test_attack_model(loss_function, ifgsm_attack)
end = time.time()
print(cr)
print(f"Average iterations to perform IFGSM: {np.mean(results["iterations"])}")
print(f"Time to test IFGSM: {end-start}")

print(np.exp(preds))
print(np.sum(np.exp(preds)))

probs = np.exp(preds)
labels = np.argsort(probs)[-3:]
percentages = np.round(probs[labels] * 100, decimals=8)
print(labels)
print(percentages)

plt.imshow(examples[0][2][0,0,:,:], cmap="gray")
plt.axis("off")
plt.show()

print("-" * 50)

print("Deepfool Attack Results on MNIST")
deepfool_attack = DeepFool(mnist_model, device, targeted=False, step=0.02, max_iter=1000, loss_function=loss_function, optimizer=mnist_optimizer)
# start = time.time()
# cr, preds, examples, results = mnist_model.test_attack_model(loss_function, deepfool_attack)
# end = time.time()
# print(cr)
# print(f"Average iterations to perform Deepfool: {np.mean(results["iterations"])}")
# print(f"Time to test Deepfool: {end-start}")

print("Deepfool Attack results on FS Model")
fs_defense = FeatureSqueezing(mnist_model, device)
start = time.time()
cr, preds, examples, results = mnist_model.test_defense_model(loss_function, deepfool_attack, fs_defense)
end = time.time()
print(cr)
print(f"Average time to test Deepfool: {end-start}")

print(np.exp(preds))
print(np.sum(np.exp(preds)))

probs = np.exp(preds)
labels = np.argsort(probs)[-3:]
percentages = np.round(probs[labels] * 100, decimals=8)
print(labels)
print(percentages)

plt.imshow(examples[0][2][0,0,:,:], cmap="gray")
plt.axis("off")
plt.show()

print("-" * 50)

mnist_targets = []
for _, target in mnist_model.test_data:
    mnist_targets.append(target)

mnist_targets = torch.tensor(mnist_targets)

print("CW Attack Results on MNIST")
cw_attack = CW(mnist_model, device, targeted=False, search_steps=5, max_steps=20, confidence=0, lr=learning_rate, loss_function=loss_function, optimizer=mnist_optimizer, targets=mnist_targets)
start = time.time()
cr, preds, examples, results = mnist_model.test_attack_model(loss_function, cw_attack)
end = time.time()
print(cr)
print(f"Average iterations to perform CW: {np.mean(results["iterations"])}")
print(f"Time to test CW: {end-start}")

print(np.exp(preds))
print(np.sum(np.exp(preds)))

probs = np.exp(preds)
labels = np.argsort(probs)[-3:]
percentages = np.round(probs[labels] * 100, decimals=8)
print(labels)
print(percentages)

plt.imshow(examples[0][2][0,0,:,:], cmap="gray")
plt.axis("off")
plt.show()

print("-" * 50)

print("Pixle Attack Results on MNIST")
pixle_attack = Pixle(mnist_model, device, targeted=False, attack_type=0, loss_function=loss_function, optimizer=mnist_optimizer, max_steps=0, max_patches=0)
start = time.time()
cr, preds, examples, results = mnist_model.test_attack_model(loss_function, pixle_attack)
end = time.time()
print(cr)
print(f"Average iterations to perform Pixle: {np.mean(results["iterations"])}")
print(f"Time to test Pixle: {end-start}")

print(np.exp(preds))
print(np.sum(np.exp(preds)))

probs = np.exp(preds)
labels = np.argsort(probs)[-3:]
percentages = np.round(probs[labels] * 100, decimals=8)
print(labels)
print(percentages)

plt.imshow(examples[0][2][0,0,:,:], cmap="gray")
plt.axis("off")
plt.show()