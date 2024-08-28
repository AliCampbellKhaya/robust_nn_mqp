import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
import torchsummary

from matplotlib.colors import NoNorm

from NeuralNetworks.MNISTNeuralNetwork import MNISTNeuralNetwork
from NeuralNetworks.CifarNeuralNetwork import CifarNeuralNetwork
from NeuralNetworks.TrafficNeuralNetwork import TrafficNeuralNetwork

from Attacks.IFGSM import IFGSM
from Attacks.DeepFool import DeepFool
from Attacks.CW import CW
from Attacks.Pixle import Pixle

from Defenses.AdverserialExamples import AdverserialExamples
from Defenses.FeatureSqueezing import FeatureSqueezing
from Defenses.GradientMasking import GradientMasking
from Defenses.Distiller import Distiller

print("MNIST Tests")

device = torch.device("cpu")
mnist_model = MNISTNeuralNetwork(device, train_split=0.8, batch_size=64).to(device)

learning_rate = 1e-4
loss_function = nn.NLLLoss()
mnist_optimizer = optim.Adam(mnist_model.parameters(), learning_rate)

mnist_model.load_model()

print("Baseline well trained MNIST model")
start = time.time()
cr, preds, examples = mnist_model.test_model(loss_function)
end = time.time()
print(cr)
print(f"Time to test baseline MNIST model: {end-start}")

plt.figure(figsize=(8,10))
plt.xticks([], [])
plt.yticks([], [])
img = examples
print(img)
#plt.imshow(img[0,:,:], cmap="gray")
# plt.imshow(img[0][1][0, 0, :,:], cmap="gray", norm=NoNorm())
plt.imshow(img[0][1][0,:,:,:].permute(1,2,0))
plt.tight_layout()
plt.show()

print("IFGSM Attack Results on MNIST")
ifgsm_attack = IFGSM(mnist_model, device, targeted=False, loss_function=loss_function, optimizer=mnist_optimizer, eps=0.7, max_steps=20, decay=1, alpha=0.7)
start = time.time()
cr, preds, examples, results = mnist_model.test_attack_model(loss_function, ifgsm_attack)
end = time.time()
print(cr)
print(f"Average iterations to perform IFGSM: {np.mean(results["iterations"])}")
print(f"Time to test IFGSM: {end-start}")

plt.figure(figsize=(8,10))
plt.xticks([], [])
plt.yticks([], [])
#plt.imshow(img[0,:,:], cmap="gray")
# plt.imshow(examples[0][2][0,0,:,:], cmap="gray", norm=NoNorm())
plt.imshow(examples[0][2][0,:,:,:].permute(1,2,0))
plt.tight_layout()
plt.show()

print("Deepfool Attack Results on MNIST")
deepfool_attack = DeepFool(mnist_model, device, targeted=False, step=0.02, max_iter=1000, loss_function=loss_function, optimizer=mnist_optimizer)
start = time.time()
cr, preds, examples, results = mnist_model.test_attack_model(loss_function, deepfool_attack)
end = time.time()
print(cr)
print(f"Average iterations to perform Deepfool: {np.mean(results["iterations"])}")
print(f"Time to test Deepfool: {end-start}")

plt.figure(figsize=(8,10))
plt.xticks([], [])
plt.yticks([], [])
#plt.imshow(img[0,:,:], cmap="gray")
# plt.imshow(examples[0][2][0,0,:,:], cmap="gray", norm=NoNorm())
plt.imshow(examples[0][2][0,:,:,:].permute(1,2,0))
plt.tight_layout()
plt.show()

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

plt.figure(figsize=(8,10))
plt.xticks([], [])
plt.yticks([], [])
#plt.imshow(img[0,:,:], cmap="gray")
# plt.imshow(examples[0][2][0,0,:,:], cmap="gray", norm=NoNorm())
plt.imshow(examples[0][2][0,:,:,:].permute(1,2,0))
plt.tight_layout()
plt.show()

print("Pixle Attack Results on MNIST")
pixle_attack = Pixle(mnist_model, device, targeted=False, attack_type=0, loss_function=loss_function, optimizer=mnist_optimizer, max_steps=0, max_patches=0)
start = time.time()
cr, preds, examples, results = mnist_model.test_attack_model(loss_function, pixle_attack)
end = time.time()
print(cr)
print(f"Average iterations to perform Pixle: {np.mean(results["iterations"])}")
print(f"Time to test Pixle: {end-start}")

plt.figure(figsize=(8,10))
plt.xticks([], [])
plt.yticks([], [])
#plt.imshow(img[0,:,:], cmap="gray")
# plt.imshow(examples[0][2][0,0,:,:], cmap="gray", norm=NoNorm())
plt.imshow(examples[0][2][0,:,:,:].permute(1,2,0))
plt.tight_layout()
plt.show()