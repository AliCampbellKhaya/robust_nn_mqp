import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np

from sklearn.metrics import classification_report

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

# print("Baseline well trained MNIST model")
# start = time.time()
# cr, preds, examples = mnist_model.test_model(loss_function)
# end = time.time()
# print(cr)
# print(f"Time to test baseline MNIST model: {end-start}")

# print("-" * 50)

# print("IFGSM Attack Results on MNIST")
# ifgsm_attack = IFGSM(mnist_model, device, targeted=False, loss_function=loss_function, optimizer=mnist_optimizer, eps=0.7, max_steps=20, decay=1, alpha=0.1)
# start = time.time()
# cr, preds, examples, results = mnist_model.test_attack_model(loss_function, ifgsm_attack)
# end = time.time()
# print(cr)
# print(f"Average iterations to perform IFGSM: {np.mean(results["iterations"])}")
# print(f"Time to test IFGSM: {end-start}")
# print(classification_report(results["final_label"], results["attack_label"]))

# plt.figure(figsize=(8,10))
# plt.xticks([], [])
# plt.yticks([], [])

# plt.imshow(results["perturbations"][0][0,0,:,:], cmap="gray")
# plt.tight_layout()
# plt.show()

# print("-" * 50)

# print("Deepfool Attack Results on MNIST")
# deepfool_attack = DeepFool(mnist_model, device, targeted=False, step=0.02, max_iter=1000, loss_function=loss_function, optimizer=mnist_optimizer)
# start = time.time()
# cr, preds, examples, results = mnist_model.test_attack_model(loss_function, deepfool_attack)
# end = time.time()
# print(cr)
# print(f"Average iterations to perform Deepfool: {np.mean(results["iterations"])}")
# print(f"Time to test Deepfool: {end-start}")

# print("-" * 50)

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

# print("-" * 50)

# print("Pixle Attack Results on MNIST")
# pixle_attack = Pixle(mnist_model, device, targeted=False, attack_type=0, loss_function=loss_function, optimizer=mnist_optimizer, max_steps=0, max_patches=0)
# start = time.time()
# cr, preds, examples, results = mnist_model.test_attack_model(loss_function, pixle_attack)
# end = time.time()
# print(cr)
# print(f"Average iterations to perform Pixle: {np.mean(results["iterations"])}")
# print(f"Time to test Pixle: {end-start}")

# print("-" * 50)

# print("Adversarial Example MNIST model")
# adverserials = AdverserialExamples(model=None, device=device, ifgsm=ifgsm_attack, cw=cw_attack, deepfool=deepfool_attack, pixle=pixle_attack, dataset="MNIST", learning_rate=learning_rate, loss_function=loss_function)
# print("Baseline MNIST AE")
# start = time.time()
# cr, preds, examples = adverserials.test_adverserials(loss_function)
# end = time.time()
# print(cr)
# print(f"Time to test baseline AE Model: {end-start}")

# print("IFGSM Attack results on AE Model")
# start = time.time()
# cr, preds, examples, results = adverserials.test_attack_adversarials(loss_function, ifgsm_attack)
# end = time.time()
# print(cr)
# print(f"Average iterations to perform IFGSM on AE Model: {np.mean(results["iterations"])}")
# print(f"Time to test IFGSM: {end-start}")

# print("Deepfool Attack results on AE Model")
# start = time.time()
# cr, preds, examples, results = adverserials.test_attack_adversarials(loss_function, deepfool_attack)
# end = time.time()
# print(cr)
# print(f"Average iterations to perform Deepfool on AE Model: {np.mean(results["iterations"])}")
# print(f"Time to test Deepfool: {end-start}")

# print("CW Attack results on AE Model")
# start = time.time()
# cr, preds, examples, results = adverserials.test_attack_adversarials(loss_function, cw_attack)
# end = time.time()
# print(cr)
# print(f"Average iterations to perform CW on AE Model: {np.mean(results["iterations"])}")
# print(f"Time to test CW: {end-start}")

# print("Pixle Attack results on AE Model")
# start = time.time()
# cr, preds, examples, results = adverserials.test_attack_adversarials(loss_function, pixle_attack)
# end = time.time()
# print(cr)
# print(f"Average iterations to perform Pixle on AE Model: {np.mean(results["iterations"])}")
# print(f"Time to test Pixle: {end-start}")

# print("-" * 50)

# print("Results for Feature Squeezing defense on MNIST")
fs_defense = FeatureSqueezing(mnist_model, device)
# print("Baseline FS MNIST")
# start = time.time()
# cr, preds, examples = mnist_model.test_baseline_defense_model(loss_function, fs_defense)
# end = time.time()
# print(cr)
# print(f"Average time to test baseline: {end-start}")

# print("IFGSM Attack results on FS Model")
# start = time.time()
# cr, preds, examples, results = mnist_model.test_defense_model(loss_function, ifgsm_attack, fs_defense)
# end = time.time()
# print(cr)
# print(f"Average time to test IFGSM: {end-start}")

# print("Deepfool Attack results on FS Model")
# start = time.time()
# cr, preds, examples, results = mnist_model.test_defense_model(loss_function, deepfool_attack, fs_defense)
# end = time.time()
# print(cr)
# print(f"Average time to test Deepfool: {end-start}")

print("CW Attack results on FS Model")
start = time.time()
cr, preds, examples, results = mnist_model.test_defense_model(loss_function, cw_attack, fs_defense)
end = time.time()
print(cr)
print(f"Average time to test CW: {end-start}")

# print("Pixle Attack results on FS Model")
# start = time.time()
# cr, preds, examples, results = mnist_model.test_defense_model(loss_function, pixle_attack, fs_defense)
# end = time.time()
# print(cr)
# print(f"Average time to test Pixle: {end-start}")

# print("-" * 50)

# print("Results for Gradient Masking defense on MNIST")
# gm_defense = GradientMasking(mnist_model, device, loss_function, 0.2)
# print("Baseline GM MNIST")
# start = time.time()
# cr, preds, examples = mnist_model.test_baseline_defense_model(loss_function, gm_defense)
# end = time.time()
# print(cr)
# print(f"Average time to test baseline: {end-start}")

# print("IFGSM Attack results on GM Model")
# start = time.time()
# cr, preds, examples, results = mnist_model.test_defense_model(loss_function, ifgsm_attack, gm_defense)
# end = time.time()
# print(cr)
# print(f"Average time to test IFGSM: {end-start}")

# print("Deepfool Attack results on GM Model")
# start = time.time()
# cr, preds, examples, results = mnist_model.test_defense_model(loss_function, deepfool_attack, gm_defense)
# end = time.time()
# print(cr)
# print(f"Average time to test Deepfool: {end-start}")

# print("CW Attack results on GM Model")
# start = time.time()
# cr, preds, examples, results = mnist_model.test_defense_model(loss_function, cw_attack, gm_defense)
# end = time.time()
# print(cr)
# print(f"Average time to test CW: {end-start}")

# print("Pixle Attack results on GM Model")
# start = time.time()
# cr, preds, examples, results = mnist_model.test_defense_model(loss_function, pixle_attack, gm_defense)
# end = time.time()
# print(cr)
# print(f"Average time to test Pixle: {end-start}")

# print("-" * 50)

# print("Results for Distillation Defence on MNIST")
# mnist_distiller = Distiller(mnist_model, device, "MNIST", learning_rate, loss_function)
# print("Baseline Distilled MNIST")
# start = time.time()
# cr, preds, examples = mnist_distiller.test_distillation()
# end = time.time()
# print(cr)
# print(f"Average time to test baseline: {end-start}")

# print("IFGSM Attack results on Distilled Model")
# start = time.time()
# cr, preds, examples, results = mnist_distiller.test_attack_distillation(ifgsm_attack)
# end = time.time()
# print(cr)
# print(f"Average iterations to perform IFGSM on Distilled Model: {np.mean(results["iterations"])}")
# print(f"Time to test IFGSM: {end-start}")

# print("Deepfool Attack results on Distilled Model")
# start = time.time()
# cr, preds, examples, results = mnist_distiller.test_attack_distillation(deepfool_attack)
# end = time.time()
# print(cr)
# print(f"Average iterations to perform Deepfool on Distilled Model: {np.mean(results["iterations"])}")
# print(f"Time to test Deepfool: {end-start}")

# print("CW Attack results on Distilled Model")
# start = time.time()
# cr, preds, examples, results = mnist_distiller.test_attack_distillation(cw_attack)
# end = time.time()
# print(cr)
# print(f"Average iterations to perform CW on Distilled Model: {np.mean(results["iterations"])}")
# print(f"Time to test CW: {end-start}")

# print("Pixle Attack results on Distilled Model")
# start = time.time()
# cr, preds, examples, results = mnist_distiller.test_attack_distillation(pixle_attack)
# end = time.time()
# print(cr)
# print(f"Average iterations to perform Pixle on Distilled Model: {np.mean(results["iterations"])}")
# print(f"Time to test Pixle: {end-start}")

# print("-" * 50)
# print("-" * 50)
# print("-" * 50)

# print("Cifar Tests")

# device = torch.device("cpu")
# cifar_model = CifarNeuralNetwork(device, train_split=0.8, batch_size=64).to(device)

# learning_rate = 1e-4
# loss_function = nn.NLLLoss()
# cifar_optimizer = optim.Adam(cifar_model.parameters(), learning_rate)

# cifar_model.load_model()

# print("Baseline well trained Cifar model")
# start = time.time()
# cr, preds, examples = cifar_model.test_model(loss_function)
# end = time.time()
# print(cr)
# print(f"Time to test baseline Cifar model: {end-start}")

# print("-" * 50)

# print("IFGSM Attack Results on Cifar")
# ifgsm_attack = IFGSM(cifar_model, device, targeted=False, loss_function=loss_function, optimizer=cifar_optimizer, eps=0.1, max_steps=20, decay=1, alpha=0.01)
# start = time.time()
# cr, preds, examples, results = cifar_model.test_attack_model(loss_function, ifgsm_attack)
# end = time.time()
# print(cr)
# print(f"Average iterations to perform IFGSM: {np.mean(results["iterations"])}")
# print(f"Time to test IFGSM: {end-start}")

# print("-" * 50)

# print("Deepfool Attack Results on Cifar")
# deepfool_attack = DeepFool(cifar_model, device, targeted=False, step=0.02, max_iter=1000, loss_function=loss_function, optimizer=cifar_optimizer)
# start = time.time()
# cr, preds, examples, results = cifar_model.test_attack_model(loss_function, deepfool_attack)
# end = time.time()
# print(cr)
# print(f"Average iterations to perform Deepfool: {np.mean(results["iterations"])}")
# print(f"Time to test Deepfool: {end-start}")

# print("-" * 50)

# cifar_targets = []
# for _, target in cifar_model.test_data:
#     cifar_targets.append(target)

# cifar_targets = torch.tensor(cifar_targets)

# print("CW Attack Results on Cifar")
# cw_attack = CW(cifar_model, device, targeted=False, search_steps=5, max_steps=20, confidence=0, lr=learning_rate, loss_function=loss_function, optimizer=cifar_optimizer, targets=cifar_targets)
# start = time.time()
# cr, preds, examples, results = cifar_model.test_attack_model(loss_function, cw_attack)
# end = time.time()
# print(cr)
# print(f"Average iterations to perform CW: {np.mean(results["iterations"])}")
# print(f"Time to test CW: {end-start}")

# print("-" * 50)

# print("Pixle Attack Results on Cifar")
# pixle_attack = Pixle(cifar_model, device, targeted=False, attack_type=0, loss_function=loss_function, optimizer=cifar_optimizer, max_steps=0, max_patches=0)
# start = time.time()
# cr, preds, examples, results = cifar_model.test_attack_model(loss_function, pixle_attack)
# end = time.time()
# print(cr)
# print(f"Average iterations to perform Pixle: {np.mean(results["iterations"])}")
# print(f"Time to test Pixle: {end-start}")

# print("-" * 50)

# print("Adversarial Example Cifar model")
# adverserials = AdverserialExamples(model=None, device=device, ifgsm=ifgsm_attack, cw=cw_attack, deepfool=deepfool_attack, pixle=pixle_attack, dataset="Cifar", learning_rate=learning_rate, loss_function=loss_function)
# print("Baseline MNIST AE")
# start = time.time()
# cr, preds, examples = adverserials.test_adverserials(loss_function)
# end = time.time()
# print(cr)
# print(f"Time to test baseline AE Model: {end-start}")

# print("IFGSM Attack results on AE Model")
# start = time.time()
# cr, preds, examples, results = adverserials.test_attack_adversarials(loss_function, ifgsm_attack)
# end = time.time()
# print(cr)
# print(f"Average iterations to perform IFGSM on AE Model: {np.mean(results["iterations"])}")
# print(f"Time to test IFGSM: {end-start}")

# print("Deepfool Attack results on AE Model")
# start = time.time()
# cr, preds, examples, results = adverserials.test_attack_adversarials(loss_function, deepfool_attack)
# end = time.time()
# print(cr)
# print(f"Average iterations to perform Deepfool on AE Model: {np.mean(results["iterations"])}")
# print(f"Time to test Deepfool: {end-start}")

# print("CW Attack results on AE Model")
# start = time.time()
# cr, preds, examples, results = adverserials.test_attack_adversarials(loss_function, cw_attack)
# end = time.time()
# print(cr)
# print(f"Average iterations to perform CW on AE Model: {np.mean(results["iterations"])}")
# print(f"Time to test CW: {end-start}")

# print("Pixle Attack results on AE Model")
# start = time.time()
# cr, preds, examples, results = adverserials.test_attack_adversarials(loss_function, pixle_attack)
# end = time.time()
# print(cr)
# print(f"Average iterations to perform Pixle on AE Model: {np.mean(results["iterations"])}")
# print(f"Time to test Pixle: {end-start}")

# print("-" * 50)

# print("Results for Feature Squeezing defense on MNIST")
# fs_defense = FeatureSqueezing(cifar_model, device)
# print("Baseline FS MNIST")
# start = time.time()
# cr, preds, examples = cifar_model.test_baseline_defense_model(loss_function, fs_defense)
# end = time.time()
# print(cr)
# print(f"Average time to test baseline: {end-start}")

# print("IFGSM Attack results on FS Model")
# start = time.time()
# cr, preds, examples, results = cifar_model.test_defense_model(loss_function, ifgsm_attack, fs_defense)
# end = time.time()
# print(cr)
# print(f"Average time to test IFGSM: {end-start}")

# print("Deepfool Attack results on FS Model")
# start = time.time()
# cr, preds, examples, results = cifar_model.test_defense_model(loss_function, deepfool_attack, fs_defense)
# end = time.time()
# print(cr)
# print(f"Average time to test Deepfool: {end-start}")

# print("CW Attack results on FS Model")
# start = time.time()
# cr, preds, examples, results = cifar_model.test_defense_model(loss_function, cw_attack, fs_defense)
# end = time.time()
# print(cr)
# print(f"Average time to test CW: {end-start}")

# print("Pixle Attack results on FS Model")
# start = time.time()
# cr, preds, examples, results = cifar_model.test_defense_model(loss_function, pixle_attack, fs_defense)
# end = time.time()
# print(cr)
# print(f"Average time to test Pixle: {end-start}")

# print("-" * 50)

# print("Results for Gradient Masking defense on MNIST")
# gm_defense = GradientMasking(cifar_model, device, loss_function, 0.2)
# print("Baseline GM MNIST")
# start = time.time()
# cr, preds, examples = cifar_model.test_baseline_defense_model(loss_function, gm_defense)
# end = time.time()
# print(cr)
# print(f"Average time to test baseline: {end-start}")

# print("IFGSM Attack results on GM Model")
# start = time.time()
# cr, preds, examples, results = cifar_model.test_defense_model(loss_function, ifgsm_attack, gm_defense)
# end = time.time()
# print(cr)
# print(f"Average time to test IFGSM: {end-start}")

# print("Deepfool Attack results on GM Model")
# start = time.time()
# cr, preds, examples, results = cifar_model.test_defense_model(loss_function, deepfool_attack, gm_defense)
# end = time.time()
# print(cr)
# print(f"Average time to test Deepfool: {end-start}")

# print("CW Attack results on GM Model")
# start = time.time()
# cr, preds, examples, results = cifar_model.test_defense_model(loss_function, cw_attack, gm_defense)
# end = time.time()
# print(cr)
# print(f"Average time to test CW: {end-start}")

# print("Pixle Attack results on GM Model")
# start = time.time()
# cr, preds, examples, results = cifar_model.test_defense_model(loss_function, pixle_attack, gm_defense)
# end = time.time()
# print(cr)
# print(f"Average time to test Pixle: {end-start}")

# print("-" * 50)

# print("Results for Distillation Defence on Cifar")
# cifar_distiller = Distiller(cifar_model, device, "Cifar", learning_rate, loss_function)
# print("Baseline Distilled MNIST")
# start = time.time()
# cr, preds, examples = cifar_distiller.test_distillation()
# end = time.time()
# print(cr)
# print(f"Average time to test baseline: {end-start}")

# print("IFGSM Attack results on Distilled Model")
# start = time.time()
# cr, preds, examples, results = cifar_distiller.test_attack_distillation(ifgsm_attack)
# end = time.time()
# print(cr)
# print(f"Average iterations to perform IFGSM on Distilled Model: {np.mean(results["iterations"])}")
# print(f"Time to test IFGSM: {end-start}")

# print("Deepfool Attack results on Distilled Model")
# start = time.time()
# cr, preds, examples, results = cifar_distiller.test_attack_distillation(deepfool_attack)
# end = time.time()
# print(cr)
# print(f"Average iterations to perform Deepfool on Distilled Model: {np.mean(results["iterations"])}")
# print(f"Time to test Deepfool: {end-start}")

# print("CW Attack results on Distilled Model")
# start = time.time()
# cr, preds, examples, results = cifar_distiller.test_attack_distillation(cw_attack)
# end = time.time()
# print(cr)
# print(f"Average iterations to perform CW on Distilled Model: {np.mean(results["iterations"])}")
# print(f"Time to test CW: {end-start}")

# print("Pixle Attack results on Distilled Model")
# start = time.time()
# cr, preds, examples, results = cifar_distiller.test_attack_distillation(pixle_attack)
# end = time.time()
# print(cr)
# print(f"Average iterations to perform Pixle on Distilled Model: {np.mean(results["iterations"])}")
# print(f"Time to test Pixle: {end-start}")

# print("-" * 50)
# print("-" * 50)
# print("-" * 50)

# print("Traffic Tests")

# device = torch.device("cpu")
# traffic_model = TrafficNeuralNetwork(device, train_split=0.8, batch_size=64).to(device)

# learning_rate = 1e-4
# loss_function = nn.NLLLoss()
# traffic_optimizer = optim.Adam(traffic_model.parameters(), learning_rate)

# traffic_model.load_model()

# print("Baseline well trained Traffic model")
# start = time.time()
# cr, preds, examples = traffic_model.test_model(loss_function)
# end = time.time()
# print(cr)
# print(f"Time to test baseline Traffic model: {end-start}")

# print("-" * 50)

# print("IFGSM Attack Results on Traffic")
# ifgsm_attack = IFGSM(traffic_model, device, targeted=False, loss_function=loss_function, optimizer=traffic_optimizer, eps=0.1, max_steps=20, decay=1, alpha=0.01)
# start = time.time()
# cr, preds, examples, results = traffic_model.test_attack_model(loss_function, ifgsm_attack)
# end = time.time()
# print(cr)
# print(f"Average iterations to perform IFGSM: {np.mean(results["iterations"])}")
# print(f"Time to test IFGSM: {end-start}")

# print("-" * 50)

# print("Deepfool Attack Results on Traffic")
# deepfool_attack = DeepFool(traffic_model, device, targeted=False, step=0.02, max_iter=1000, loss_function=loss_function, optimizer=traffic_optimizer)
# start = time.time()
# cr, preds, examples, results = traffic_model.test_attack_model(loss_function, deepfool_attack)
# end = time.time()
# print(cr)
# print(f"Average iterations to perform Deepfool: {np.mean(results["iterations"])}")
# print(f"Time to test Deepfool: {end-start}")

# print("-" * 50)

# traffic_targets = []
# for _, target in traffic_model.test_data:
#     traffic_targets.append(target)

# traffic_targets = torch.tensor(traffic_targets)

# print("CW Attack Results on Traffic")
# cw_attack = CW(traffic_model, device, targeted=False, search_steps=5, max_steps=20, confidence=0, lr=learning_rate, loss_function=loss_function, optimizer=traffic_optimizer, targets=traffic_targets)
# start = time.time()
# cr, preds, examples, results = traffic_model.test_attack_model(loss_function, cw_attack)
# end = time.time()
# print(cr)
# print(f"Average iterations to perform CW: {np.mean(results["iterations"])}")
# print(f"Time to test CW: {end-start}")

# print("-" * 50)

# print("Pixle Attack Results on Traffic")
# pixle_attack = Pixle(traffic_model, device, targeted=False, attack_type=0, loss_function=loss_function, optimizer=traffic_optimizer, max_steps=0, max_patches=0)
# start = time.time()
# cr, preds, examples, results = traffic_model.test_attack_model(loss_function, pixle_attack)
# end = time.time()
# print(cr)
# print(f"Average iterations to perform Pixle: {np.mean(results["iterations"])}")
# print(f"Time to test Pixle: {end-start}")

# print("-" * 50)

# print("Adversarial Example Traffic model")
# adverserials = AdverserialExamples(model=None, device=device, ifgsm=ifgsm_attack, cw=cw_attack, deepfool=deepfool_attack, pixle=pixle_attack, dataset="Traffic", learning_rate=learning_rate, loss_function=loss_function)
# print("Baseline Traffic AE")
# start = time.time()
# cr, preds, examples = adverserials.test_adverserials(loss_function)
# end = time.time()
# print(cr)
# print(f"Time to test baseline AE Model: {end-start}")

# print("IFGSM Attack results on AE Model")
# start = time.time()
# cr, preds, examples, results = adverserials.test_attack_adversarials(loss_function, ifgsm_attack)
# end = time.time()
# print(cr)
# print(f"Average iterations to perform IFGSM on AE Model: {np.mean(results["iterations"])}")
# print(f"Time to test IFGSM: {end-start}")

# print("Deepfool Attack results on AE Model")
# start = time.time()
# cr, preds, examples, results = adverserials.test_attack_adversarials(loss_function, deepfool_attack)
# end = time.time()
# print(cr)
# print(f"Average iterations to perform Deepfool on AE Model: {np.mean(results["iterations"])}")
# print(f"Time to test Deepfool: {end-start}")

# print("CW Attack results on AE Model")
# start = time.time()
# cr, preds, examples, results = adverserials.test_attack_adversarials(loss_function, cw_attack)
# end = time.time()
# print(cr)
# print(f"Average iterations to perform CW on AE Model: {np.mean(results["iterations"])}")
# print(f"Time to test CW: {end-start}")

# print("Pixle Attack results on AE Model")
# start = time.time()
# cr, preds, examples, results = adverserials.test_attack_adversarials(loss_function, pixle_attack)
# end = time.time()
# print(cr)
# print(f"Average iterations to perform Pixle on AE Model: {np.mean(results["iterations"])}")
# print(f"Time to test Pixle: {end-start}")

# print("-" * 50)

# print("Results for Feature Squeezing defense on Traffic")
# fs_defense = FeatureSqueezing(traffic_model, device)
# print("Baseline FS MNIST")
# start = time.time()
# cr, preds, examples = traffic_model.test_baseline_defense_model(loss_function, fs_defense)
# end = time.time()
# print(cr)
# print(f"Average time to test baseline: {end-start}")

# print("IFGSM Attack results on FS Model")
# start = time.time()
# cr, preds, examples, results = traffic_model.test_defense_model(loss_function, ifgsm_attack, fs_defense)
# end = time.time()
# print(cr)
# print(f"Average time to test IFGSM: {end-start}")

# print("Deepfool Attack results on FS Model")
# start = time.time()
# cr, preds, examples, results = traffic_model.test_defense_model(loss_function, deepfool_attack, fs_defense)
# end = time.time()
# print(cr)
# print(f"Average time to test Deepfool: {end-start}")

# print("CW Attack results on FS Model")
# start = time.time()
# cr, preds, examples, results = traffic_model.test_defense_model(loss_function, cw_attack, fs_defense)
# end = time.time()
# print(cr)
# print(f"Average time to test CW: {end-start}")

# print("Pixle Attack results on FS Model")
# start = time.time()
# cr, preds, examples, results = traffic_model.test_defense_model(loss_function, pixle_attack, fs_defense)
# end = time.time()
# print(cr)
# print(f"Average time to test Pixle: {end-start}")

# print("-" * 50)

# print("Results for Gradient Masking defense on MNIST")
# gm_defense = GradientMasking(traffic_model, device, loss_function, 0.2)
# print("Baseline GM MNIST")
# start = time.time()
# cr, preds, examples = traffic_model.test_baseline_defense_model(loss_function, gm_defense)
# end = time.time()
# print(cr)
# print(f"Average time to test baseline: {end-start}")

# print("IFGSM Attack results on GM Model")
# start = time.time()
# cr, preds, examples, results = traffic_model.test_defense_model(loss_function, ifgsm_attack, gm_defense)
# end = time.time()
# print(cr)
# print(f"Average time to test IFGSM: {end-start}")

# print("Deepfool Attack results on GM Model")
# start = time.time()
# cr, preds, examples, results = traffic_model.test_defense_model(loss_function, deepfool_attack, gm_defense)
# end = time.time()
# print(cr)
# print(f"Average time to test Deepfool: {end-start}")

# print("CW Attack results on GM Model")
# start = time.time()
# cr, preds, examples, results = traffic_model.test_defense_model(loss_function, cw_attack, gm_defense)
# end = time.time()
# print(cr)
# print(f"Average time to test CW: {end-start}")

# print("Pixle Attack results on GM Model")
# start = time.time()
# cr, preds, examples, results = traffic_model.test_defense_model(loss_function, pixle_attack, gm_defense)
# end = time.time()
# print(cr)
# print(f"Average time to test Pixle: {end-start}")

# print("-" * 50)

# print("Results for Distillation Defence on Traffic")
# traffic_distiller = Distiller(traffic_model, device, "Traffic", learning_rate, loss_function)
# print("Baseline Distilled Traffic")
# start = time.time()
# cr, preds, examples = traffic_distiller.test_distillation()
# end = time.time()
# print(cr)
# print(f"Average time to test baseline: {end-start}")

# print("IFGSM Attack results on Distilled Model")
# start = time.time()
# cr, preds, examples, results = traffic_distiller.test_attack_distillation(ifgsm_attack)
# end = time.time()
# print(cr)
# print(f"Average iterations to perform IFGSM on Distilled Model: {np.mean(results["iterations"])}")
# print(f"Time to test IFGSM: {end-start}")

# print("Deepfool Attack results on Distilled Model")
# start = time.time()
# cr, preds, examples, results = traffic_distiller.test_attack_distillation(deepfool_attack)
# end = time.time()
# print(cr)
# print(f"Average iterations to perform Deepfool on Distilled Model: {np.mean(results["iterations"])}")
# print(f"Time to test Deepfool: {end-start}")

# print("CW Attack results on Distilled Model")
# start = time.time()
# cr, preds, examples, results = traffic_distiller.test_attack_distillation(cw_attack)
# end = time.time()
# print(cr)
# print(f"Average iterations to perform CW on Distilled Model: {np.mean(results["iterations"])}")
# print(f"Time to test CW: {end-start}")

# print("Pixle Attack results on Distilled Model")
# start = time.time()
# cr, preds, examples, results = traffic_distiller.test_attack_distillation(pixle_attack)
# end = time.time()
# print(cr)
# print(f"Average iterations to perform Pixle on Distilled Model: {np.mean(results["iterations"])}")
# print(f"Time to test Pixle: {end-start}")

# print("\n")
# print("Done! :)")