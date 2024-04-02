import torch
import torch.nn as nn

from NeuralNetworks.TrafficNeuralNetwork import TrafficNeuralNetwork
from NeuralNetworks.CifarNeuralNetwork import CifarNeuralNetwork
from NeuralNetworks.MNISTNeuralNetwork import MNISTNeuralNetwork

from Defenses.AdverserialExamples import AdverserialExamples
from Defenses.Distiller import Distiller
from Attacks.CW import CW
from Attacks.DeepFool import DeepFool
from Attacks.IFGSM import IFGSM
from Attacks.Pixle import Pixle

from sklearn.metrics import classification_report

# ssl certificate not working - but link is secure so override
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

device = torch.device("cpu")

# Between 1e-3 and 1e-5
learning_rate = 1e-4
epochs = 20
loss_function = nn.NLLLoss()

# for e in range(epochs):
#     print(f"Epoch {e+1}")
#     print(model.train_model(loss_function=loss_function, optimizer=optimizer))
#     print("-"*50)

# cr, preds, examples = model.test_model(loss_function)
# print(cr)

"""
MNIST
"""

# epochs = 1

# model = MNISTNeuralNetwork(device, train_split=0.8, batch_size=64).to(device)
# model.load_model()
# optimizer = torch.optim.Adam(model.parameters(), learning_rate)

# targets = []
# for _, target in model.test_data:
#     targets.append(target)

# targets = torch.tensor(targets)

# ifgsm_attack = IFGSM(model, device, targeted=False, loss_function=loss_function, optimizer=optimizer, eps=0.1, max_steps=20, decay=1, alpha=0.01)
# cw_attack = CW(model, device, targeted=False, search_steps=5, max_steps=20, confidence=0, lr=learning_rate, loss_function=loss_function, optimizer=optimizer, targets=targets)
# deepfool_attack = DeepFool(model, device, targeted=False, step=0.02, max_iter=1000, loss_function=loss_function, optimizer=optimizer)
# pixle_attack = Pixle(model, device, targeted=False, attack_type=0, loss_function=loss_function, optimizer=optimizer, max_steps=0, max_patches=0)

# adverserials = AdverserialExamples(model=None, device=device, ifgsm=ifgsm_attack, cw=cw_attack, deepfool=deepfool_attack, pixle=pixle_attack, dataset="MNIST", learning_rate=learning_rate, loss_function=loss_function)
# print(adverserials.model)

# # for e in range(epochs):
# #     print(f"Epoch {e+1}")
# #     history = adverserials.train_model_adverserial_examples(loss_function, optimizer)
# #     print(history)
# #     print("-"*50)

# print("baseline - adversarial")

# cr, preds, examples = adverserials.test_adverserials(loss_function)
# print(cr)

# print("baseline - original")

# cr, preds, examples = model.test_model(loss_function)
# print(cr)

# print("-" * 50)

# print("ifgsm - adversarial")

# cr, preds, examples, results = adverserials.test_attack_adversarials(loss_function, ifgsm_attack)
# print(cr)
# print(classification_report(results["final_label"], results["attack_label"], zero_division=0.0))

# print("ifgsm - original")

# cr, preds, examples, results = model.test_attack_model(loss_function, ifgsm_attack)
# print(cr)

# print("-" * 50)

# print("deepfool - adversarial")

# cr, preds, examples, results = adverserials.test_attack_adversarials(loss_function, deepfool_attack)
# print(cr)
# print(classification_report(results["final_label"], results["attack_label"], zero_division=0.0))

# print("deepfool - original")

# cr, preds, examples, results = model.test_attack_model(loss_function, deepfool_attack)
# print(cr)

# print("-" * 50)

# print("cw - adversarial")

# cr, preds, examples, results = adverserials.test_attack_adversarials(loss_function, cw_attack)
# print(cr)
# print(classification_report(results["final_label"], results["attack_label"], zero_division=0.0))

# print("cw - original")

# cr, preds, examples, results = model.test_attack_model(loss_function, cw_attack)
# print(cr)



# mnist_distiller = Distiller(model, device, "MNIST", learning_rate, loss_function)
# for e in range(epochs):
#     print(f"Epoch {e+1}")
#     teacher_history, student_history = mnist_distiller.train_distillation()
#     print(teacher_history)
#     print(student_history)
#     print("-"*50)

"""
Cifar
"""

model = CifarNeuralNetwork(device, train_split=0.8, batch_size=64).to(device)
model.load_model()
optimizer = torch.optim.Adam(model.parameters(), learning_rate)

targets = []
for _, target in model.test_data:
    targets.append(target)

targets = torch.tensor(targets)

ifgsm_attack = IFGSM(model, device, targeted=False, loss_function=loss_function, optimizer=optimizer, eps=0.1, max_steps=20, decay=1, alpha=0.01)
cw_attack = CW(model, device, targeted=False, search_steps=5, max_steps=20, confidence=0, lr=learning_rate, loss_function=loss_function, optimizer=optimizer, targets=targets)
deepfool_attack = DeepFool(model, device, targeted=False, step=0.02, max_iter=1000, loss_function=loss_function, optimizer=optimizer)
pixle_attack = Pixle(model, device, targeted=False, attack_type=0, loss_function=loss_function, optimizer=optimizer, max_steps=0, max_patches=0)

adverserials = AdverserialExamples(model=None, device=device, ifgsm=ifgsm_attack, cw=cw_attack, deepfool=deepfool_attack, pixle=pixle_attack, dataset="Cifar", learning_rate=learning_rate, loss_function=loss_function)

# for e in range(epochs):
#     print(f"Epoch {e+1}")
#     history = adverserials.train_model_adverserial_examples(loss_function, optimizer)
#     print(history)
#     print("-"*50)

# cr, preds, examples = adverserials.test_adverserials(loss_function)
# print(cr)

# cifar_distiller = Distiller(model, device, "Cifar", learning_rate, loss_function)
# for e in range(epochs):
#     print(f"Epoch {e+1}")
#     teacher_history, student_history = cifar_distiller.train_distillation()
#     print(teacher_history)
#     print(student_history)
#     print("-"*50)

# cr, preds, examples = cifar_distiller.test_distillation()
# print(cr)

"""
Traffic
"""

model = TrafficNeuralNetwork(device, train_split=0.8, batch_size=64).to(device)
model.load_model()
optimizer = torch.optim.Adam(model.parameters(), learning_rate)

targets = []
for _, target in model.test_data:
    targets.append(target)

targets = torch.tensor(targets)

ifgsm_attack = IFGSM(model, device, targeted=False, loss_function=loss_function, optimizer=optimizer, eps=0.1, max_steps=20, decay=1, alpha=0.01)
cw_attack = CW(model, device, targeted=False, search_steps=5, max_steps=20, confidence=0, lr=learning_rate, loss_function=loss_function, optimizer=optimizer, targets=targets)
deepfool_attack = DeepFool(model, device, targeted=False, step=0.02, max_iter=1000, loss_function=loss_function, optimizer=optimizer)
pixle_attack = Pixle(model, device, targeted=False, attack_type=0, loss_function=loss_function, optimizer=optimizer, max_steps=0, max_patches=0)

adverserials = AdverserialExamples(model=None, device=device, ifgsm=ifgsm_attack, cw=cw_attack, deepfool=deepfool_attack, pixle=pixle_attack, dataset="Traffic", learning_rate=learning_rate, loss_function=loss_function)

# for e in range(epochs):
#     print(f"Epoch {e+1}")
#     history = adverserials.train_model_adverserial_examples(loss_function, optimizer)
#     print(history)
#     print("-"*50)

cr, preds, examples = adverserials.test_adverserials(loss_function)
print(cr)

traffic_distiller = Distiller(model, device, "Traffic", learning_rate, loss_function)
for e in range(epochs):
    print(f"Epoch {e+1}")
    teacher_history, student_history = traffic_distiller.train_distillation()
    print(teacher_history)
    print(student_history)
    print("-"*50)

cr, preds, examples = traffic_distiller.test_distillation()
print(cr)