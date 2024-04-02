import torch
import torch.nn as nn

from NeuralNetworks.MNISTNeuralNetwork import MNISTNeuralNetwork

from Defenses.AdverserialExamples import AdverserialExamples
from Defenses.Distiller import Distiller
from Attacks.CW import CW
from Attacks.DeepFool import DeepFool
from Attacks.IFGSM import IFGSM
from Attacks.Pixle import Pixle

device = torch.device("cpu")
model = MNISTNeuralNetwork(device, train_split=0.8, batch_size=64).to(device)

# Between 1e-3 and 1e-5
learning_rate = 1e-4
epochs = 20
loss_function = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), learning_rate)

# for e in range(epochs):
#     print(f"Epoch {e+1}")
#     history = model.train_model(loss_function=loss_function, optimizer=optimizer)
#     print(history)
#     print("-"*50)

# cr, preds = model.test_model(loss_function)
# print(cr)

# Adverserial Example generation and training

model.load_model()

ifgsm_attack = IFGSM(model, device, targeted=False, loss_function=loss_function, optimizer=optimizer, eps=0.1, max_steps=20)
cw_attack = CW(model, device, targeted=False, search_steps=5, max_steps=20, confidence=0, lr=learning_rate, loss_function=loss_function, optimizer=optimizer)
deepfool_attack = DeepFool(model, device, targeted=False, step=0.02, max_iter=1000, loss_function=loss_function, optimizer=optimizer)
pixle_attack = Pixle(model, device, targeted=False, attack_type=0, loss_function=loss_function, optimizer=optimizer, max_steps=0, max_patches=0)

adverserials = AdverserialExamples(model=None, device=device, ifgsm=ifgsm_attack, cw=cw_attack, deepfool=deepfool_attack, pixle=pixle_attack, dataset="MNIST", learning_rate=learning_rate, loss_function=loss_function)

for e in range(epochs):
    print(f"Epoch {e+1}")
    history = adverserials.train_model_adverserial_examples(loss_function, optimizer)
    print(history)
    print("-"*50)

cr, preds = adverserials.test_adverserials(loss_function)
print(cr)

mnist_distiller = Distiller(model, device, "MNIST", learning_rate, loss_function)
for e in range(epochs):
    print(f"Epoch {e+1}")
    teacher_history, student_history = mnist_distiller.train_distillation()
    print(teacher_history)
    print(student_history)
    print("-"*50)