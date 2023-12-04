import torch
import torch.nn as nn
from torch.nn import functional as F

from NeuralNetworks.MNISTNeuralNetwork import MNISTNeuralNetwork
from Attacks.FGSM import FGSM
from Attacks.CW import CW

device = torch.device("cuda")
student_model = MNISTNeuralNetwork(device, train_split=0.8, batch_size=64).to(device)
teacher_model = MNISTNeuralNetwork(device, train_split=0.8, batch_size=64).to(device)

# Between 1e-3 and 1e-5
learning_rate = 1e-4
epochs = 5
loss_function = nn.NLLLoss()
temperature = 5
student_optimizer = torch.optim.Adam(student_model.parameters(), learning_rate)
teacher_optimizer = torch.optim.Adam(teacher_model.parameters(), learning_rate)

def distillation_loss(logits_student, logits_teacher):
    # Implement the distillation loss (e.g., cross-entropy)
    soft_targets = F.log_softmax(logits_student / temperature, dim=1)
    return F.kl_div(F.log_softmax(logits_teacher / temperature, dim=1),
                                soft_targets, reduction='batchmean') * temperature
    
print("Training Student")

for e in range(epochs):
    print(f"Epoch {e+1}")
    print(student_model.train_model(loss_function=loss_function, optimizer=student_optimizer))
    print("-"*50)

print("Testing Student")

cr, preds = student_model.test_model(loss_function)
print(cr)

print("Training Teacher")

for e in range(epochs):
    print(f"Epoch {e+1}")
    print(teacher_model.train_model(loss_function=distillation_loss, optimizer=teacher_optimizer))
    print("-"*50)

print("Testing Teacher")

cr, preds = teacher_model.test_model(loss_function)
print(cr)

print("Testing FGSM Student")

fgsm_attack = FGSM(student_model, device, False, loss_function, student_optimizer, 0.1)
cr, preds = student_model.test_attack_model(loss_function, fgsm_attack)
print(cr)

print("Testing FGSM Teacher")
fgsm_attack = FGSM(teacher_model, device, False, loss_function, teacher_optimizer, 0.1)
cr, preds = teacher_model.test_attack_model(loss_function, fgsm_attack)
print(cr)

# fgsm_attack = FGSM(model, device, False, loss_function, optimizer, 0.2)
# cr, preds = model.test_attack_model(loss_function, fgsm_attack)
# print(cr)

# cw_attack = CW(model, device, False, 0.1, 20, loss_function, optimizer)
# cr, preds = model.test_attack_model(loss_function, cw_attack)
# print(cr)