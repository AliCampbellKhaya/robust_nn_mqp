import torch
from Attacks.BaseAttack import BaseAttack

class FGSM(BaseAttack):
    def __init__(self, model, device, targeted, eps):
        super().__init__("FGSM", model, device, targeted)
        self.eps = eps

    def forward(self, inputs, label):
        input_grad = input.grad.data
        sign_data_grad = input_grad.sign()

        perturbed_input = inputs + self.eps * sign_data_grad
        perturbed_input = torch.clamp(perturbed_input, 0, 1)

        return perturbed_input
    