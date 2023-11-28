import torch
from Attacks.BaseAttack import BaseAttack

class FGSM(BaseAttack):
    def __init__(self, model, device, targeted, loss_function, optimizer, eps):
        super(FGSM, self).__init__("FGSM", model, device, targeted, loss_function, optimizer)
        self.eps = eps

    def forward(self, input, label):
        input_grad = input.grad.data
        input_denorm = self.denormalize(input)

        sign_data_grad = input_grad.sign()

        perturbed_input = input_denorm + self.eps * sign_data_grad
        perturbed_input = torch.clamp(perturbed_input, 0, 1)

        return self.normalize(perturbed_input)
    