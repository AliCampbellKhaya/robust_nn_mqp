import torch
from Attacks.BaseAttack import BaseAttack

"""
TODO: Write comment explaining attack
TODO: Look into why results changed when refactored into class
"""

class FGSM(BaseAttack):
    def __init__(self, model, device, targeted, loss_function, optimizer, eps):
        super(FGSM, self).__init__("FGSM", model, device, targeted, loss_function, optimizer)
        self.eps = eps

    def forward_individual(self, input, label):
        input = input[None,:].requires_grad_(True)
        label = label.unsqueeze(0)

        # TODO: Make decision
        if self.targeted:
            target_label = self.get_target_label()

        input.requires_grad = True

        init_pred = self.model(input)

        if self.targeted:
            init_loss = -self.loss_function(init_pred, target_label)
        else:
            init_loss = self.loss_function(init_pred, label)

        input_grad = torch.autograd.grad(init_loss, input, retain_graph=False, create_graph=False)[0]
        sign_data_grad = input_grad.sign()

        perturbed_input = input + self.eps * sign_data_grad
        perturbed_input = torch.clamp(perturbed_input, 0, 1)

        print(f"Attacked Label: {self.model(perturbed_input)}, Actual Label: {label}")

        return perturbed_input
    