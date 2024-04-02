import torch
import numpy as np
from Attacks.BaseAttack import BaseAttack

"""
TODO: Write comment explaining attack
"""

class IFGSM(BaseAttack):
    def __init__(self, model, device, targeted, loss_function, optimizer, eps, max_steps, decay, alpha):
        super(IFGSM, self).__init__("IFGSM", model, device, targeted, loss_function, optimizer)
        self.eps = eps
        self.max_steps = max_steps
        self.decay = decay
        self.alpha = alpha

    def forward_individual(self, input, label):
        x = input[None,:].requires_grad_(True)
        adv_x = x.detach().clone()
        label = label.unsqueeze(0)
        shape = x.detach().cpu().numpy().shape
        steps = 0
        total_pert = torch.zeros(shape).to(self.device)

        # TODO: Reconsider targeting

        attack_label = label.cpu().numpy().item()
        momentum = torch.zeros(shape).to(self.device)

        while attack_label == label.cpu().numpy().item() and steps < self.max_steps:

            init_output = self.model(x)
            init_pred = init_output.detach().argmax(dim=1)

            if init_pred.item() != label.item():
                attack_label = init_pred
                continue

            loss = self.loss_function(init_output, label)

            self.model.zero_grad()
            loss.backward(retain_graph=True)

            x_grad = x.grad.data
            x_grad = x_grad / torch.mean(torch.abs(x_grad), dim=(1, 2, 3), keepdim=True)
            x_grad = x_grad + momentum * self.decay

            momentum = x_grad
            sign_data_grad = x_grad.sign()

            total_pert = total_pert + self.alpha * sign_data_grad
            adv_x = adv_x.detach() + self.alpha * sign_data_grad
            delta = torch.clamp(adv_x - x, min=-self.eps, max=self.eps)
            x = torch.clamp(x + delta, 0, 1)

            x.requires_grad_(True).retain_grad()

            attack_pred = self.model(x)
            attack_label = attack_pred.detach().argmax(dim=1)

            steps += 1

        return x, label.cpu().numpy().item(), attack_label.cpu().numpy().item(), steps, total_pert
