import torch
import numpy as np
from Attacks.BaseAttack import BaseAttack

"""
TODO: Write comment explaining attack
"""

class IFGSM(BaseAttack):
    def __init__(self, model, device, targeted, loss_function, optimizer, eps, max_steps):
        super(IFGSM, self).__init__("IFGSM", model, device, targeted, loss_function, optimizer)
        self.eps = eps
        self.max_steps = max_steps

    def forward_individual(self, input, label):
        label = label.unsqueeze(0)

        shape = input.cpu().numpy().shape
        steps = 0
        pert_image = input.detach().clone()
        x = pert_image[None, :].requires_grad_(True)
        
        total_pert = torch.zeros(shape).to(self.device)
        #total_pert = np.zeros(shape)

        if self.targeted:
            target_label = self.get_target_label()

            
        attack_label = label.cpu().numpy().item()

        while attack_label == label.cpu().numpy().item() and steps < self.max_steps:

            init_pred = self.model(x)

            if self.targeted:
                loss = -self.loss_function(init_pred, target_label)
            else:
                loss = self.loss_function(init_pred, label)

            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()

            x_grad = x.grad.data
            sign_data_grad = x_grad.sign()

            total_pert = total_pert + self.eps * sign_data_grad
            pert_image = x + self.eps * sign_data_grad
            x = torch.clamp(pert_image, 0, 1)
            #x = pert_image
            # total_pert = total_pert + self.eps * sign_data_grad
            # pert_image = input + total_pert
            # x = torch.clamp(pert_image, 0,  1)


            x.requires_grad_(True).retain_grad()

            attack_pred = self.model(x)
            attack_label = attack_pred.argmax(axis=1).cpu().numpy().item()

            steps += 1
            # print(steps)

        print(attack_label, label.cpu().numpy().item())

        return x, label.cpu().numpy().item(), attack_label, steps, total_pert
    