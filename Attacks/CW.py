import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from Attacks.BaseAttack import BaseAttack

import matplotlib.pyplot as plt

"""
TODO: Write comment explaining attack
1e10 used in place of infinity
"""

class CW(BaseAttack):
    def __init__(self, model, device, targeted, search_steps, max_steps, confidence, lr, loss_function, optimizer):
        super(CW, self).__init__("CW", model, device, targeted, loss_function, optimizer)
        self.search_steps = search_steps
        self.max_steps = max_steps
        self.confidence = confidence
        self.lr = lr  

    def forward_individual(self, input, label):
        x = input[None, :].requires_grad_(True)
        label = label.unsqueeze(0)

        lower_bound_const = torch.zeros(1)
        upper_bound_const = torch.ones(1) * 1e10
        const = torch.ones(1) * 1e-3

        rows = range(len(x))

        x_tanh = self.to_tanh(x)
        x_from_tanh = self.from_tanh(x_tanh)

        best_adv = torch.zeros_like(x)
        best_adv_norm = torch.full_like(x, 1e10)

        for outer_step in range(self.search_steps):
            if outer_step == self.search_steps - 1 and self.search_steps >= 10:
                const = torch.minimum(upper_bound_const, 1e10)

            delta = torch.zeros_like(x_tanh)
            attack_optimizer = optim.Adam([delta], lr=self.lr)

            current_adv = torch.full((1,), fill_value=False)
            previous_loss = 1e10
            previous_loss_abort = np.inf

            #step_const = torch.tensor(x, const)

            for inner_step in range(self.max_steps):
                loss, x_pert, x_pert_logits, model_grads = self.optimize(x_tanh, x_from_tanh, delta, const, rows, attack_optimizer)
                delta -= model_grads

                #Abort Early
                if inner_step % (np.ceil(self.max_steps / 5)) == 0:
                    if not (loss <= 0.9999 * previous_loss_abort):
                        break
                    previous_loss_abort = loss

                # change class logits maybe
                current_adv_iter = x_pert

                current_adv = torch.tensor(torch.abs(previous_loss - loss) > 0.5)

                previous_loss = loss

                norm_iter = torch.flatten(x_pert - x).norm(p=2, dim=-1)
                closest_norm = norm_iter < best_adv_norm
                best_adv_norm_iter = torch.logical_and(closest_norm, current_adv_iter)

                best_adv = torch.where(best_adv_norm_iter, x_pert, best_adv)
                best_adv_norm = torch.where(best_adv_norm_iter, norm_iter, best_adv_norm)                

            lower_bound_const = torch.where(current_adv, lower_bound_const, const)
            upper_bound_const = torch.where(current_adv, const, upper_bound_const)

            const_exp_search = const * 10
            const_binary_search = (lower_bound_const + upper_bound_const) / 2
            const = torch.where(torch.isinf(upper_bound_const), const_exp_search, const_binary_search)

        best_adv_pred = torch.argmax(self.model(best_adv))

        return best_adv, label.cpu().numpy().item(), best_adv_pred.detach().cpu().numpy().item(), outer_step * inner_step, best_adv


    def to_tanh(self, x):
        max_bound = 0.5
        min_bound = -0.5

        x = torch.clamp(x, min_bound, max_bound)
        x = (x - min_bound) / (max_bound - min_bound)
        x = ((x * 2) - 1) * 0.9999
        return torch.atanh(x)
    
    def from_tanh(self, x):

        max_bound = 0.5
        min_bound = -0.5

        x = (torch.tanh(x) / 0.9999 + 1) / 2
        x = x * (max_bound - min_bound) + min_bound
        return torch.clamp(x, 0, 1)
    
    def optimize(self, x_tanh, x_from_tanh, delta, const, rows, attack_optimizer):
        x = self.from_tanh(x_tanh + delta)

        logits = self.model.forward_omit_softmax(x)
        one_hot_like = torch.eye(logits.size(dim=1))
        one_hot_like[self.model.test_data.targets] = 1e10
        other_logits = torch.argmax((logits - one_hot_like), axis=-1)

        c_min = self.model.test_data.targets
        c_max = torch.argmax(other_logits, axis=-1)

        adv_loss = logits[rows, c_min] - logits[rows, c_max]

        adv_loss += self.confidence
        adv_loss = torch.maximum(torch.zeros(1), adv_loss)

        adv_loss *= const

        sq_norm = torch.flatten(x - x_from_tanh).square().sum(axis=-1)
        loss = torch.sum(adv_loss) + torch.sum(sq_norm)

        attack_optimizer.zero_grad()
        loss.backward(retain_graph=True)
        attack_optimizer.step()

        loss_grad = loss.data

        return loss, x, logits, loss_grad
