import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from Attacks.BaseAttack import BaseAttack

"""
TODO: Write comment explaining attack
"""

class CW(BaseAttack):
    def __init__(self, model, device, targeted, init_const, binary_search_steps, confidence, lr, max_steps, loss_function, optimizer):
        super(CW, self).__init__("CW", model, device, targeted, loss_function, optimizer)
        self.init_const = init_const
        self.binary_search_steps = binary_search_steps
        self.confidence = confidence
        self.lr = lr
        self.max_steps = max_steps

    def forward_individual(self, input, label):
        label = label.unsqueeze(0)

        boxmax = 0.5
        boxmin = -0.5
        boxmul = (boxmax - boxmin) / 2.0
        boxplus = (boxmin + boxmax) / 2.0

        #x = np.arctanh( (input - boxplus) / boxmul * 0.99999)
        pert_image = input.detach().clone()
        x = pert_image[None, :].requires_grad_(True)

        lower_bound_const = np.zeros(1)
        const = np.ones(1) * self.init_const
        upper_bound_const = np.ones(1) * 1e10
        #upper_bound_const = np.ones(1)

        best_l2 = 1e10
        best_score = -1
        best_attack = torch.zeros(x.shape)
        best_pert = torch.zeros(x.shape)

        attack_label = label.cpu().numpy().item()

        for outer_step in range(self.binary_search_steps):

            const = (lower_bound_const + upper_bound_const) / 2

            i_best_l2 = 1e10
            i_best_score = -1

            prev = np.inf

            optimizer_x = x.requires_grad_(True)
            optimizer = optim.Adam([optimizer_x], lr=self.lr)

            for inner_step in range(self.max_steps):

                init_pred = self.model(optimizer_x)

                one_hot = F.one_hot(label, num_classes=init_pred.size(-1)).float()  
                
                i = torch.max( (1 - one_hot) * init_pred, dim=1)[0].detach()
                j = torch.max(one_hot * init_pred, dim=1)[0].detach()

                orig_loss = self.loss_function(init_pred, label)
                attack_loss = torch.sum( torch.clamp( (i - j), min=self.confidence) * const )
                loss = orig_loss + attack_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                pert = x.detach() + (optimizer_x.detach() - x.detach()).clamp(-0.3, 0.3)
                optimizer_x = torch.clamp(pert, 0, 1)

            attack_pred = self.model(optimizer_x)
            attack_label = attack_pred.argmax(axis=1).cpu().numpy().item()

            if attack_label != label.cpu().numpy().item():
                upper_bound_const = const
                best_attack = optimizer_x
                best_pert = pert
            else:
                lower_bound_const = const

        # print(type(best_attack))
        # print(best_attack.size)

        return best_attack, label.cpu().numpy().item(), attack_label, outer_step * inner_step, best_pert







    
    # def forward_individual(self, input, label):
    #     pert_image = input.detach().clone()
    #     x = pert_image[None, :].requires_grad_(True) 
    #     label = label.unsqueeze(0)
    #     steps = 0
    #     attack_label = label.cpu().numpy().item()
    #     shape = input.cpu().numpy().shape
    #     total_pert = np.zeros(shape)

    #     while attack_label == label.cpu().numpy().item() and steps <self.max_steps:
    #         init_pred = self.model(x)

    #         loss = self.loss_function(init_pred, label)

    #         self.optimizer.zero_grad()
    #         loss.backward(retain_graph=True)
    #         self.optimizer.step()

    #         # Checks to see if image is missclassified - ie attack has worked, so terminate for loop
    #         # if init_pred.argmax(1)[0] != label[0]:
    #         #     break

    #         x_grad = x.grad.data

    #         normalization = torch.norm(x_grad, dim=1, keepdim=True)

    #         perturbation = x_grad * (self.c / normalization)
    #         total_pert = np.float32(total_pert + perturbation.numpy())

    #         pert_image = x + perturbation

    #         # Clip perturbed input to be within the valid range [0, 1]
    #         x = torch.clamp(pert_image, 0, 1)

    #         x.requires_grad_(True).retain_grad()

    #         attack_pred = self.model(x)
    #         attack_label = attack_pred.argmax(axis=1).cpu().numpy().item()

    #         steps += 1

    #         #if (self.model(input) != labels)

    #     return x, label.cpu().numpy().item(), attack_label, steps, total_pert
        

        

    # def forward(self, input, labels):
    #     input = input.clone().detach().to(self.device)
    #     labels = labels.clone().detach().to(self.device)

    #     input.requires_grad = True

    #     for _ in range(self.max_steps):
    #         output = self.model(input)

    #         output_labels = torch.eye(output.shape[1]).to(self.device)[labels]

    #         # most likely pred other than target
    #         closest_logit = torch.max((1 - output_labels) * output, dim=1)[0]
    #         # actual pred
    #         actual_logit = torch.max(output_labels * output, dim=1)[0]

    #         if self.targeted:
    #             f_loss = torch.clamp((closest_logit - actual_logit), min = -self.confidence) # min = -kappa
    #         else:
    #             f_loss =  torch.clamp((actual_logit - closest_logit), min = -self.confidence)

    #         # TODO: Make Targeted
    #         # If Targeted loss is:
    #         #   -( loss(output, target) - loss(output(current pred/argmax)))

    #         init_loss = self.loss_function(output, labels)
    #         total_loss = init_loss + self.c * f_loss

    #         self.optimizer.zero_grad()
    #         init_loss.backward()
    #         self.optimizer.step()

    #         input_grad = input.grad.data

    #         #normalization = torch.norm(input_grad.view(input_grad.size(0), -1), dim=1, keepdim=True)
    #         normalization = torch.norm(input_grad, dim=1, keepdim=True)
    #         #normalization = torch.clamp(normalization, 0, 1)
    #         # print(input_grad.shape)
    #         # print(normalization.shape)

    #         perturbation = input_grad * (self.c / normalization)
    #         input = input + perturbation
    #         input = torch.clamp(input, 0, 1)

    #         # TODO: Make Targeted
    #         # If Targeted:
    #         # I think I want to check for when the image is missclassified before max steps is reached
    #         # Will check back later and update

    #     return input
