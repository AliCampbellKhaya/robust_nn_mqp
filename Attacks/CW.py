import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from Attacks.BaseAttack import BaseAttack

import matplotlib.pyplot as plt

"""
TODO: Write comment explaining attack
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

        optimal_best_l2 = torch.ones(1) * 1e10
        optimal_best_l2_pred = -torch.ones(1)
        optimal_best_image = x.detach().clone()

        # print("x before tanh")
        # print(x)
        x_tanh = self.to_tanh_space(x)
        # print("x after tanh")
        # print(x_tanh)

        # plt.subplot(1, 2, 1)
        # plt.imshow(x_tanh.squeeze().detach().permute(1, 2, 0))
        # plt.subplot(1, 2, 2)
        # plt.imshow(x.squeeze().detach().permute(1, 2, 0))
        # plt.show()
        
        label_one_hot = F.one_hot(label, self.model.num_classes)

        pert_tanh = torch.zeros(x.size()).requires_grad_(True)
        self.optimizer = optim.Adam([pert_tanh], lr=self.lr)

        for outer_step in range(self.search_steps):
            
            best_l2 = torch.ones(1) * 1e10
            best_l2_pred = -torch.ones(1)
            prev_loss = np.inf

            for inner_step in range(self.max_steps):
                pert_image, pert_norm, pert_output, loss = self.optimize(x_tanh, label_one_hot, pert_tanh, const)

                #TODO: Early abort
                # print("pert_image")
                # print(type(pert_image))

                # print("pert_norm")
                # print(type(pert_norm))

                # print("pert_output")
                # print(type(pert_output))

                # print("loss")
                # print(type(loss))

                pert_pred = pert_output.detach().argmax(1)

                # print("pert_pred")
                # print(pert_pred)
                # print("pert_output")
                # print(pert_output)

                # print("pert_norm")
                # print(pert_norm)
                # print("best_l2")
                # print(best_l2)
                # print(best_l2[0])

                if (pert_pred.cpu().numpy().item() != label.cpu().numpy().item()):

                    if pert_norm < best_l2[0]:
                        best_l2[0] = pert_norm
                        best_l2_pred[0] = pert_pred

                    if pert_norm < optimal_best_l2[0]:
                        optimal_best_l2[0] = pert_norm
                        optimal_best_l2_pred = pert_pred
                        optimal_best_image = pert_image

            if best_l2_pred[0] != -1:
                if const[0] < upper_bound_const[0]:
                    upper_bound_const[0] = const[0]

                if upper_bound_const[0] < 1e10 * 0.1:
                    const[0] = (lower_bound_const[0] + upper_bound_const[0]) / 2

            else:
                if const[0] > lower_bound_const[0]:
                    lower_bound_const[0] = const[0]

                if upper_bound_const[0] < 1e10 * 0.1:
                    const[0] = (lower_bound_const[0] + upper_bound_const[0]) / 2
                
                else:
                    const[0] = const[0] * 10

        return optimal_best_image, label.cpu().numpy().item(), optimal_best_l2_pred.cpu().numpy().item(), outer_step * inner_step, pert_output


    def optimize(self, x, label_one_hot, pert, const):

        pert_image = self.from_tanh_space(x + pert)
        pert_output = self.model.forward_omit_softmax(pert_image)
        orig_image = self.from_tanh_space(x)

        # print(pert_image)
        # print(x)

        # print("x after tanh")
        # print(x)
        # print("x from tanh")
        # print(orig_image)
        # print("pert_image")
        # print(pert_image)

        # plt.subplot(1, 3, 1)
        # plt.imshow(x.squeeze().detach().permute(1, 2, 0))
        # plt.subplot(1, 3, 2)
        # plt.imshow(orig_image.squeeze().detach().permute(1, 2, 0))
        # plt.subplot(1, 3, 3)
        # plt.imshow(pert_image.squeeze().detach().permute(1, 2, 0))
        # plt.show()

        pert_norm = torch.pow(pert_image - orig_image, 2)
        pert_norm = torch.sum(pert_norm.view(pert_norm.size(0), -1), 1)

        real = torch.sum(label_one_hot * pert_output, 1)
        # replace np.inf with 1e4
        other = torch.max(((1 - label_one_hot) * pert_output - label_one_hot * 1e4), 1)[0]

        #TODO: Targeted
        f = torch.clamp(real - other + self.confidence, min = 0.0)

        loss = torch.sum(pert_norm + const * f)

        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()

        loss = loss.data

        return pert_image, pert_norm, pert_output, loss
    
    # def to_tanh_space(self, x):
    #     return torch.atanh(torch.clamp(x * 2 - 1, min = -1, max = 1))
    
    # def from_tanh_space(self, x):
    #     return 0.5 * ( torch.tanh(x) + 1 )

    # def atanh(self, x):
    #     x = x * (1 - 1e-6)
    #     return 0.5 * torch.log( (1.0 + x) / (1.0 - x) )

    def to_tanh_space(self, x):

        boxmax = 0.5
        boxmin = -0.5

        boxmul = (boxmax - boxmin) / 2.0
        boxplus = (boxmin + boxmax) / 2.0

        # return self.atanh( (x - boxplus) / boxmul)
        return torch.atanh( (x - boxplus) / boxmul)
    
    def from_tanh_space(self, x):

        boxmax = 0.5
        boxmin = -0.5

        boxmul = (boxmax - boxmin) / 2.0
        boxplus = (boxmin + boxmax) / 2.0

        return torch.tanh(x) * boxmul + boxplus


# class CW(BaseAttack):
#     def __init__(self, model, device, targeted, init_const, binary_search_steps, confidence, lr, max_steps, loss_function, optimizer):
#         super(CW, self).__init__("CW", model, device, targeted, loss_function, optimizer)
#         self.init_const = init_const
#         self.binary_search_steps = binary_search_steps
#         self.confidence = confidence
#         self.lr = lr
#         self.max_steps = max_steps

#     def forward_individual(self, input, label):
#         label = label.unsqueeze(0)

#         boxmax = 0.5
#         boxmin = -0.5
#         boxmul = (boxmax - boxmin) / 2.0
#         boxplus = (boxmin + boxmax) / 2.0

#         #x = np.arctanh( (input - boxplus) / boxmul * 0.99999)
#         pert_image = input.detach().clone()
#         x = pert_image[None, :].requires_grad_(True)

#         lower_bound_const = np.zeros(1)
#         const = np.ones(1) * self.init_const
#         upper_bound_const = np.ones(1) * 1e10
#         #upper_bound_const = np.ones(1)

#         best_l2 = 1e10
#         best_score = -1
#         best_attack = torch.zeros(x.shape)
#         best_pert = torch.zeros(x.shape)

#         attack_label = label.cpu().numpy().item()

#         for outer_step in range(self.binary_search_steps):

#             const = (lower_bound_const + upper_bound_const) / 2

#             i_best_l2 = 1e10
#             i_best_score = -1

#             prev = np.inf

#             optimizer_x = x.requires_grad_(True)
#             optimizer = optim.Adam([optimizer_x], lr=self.lr)

#             for inner_step in range(self.max_steps):

#                 init_pred = self.model(optimizer_x)

#                 one_hot = F.one_hot(label, num_classes=init_pred.size(-1)).float()  
                
#                 i = torch.max( (1 - one_hot) * init_pred, dim=1)[0].detach()
#                 j = torch.max(one_hot * init_pred, dim=1)[0].detach()

#                 orig_loss = self.loss_function(init_pred, label)
#                 attack_loss = torch.sum( torch.clamp( (i - j), min=self.confidence) * const )
#                 loss = orig_loss + attack_loss

#                 self.optimizer.zero_grad()
#                 loss.backward()
#                 self.optimizer.step()

#                 pert = x.detach() + (optimizer_x.detach() - x.detach()).clamp(-0.3, 0.3)
#                 optimizer_x = torch.clamp(pert, 0, 1)

#             attack_pred = self.model(optimizer_x)
#             attack_label = attack_pred.argmax(axis=1).cpu().numpy().item()

#             if attack_label != label.cpu().numpy().item():
#                 upper_bound_const = const
#                 best_attack = optimizer_x
#                 best_pert = pert
#             else:
#                 lower_bound_const = const

#         # print(type(best_attack))
#         # print(best_attack.size)

#         return best_attack, label.cpu().numpy().item(), attack_label, outer_step * inner_step, best_pert







    
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
