import torch
import numpy as np

from Attacks.BaseAttack import BaseAttack

"""
TODO: Write comment explaining attack
TODO: Move Pixle into here
"""

class Pixle(BaseAttack):
    def __init__(self, model, device, targeted, attack_type, loss_function, optimizer):
        super().__init__("Pixle", model, device, targeted, loss_function, optimizer)
        self.attack_type = attack_type

    def forward_individual(self, input, label):

        if self.attack_type == 0:
            return self.forward_basic(input, label)
        
        elif self.attack_type == 1:
            return self.forward_random_rows(input, label)
        
        elif self.attack_type == 2:
            return self.forward_random()
        
        else:
            print("Please enter valid attack type (0-2)")

    def forward_basic(self, input, label):
        label = label.unsqueeze(0)
        pert_image = input.detach().clone()
        x = pert_image[None, :]


        #flat_img = x.view(-1)
        #print(x.size())
        flat_img = x.flatten(start_dim=2)
        #print(flat_img.size())


        indexes = []
        counter = 0
        print(flat_img[0].size())
        for i in range(len(flat_img[0])):
            #print(item.size())
            indexes.append([flat_img[0][0][i] + flat_img[0][1][i] + flat_img[0][2][i], counter, [flat_img[0][0][i] , flat_img[0][1][i] , flat_img[0][2][i]]])
            counter += 1

        indexes = sorted(indexes)
        attack_img = flat_img[0]
        swap = 0


        #print(attack_img)

        counter = 0
        for counter in range(len(indexes)):
            if counter % 2 == 0:
                counter += 1
                continue

            #print(attack_img[indexes[counter][1]] )
            #print(attack_img)

            print(torch.Tensor(indexes[counter-1][2]).size())

            attack_img[indexes[counter][1]] = torch.Tensor(indexes[counter-1][2])
            attack_img[counter - 1] = torch.Tensor(indexes[counter-1][2])
            counter += 1

        # print(attack_img)
        # print("-"*50)
        print(attack_img.size())
        attack_img = attack_img.view(x.size())

        attack_pred = self.model(attack_img)
        attack_label = attack_pred.argmax(axis=1).cpu().numpy().item()

        return attack_img, label.cpu().numpy().item(), attack_label, 0, torch.zeros(attack_img.cpu().numpy().shape)
    
    def forward_random_rows(self, input, label, start_row, end_row):
        section = input[:, start_row:end_row, :].view(-1)
        indexes = torch.randperm(section.size(0))
        shuffled = section[indexes]
        input[:, start_row:end_row, :] = shuffled.view(input[:, start_row:end_row, :].size())
        return input
    
    def forward_random(self, input, label):
        flat_image = input.view(-1)
        indexes = torch.randperm(flat_image.size(0))
        attack_image = flat_image[indexes]
        attack_image = attack_image.view(input.size())
        return attack_image