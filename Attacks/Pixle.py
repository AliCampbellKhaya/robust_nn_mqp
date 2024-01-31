import torch
import numpy as np
from torch.autograd import Variable
import copy

from Attacks.BaseAttack import BaseAttack

"""
TODO: Write comment explaining attack
TODO: Move Pixle into here
"""

class Pixle(BaseAttack):
    def __init__(self, model, device, targeted, attack_type):
        super().__init__("Pixle", model, device, targeted)
        self.attack_type = attack_type

    def forward(self, input, labels):

        if self.attack_type == 0:
            return forward_basic(input, labels)
        
        elif self.attack_type == 1:
            return forward_random_rows(input, labels)
        
        elif self.attack_type == 2:
            return forward_random()
        
        else:
            print("Please enter valid attack type (0-2)")

    def forward_basic(self, input, labels):
        flat_img = input.view(-1)

        indexes = []
        counter = 0
        for item in flat_img:
            indexes.append([item, counter])
            counter += 1

        indexes = sorted(indexes)
        attack_img = flat_img
        swap = 0

        counter = 0
        for counter in range(len(indexes)):
            if counter % 2 == 0:
                counter += 1
                continue

            attack_img[indexes[counter][1]] = indexes[counter-1][0]
            attack_img[counter-1] = indexes[counter-1][0]
            counter += 1

        attack_img = attack_img.view(input.size())
        return attack_img
    
    def forward_random_rows(self, input, labels, start_row, end_row):
        section = input[:, start_row:end_row, :].view(-1)
        indexes = torch.randperm(section.size(0))
        shuffled = section[indexes]
        input[:, start_row:end_row, :] = shuffled.view(input[:, start_row:end_row, :].size())
        return input
    
    def forward_random(self, input, labels):
        flat_image = input.view(-1)
        indexes = torch.randperm(flat_image.size(0))
        attack_image = flat_image[indexes]
        attack_image = attack_image.view(input.size())
        return attack_image