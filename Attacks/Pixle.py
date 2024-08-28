import torch
import numpy as np

from Attacks.BaseAttack import BaseAttack

"""
TODO: Write comment explaining attack
TODO: Move Pixle into here
"""

class Pixle(BaseAttack):
    def __init__(self, model, device, targeted, attack_type, max_steps, max_patches, loss_function, optimizer):
        super().__init__("Pixle", model, device, targeted, loss_function, optimizer)
        self.attack_type = attack_type
        self.max_steps = max_steps
        self.max_patches = max_patches

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
        origLabel = self.model(input.unsqueeze(0)).argmax(dim=1)[0].item()
        flatImageR = input[0].view(-1)
        flatImageG = input[1].view(-1)
        flatImageB = input[2].view(-1)
        indexes = []
        counter = 0
        for i in range(len(flatImageR)):
            indexes.append([flatImageR[i] + flatImageG[i] + flatImageB[i], counter])
            counter = counter + 1
        #the image is now stored in a multidimensional array where indexes[number][0] is the value of the given pixel and indexes[number][1] is the index of the pixel in the original image

        #indexes now must be sorted
        indexes = sorted(indexes)
        attackedImageR = flatImageR
        attackedImageG = flatImageG
        attackedImageB = flatImageB
        swapMe = 0  
        counterTwo = 0
        for counterTwo in range(len(indexes)//2):
            if self.model(torch.stack((attackedImageR.view(32,32), attackedImageG.view(32,32), attackedImageB.view(32,32))).unsqueeze(0)).argmax(dim=1)[0].item()!=origLabel:
                break
            #swap pixels of indexes n and length-n
            attackedImageR[indexes[counterTwo][1]] = flatImageR[indexes[1023-counterTwo][1]]
            attackedImageR[1023-counterTwo] = flatImageR[indexes[counterTwo][1]]
            attackedImageG[indexes[counterTwo][1]] = flatImageG[indexes[1023-counterTwo][1]]
            attackedImageG[1023-counterTwo] = flatImageG[indexes[counterTwo][1]]
            attackedImageB[indexes[counterTwo][1]] = flatImageB[indexes[1023-counterTwo][1]]
            attackedImageB[1023-counterTwo] = flatImageB[indexes[counterTwo][1]]
            counterTwo = counterTwo + 1

        attackedImage = torch.stack((attackedImageR.view(32,32), attackedImageG.view(32,32), attackedImageB.view(32,32))).unsqueeze(0)
        attacked_label = self.model(attackedImage).argmax(dim=1)[0].item()
        return attackedImage, origLabel, attacked_label, counterTwo, attackedImage
    
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