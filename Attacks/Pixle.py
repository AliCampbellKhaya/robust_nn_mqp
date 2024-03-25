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
        label = label.unsqueeze(0)
        pert_image = input.detach().clone()
        x = pert_image[None, :]

        x_bounds = (2, 10)
        y_bounds = (2, 10)

        best_image = x.clone()
        best_loss = self.get_loss(x, label)

        for outer_step in range(self.max_steps):

            for inner_step in range(self.max_patches):
                x_coord, y_coord, x_offset, y_offset = self.get_patches(x, x_bounds, y_bounds)

                destinations = self.get_patch_mapping(x, best_image, x_coord, x_offset, y_coord, y_offset)

                solution = [x_coord, y_coord, x_offset, y_offset]

                attacked_image = self.generate_perturbation(x, solution, destinations, best_image)

                loss = self.get_loss(attacked_image, label)
                print(loss)

                if loss < best_loss:
                    best_loss = loss
                    best_image = attacked_image

        best_label = torch.argmax(self.model(best_image))

        return best_image, label.cpu().numpy().item(), best_label.detach().cpu().numpy().item(), outer_step * inner_step, torch.zeros(attacked_image.cpu().numpy().shape)

    
    def get_patches(self, image, x_bounds, y_bounds):
        height, width = image.shape[2:]

        x_coord, y_coord = np.random.uniform(0, 1, 2)

        x_offset = np.random.randint(x_bounds[0], x_bounds[1] + 1)
        y_offset = np.random.randint(y_bounds[0], y_bounds[1] + 1)

        x_coord = int(x_coord * (width - 1))
        y_coord = int(y_coord * (height - 1))

        if x_coord + x_offset > width:
            x_offset = width - x_coord

        if y_coord + y_offset > height:
            y_offset = height - y_coord

        return x_coord, y_coord, x_offset, y_offset
    
    # change from random to max probability
    def get_patch_mapping(self, image, destination_image, x_coord, x_offset, y_coord, y_offset):
        height, width = image.shape[2:]
        destinations = []

        # True Random
        # for i in range(x_offset):
        #     for j in range(y_offset):
        #         dx, dy = np.random.uniform(0, 1, 2)
        #         dx = int(dx * (width - 1))
        #         dy = int(dy * (height - 1))
        #         destinations.append([dx, dy])

        for i in np.arange(y_coord, y_coord + y_offset):
            for j in np.arange(x_coord, x_coord + x_offset):
                pixel = image[:, :, i : i +1, j : j + 1]
                diff = destination_image - pixel
                diff = diff[0].abs().mean(0).view(-1)

                # using similarity instead of difference
                diff = 1 / (1 + diff)
                diff[diff == 1] = 0

                probs = torch.softmax(diff, 0).cpu().numpy()
                indexes = np.arange(len(diff))

                pair = None

                pixel_iter = iter( sorted( zip( indexes, probs), key=lambda pit: pit[1], reverse=True ) )

                while True:
                    index = next(pixel_iter)[0]

                    dy, dx = np.unravel_index(index, (height, width))

                    if dy == i and dx == j:
                        continue

                    pair = (dx, dy)
                    break

                destinations.append(pair)

        return destinations

    def generate_perturbation(self, image, solution, destinations, destination_image):
        channels, height, width = image.shape[1:]

        x1, y1, x2, y2 = solution

        targeted_pixels = np.ix_(range(channels), np.arange(y1, y1 + y2), np.arange(x1, x1 + x2))

        perturbation = image[0][targeted_pixels].view(channels, -1)

        indexes = torch.tensor(destinations)
        destination_image = destination_image.detach().clone()
        destination_image[:, :, indexes[:, 0], indexes[:, 1]] = perturbation

        return destination_image
    
    def get_loss(self, image, label):
        prob = self.model.forward(image).clone().detach().cpu().numpy()
        prob = prob[np.arange(len(prob)), label]

        return prob.sum()


    # def forward_basic(self, input, label):
    #     label = label.unsqueeze(0)
    #     pert_image = input.detach().clone()
    #     x = pert_image[None, :]

    #     batch_size, num_channels, height, width = x.size()

    #     flat_image = x.view(-1)
    #     print(flat_image.size())
    #     #flat_image = x.flatten(start_dim=2)
    #     print(flat_image.size())

    #     indexes = []
    #     for b in range(batch_size):
    #         for c in range(num_channels):
    #             for h in range(height):
    #                 for w in range(width):
    #                     indexes.append([flat_image[(c*height*width) + (h*width) + w], (c, h, w)])

    #     indexes = sorted(indexes)

    #     attacked_image = flat_image.clone()

    #     for i in range(1, len(indexes), 2):
    #         idx1 = ((indexes[i-1][0] // (height * width)).type(torch.int64), (indexes[i-1][0] % (height * width)) // width, indexes[i][0] % width)
    #         idx2 = ((indexes[i][0] // (height * width)).type(torch.int64), (indexes[i][0] % (height * width)) // width, indexes[i][0] % width)

    #         print(idx1[0])
    #         print(type(idx1[0]))
    #         print(idx1[0].dtype)

    #         attacked_image[idx1[0]][idx1[1] * height * width + idx1[2]] = indexes[i][0]
    #         attacked_image[idx2[0]][idx2[1] * height * width + idx2[2]] = indexes[i-1][0]


    #     attacked_image = attacked_image.view(batch_size, num_channels, height, width)

    #     attack_pred = self.model(attacked_image)
    #     attack_label = attack_pred.argmax(axis=1).cpu().numpy().item()

    #     return attacked_image, label.cpu().numpy().item(), attack_label, 0, torch.zeros(attacked_image.cpu().numpy().shape)


    # def forward_basic(self, input, label):
    #     label = label.unsqueeze(0)
    #     pert_image = input.detach().clone()
    #     x = pert_image[None, :]


    #     #flat_img = x.view(-1)
    #     #print(x.size())
    #     flat_img = x.flatten(start_dim=2)
    #     #print(flat_img.size())


    #     indexes = []
    #     counter = 0
    #     print(flat_img[0].size())
    #     for i in range(len(flat_img[0])):
    #         #print(item.size())
    #         indexes.append([flat_img[0][0][i] + flat_img[0][1][i] + flat_img[0][2][i], counter, [flat_img[0][0][i] , flat_img[0][1][i] , flat_img[0][2][i]]])
    #         counter += 1

    #     indexes = sorted(indexes)
    #     print(flat_img.size())
    #     attack_img = flat_img
    #     swap = 0


    #     #print(attack_img)
    #     print(attack_img.size())

    #     counter = 0
    #     for counter in range(len(indexes)):
    #         if counter % 2 == 0:
    #             counter += 1
    #             continue

    #         #print(attack_img[indexes[counter][1]] )
    #         #print(attack_img)

    #         print(torch.Tensor(indexes[counter-1][2]).size())

    #         attack_img[0][indexes[counter][1]] = torch.Tensor(indexes[counter-1][2])
    #         attack_img[0][counter - 1] = torch.Tensor(indexes[counter-1][2])
    #         counter += 1

    #     # print(attack_img)
    #     # print("-"*50)
    #     print(attack_img.size())
    #     attack_img = attack_img.view(x.size())

    #     attack_pred = self.model(attack_img)
    #     attack_label = attack_pred.argmax(axis=1).cpu().numpy().item()

    #     return attack_img, label.cpu().numpy().item(), attack_label, 0, torch.zeros(attack_img.cpu().numpy().shape)
    
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