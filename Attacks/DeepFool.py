import torch
import numpy as np

from Attacks.BaseAttack import BaseAttack

"""
TODO: Write comment explaining attack
"""

class DeepFool(BaseAttack):
    def __init__(self, model, device, targeted, step, max_iter, loss_function, optimizer):
        super().__init__("DeepFool", model, device, targeted, loss_function, optimizer)
        self.step = step
        self.max_iter = max_iter


    def forward_individual(self, input, labels):
        x_0 = input[None, :, :, :].requires_grad_(True)

        fwd = self.model.forward_omit_softmax(x_0)
        fwd_img = fwd.data.cpu().numpy().flatten()

        classes_indexes = fwd_img.flatten().argsort()[::-1]
        classes_indexes = classes_indexes[0:self.model.num_classes]

        label = classes_indexes[0]
        shape = input.cpu().numpy().shape
        pert_image = input.detach().clone()
        direction = np.zeros(shape)
        total_pert = np.zeros(shape)
        iLoops = 0
        x = pert_image[None, :].requires_grad_(True)

        logits = self.model.forward_omit_softmax(x)


        preds = [logits[0, classes_indexes[k]] for k in range(self.model.num_classes)]
        attack_label = label
        while attack_label == label and iLoops < self.max_iter:
          min_pert = np.inf
          logits[0, classes_indexes[0]].backward(retain_graph=True)
          old_grad = x.grad.data.cpu().numpy().copy()

          for k in range(1, self.model.num_classes):
            x.grad = None
            logits[0, classes_indexes[k]].backward(retain_graph=True)
            new_grad = x.grad.data.cpu().numpy().copy()

            pert_direction = new_grad - old_grad
            distance = (logits[0, classes_indexes[k]] - logits[0, classes_indexes[0]]).data.cpu().numpy()
            pert_class = abs(distance)/np.linalg.norm(pert_direction.flatten())

            if pert_class < min_pert:
              min_pert = pert_class
              direction = pert_direction
          
          pert_ILoop = (min_pert+1e-4) * direction / np.linalg.norm(direction)
          total_pert = np.float32(total_pert + pert_ILoop)
          pert_image = input + (1 + self.step) * torch.from_numpy(total_pert)
          x = pert_image.requires_grad_(True) 

          logits = self.model.forward_omit_softmax(x)

          attack_label = np.argmax(logits.data.cpu().numpy().flatten())
          iLoops += 1

        total_pert = (1 + self.step) * total_pert

        return pert_image, labels.cpu().numpy().item(), attack_label, iLoops, total_pert

