import torch
import copy
import numpy as np
from torch.autograd import Variable

from Attacks.BaseAttack import BaseAttack

"""
TODO: Write comment explaining attack
TODO: Move DeepFool into here
"""

class DeepFool(BaseAttack):
    def __init__(self, model, device, targeted, step, maxIter):
        super().__init__("DeepFool", model, device, targeted)
        self.step = step
        self.maxIter = maxIter

    def forward(self, input, label):
      fwd = omit_softmax(input)
      fwd_img = fwd.data.cpu().numpy().flatten()

      classes_indexes = (np.array(fwd_img)).flatten().argsort()[::-1]
      classes_indexes = classes_indexes[0:self.model.num_classes]
      label = classes_indexes[0]
      shape = input.cpu().numpy().shape
      pert_image = copy.deepcopy(input)
      direction = np.zeros(shape)
      total_pert = np.zeros(shape)
      iLoops = 0
      x = Variable(pert_image[None, :], requires_grad=True)

      logits = omit_softmax(x)

      preds = [logits[0, classes_indexes[k]] for k in range(self.model.num_classes)]
      attack_label = label
      while attack_label == label and iLoops < self.maxIter:
        min_pert = np.inf
        logits[0, classes_indexes[0]].backward(retain_graph=True)
        old_grad = x.grad.data.cpu().numpy().copy()

        for k in range(i, self.model.num_classes):
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
        pert_image = input + (1 + step) * torch.from_numpy(total_pert)
        x = Variable(pert_image, requires_grad=True)

        logits = omit_softmax(x)

        attack_label = np.argmax(logits.data.cpu().numpy().flatten())
        iLoops += 1

      total_pert = (1 + step) * total_pert

      #return total_pert, iLoops, label, attack_label, pert_image
      return pert_image


    def omit_softmax(self, input):
        fwd = self.model.conv_layer1(Variable(input[None, :, :, :], requires_grad=True))
        fwd = self.model.conv_layer2(fwd)
        fwd = fwd.view(fwd.size(0), -1)
        fwd = self.model.fc_layer(fwd)

        return fwd


