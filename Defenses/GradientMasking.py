import torch

from Defenses.BaseDefense import BaseDefense
from scipy.ndimage import gaussian_filter

"""
TODO: Write Defense / Update defense and see what is going on here
"""

class GradientMasking(BaseDefense):
    def __init__(self, model, loss_function, epsilon):
        super(GradientMasking, self).__init__("GM", model)
        self.loss_function = loss_function
        self.epsilon = epsilon

    def forward(self, input, labels):
        input.requires_grad_(True)
        outputs = self.model(input)

        loss = self.loss_function(outputs, labels)
        loss.backward()

        # Add random noise to the gradients
        input_grad = input.grad.detach()
        noise = torch.randn_like(input) * self.epsilon
        input_grad_masked = input_grad + noise

        # Zero out the gradients to prevent them from being used during optimization
        input.grad.zero_()

        # Update the input with the masked gradients
        masked_input = input.detach() + input_grad_masked
        masked_input.requires_grad_(True)

        #return torch.from_numpy(masked_input).float()
        return torch.tensor(masked_input)