import torch

from Defenses.BaseDefense import BaseDefense
from scipy.ndimage import gaussian_filter

"""
TODO: Write Defense / Update defense and see what is going on here
"""

class GradientMasking(BaseDefense):
    def __init__(self, model, device, loss_function, epsilon):
        super(GradientMasking, self).__init__("GM", model, device)
        self.loss_function = loss_function
        self.epsilon = epsilon

    def forward_individual(self, input, label):
        input = input[None, :].requires_grad_(True)
        label = label.unsqueeze(0)
        output = self.model(input)

        loss = self.loss_function(output, label)
        self.model.zero_grad()
        loss.backward(retain_graph=True)

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