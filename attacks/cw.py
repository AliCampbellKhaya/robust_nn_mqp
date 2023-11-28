import torch
from Attacks.BaseAttack import BaseAttack

class CW(BaseAttack):
    def __init__(self, model, device, targeted, c, confidence, max_steps):
        super().__init__("CW", model, device, targeted)
        self.c = c
        self.max_steps = max_steps

    def forward(self, inputs, labels, loss_function, optimizer):
        inputs.requires_grad = True

        for _ in range(self.max_steps):
            output = self.model(inputs)

            # TODO: Make Targeted
            # If Targeted loss is:
            #   -( loss(output, target) - loss(output(current pred/argmax)))

            loss = loss_function(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            input_grad = input.grad.data

            normalization = torch.norm(input_grad.view(input_grad.size(0), -1), dim=1, keepdim=True)

            perturbation = input_grad * (self.c / normalization)
            perturbed_input = input + perturbation

            # TODO: Make Targeted
            # If Targeted:
            # I think I want to check for when the image is missclassified before max steps is reached
            # Will check back later and update

        return perturbed_input
