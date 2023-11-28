import torch
from Attacks.BaseAttack import BaseAttack

class CW(BaseAttack):
    def __init__(self, model, device, targeted, c, max_steps, loss_function, optimizer):
        super(CW, self).__init__("CW", model, device, targeted, loss_function, optimizer)
        self.c = c
        self.max_steps = max_steps

    def forward(self, inputs, labels):
        inputs.requires_grad = True

        for _ in range(self.max_steps):
            output = self.model(inputs)

            # TODO: Make Targeted
            # If Targeted loss is:
            #   -( loss(output, target) - loss(output(current pred/argmax)))

            loss = self.loss_function(output, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            input_grad = input.grad.data

            normalization = torch.norm(input_grad.view(input_grad.size(0), -1), dim=1, keepdim=True)

            perturbation = input_grad * (self.c / normalization)
            perturbed_input = input + perturbation

            # TODO: Make Targeted
            # If Targeted:
            # I think I want to check for when the image is missclassified before max steps is reached
            # Will check back later and update

        return perturbed_input
