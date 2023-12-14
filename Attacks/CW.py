import torch
from Attacks.BaseAttack import BaseAttack

class CW(BaseAttack):
    def __init__(self, model, device, targeted, c, confidence, max_steps, loss_function, optimizer):
        super(CW, self).__init__("CW", model, device, targeted, loss_function, optimizer)
        self.c = c
        self.confidence = confidence
        self.max_steps = max_steps

    def forward(self, input, label):
        input = input.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        input.requires_grad = True

        for _ in range(self.max_steps):
            init_pred = self.model(input)

            loss = self.loss_function(init_pred, label)

            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()

            # Checks to see if image is missclassified - ie attack has worked, so terminate for loop
            # if init_pred.argmax(1)[0] != label[0]:
            #     break

            input_grad = input.grad.data

            normalization = torch.norm(input_grad, dim=1, keepdim=True)

            perturbation = input_grad * (self.c / normalization)

            input = input + perturbation

            # Clip perturbed input to be within the valid range [0, 1]
            input = torch.clamp(input, 0, 1)

            input.requires_grad_(True).retain_grad()

            #if (self.model(input) != labels)

        return input

    # def forward(self, input, labels):
    #     input = input.clone().detach().to(self.device)
    #     labels = labels.clone().detach().to(self.device)

    #     input.requires_grad = True

    #     for _ in range(self.max_steps):
    #         output = self.model(input)

    #         output_labels = torch.eye(output.shape[1]).to(self.device)[labels]

    #         # most likely pred other than target
    #         closest_logit = torch.max((1 - output_labels) * output, dim=1)[0]
    #         # actual pred
    #         actual_logit = torch.max(output_labels * output, dim=1)[0]

    #         if self.targeted:
    #             f_loss = torch.clamp((closest_logit - actual_logit), min = -self.confidence) # min = -kappa
    #         else:
    #             f_loss =  torch.clamp((actual_logit - closest_logit), min = -self.confidence)

    #         # TODO: Make Targeted
    #         # If Targeted loss is:
    #         #   -( loss(output, target) - loss(output(current pred/argmax)))

    #         init_loss = self.loss_function(output, labels)
    #         total_loss = init_loss + self.c * f_loss

    #         self.optimizer.zero_grad()
    #         init_loss.backward()
    #         self.optimizer.step()

    #         input_grad = input.grad.data

    #         #normalization = torch.norm(input_grad.view(input_grad.size(0), -1), dim=1, keepdim=True)
    #         normalization = torch.norm(input_grad, dim=1, keepdim=True)
    #         #normalization = torch.clamp(normalization, 0, 1)
    #         # print(input_grad.shape)
    #         # print(normalization.shape)

    #         perturbation = input_grad * (self.c / normalization)
    #         input = input + perturbation
    #         input = torch.clamp(input, 0, 1)

    #         # TODO: Make Targeted
    #         # If Targeted:
    #         # I think I want to check for when the image is missclassified before max steps is reached
    #         # Will check back later and update

    #     return input
