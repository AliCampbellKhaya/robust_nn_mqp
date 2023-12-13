import torch
from Attacks.BaseAttack import BaseAttack

class FGSM(BaseAttack):
    def __init__(self, model, device, targeted, loss_function, optimizer, eps):
        super(FGSM, self).__init__("FGSM", model, device, targeted, loss_function, optimizer)
        self.eps = eps

    # def forward(self, input, label):
    #     input_grad = input.grad.data

    #     sign_data_grad = input_grad.sign()

    #     perturbed_input = input + self.eps * sign_data_grad
    #     perturbed_input = torch.clamp(perturbed_input, 0, 1)

    #     return perturbed_input
    
    # def forward(self, input, label):
    #     input.requires_grad = True

    #     init_pred = self.model(input)
    #     init_loss = self.loss_function(init_pred, label)

    #     self.model.zero_grad()
    #     init_loss.backward()

    #     input_grad = input.grad.data
    #     #input_denorm = self.denormalize(input)

    #     sign_data_grad = input_grad.sign()

    #     perturbed_input = input + self.eps * sign_data_grad
    #     perturbed_input = torch.clamp(perturbed_input, 0, 1)

    #     #return self.normalize(perturbed_input)
    #     return perturbed_input

    def forward(self, input, label):
        input.requires_grad = True

        init_pred = self.model(input)
        init_loss = self.loss_function(init_pred, label)

        input_grad = torch.autograd.grad(init_loss, input, retain_graph=False, create_graph=False)[0]
        sign_data_grad = input_grad.sign()

        perturbed_input = input + self.eps * sign_data_grad
        perturbed_input = torch.clamp(perturbed_input, 0, 1)

        return perturbed_input
    