import torch
from Attacks.BaseAttack import BaseAttack

class IFGSM(BaseAttack):
    def __init__(self, model, device, targeted, loss_function, optimizer, eps, max_steps):
        super(IFGSM, self).__init__("IFGSM", model, device, targeted, loss_function, optimizer)
        self.eps = eps
        self.max_steps = max_steps

    def forward(self, input, label):
        input = input.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        if self.targeted:
            target_label = self.get_target_label()

        input.requires_grad = True

        for steps in range(self.max_steps):
            init_pred = self.model(input)

            if self.targeted:
                loss = -self.loss_function(init_pred, target_label)
            else:
                loss = self.loss_function(init_pred, label)

            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()

            # Checks to see if image is missclassified - ie attack has worked, so terminate for loop
            #if init_pred.argmax(1)[0] != label[0]:
            # if torch.equal(init_pred.argmax(1), label):
            #     break

            input_grad = input.grad.data
            sign_data_grad = input_grad.sign()

            input = input + self.eps * sign_data_grad
            input = torch.clamp(input, 0, 1)

            input.requires_grad_(True).retain_grad()

            #print(f"Attacked Label: {self.model(input)}, Actual Label: {label}")

        return input
    