import torch
import BaseAttack

class FGSM(BaseAttack):
    def __init__(self, model, device, eps):
        super().__init__("FGSM", model, device)
        self.eps = eps

    def forward(self, image, label):
        image_grad = image.grad.data

        sign_data_grad = image_grad.sign()

        perturbed_image = image + self.eps * sign_data_grad
        perturbed_image = torch.clamp(perturbed_image, 0, 1)

        return perturbed_image

    