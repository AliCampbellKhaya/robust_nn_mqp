import torch
import BaseAttack

class FGSM(BaseAttack):
    def __init__(self, model, device, eps):
        super().__init__("FGSM", model, device)
        self.eps = eps

    def forward(self, image, label):
        image_grad = image.grad.data
        image_denorm = self.denorm(image)

        sign_data_grad = image_grad.sign()

        perturbed_image = image + self.eps * sign_data_grad
        perturbed_image = torch.clamp(perturbed_image, 0, 1)

        return perturbed_image


    # Restores tensors to original levels
    def denorm(self, batch, mean=[0.1307], std=[0.3081]):
        if isinstance(mean, list):
            mean = torch.tensor(mean).to(self.device)
        if isinstance(std, list):
            std = torch.tensor(std).to(self.device)

        return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)
    