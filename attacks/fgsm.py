import torch

class FGSM():
    def __init__(self, model, epsilons, device):
        self.model = model
        self.epsilons = epsilons
        self.device = device

    def forward(self, images, labels):
        pass

    # Restores tensors to original levels
    def denorm(self, batch, mean=[0.1307], std=[0.3081]):
        if isinstance(mean, list):
            mean = torch.tensor(mean).to(device)
        if isinstance(std, list):
            std = torch.tensor(std).to(device)
    