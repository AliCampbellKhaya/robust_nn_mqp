import torch

"""
TODO: Write/Update Base Defense
"""

class BaseDefense():
    def __init__(self, name, model, device):
        self.defense = name
        self.model = model
        self.device = device

    def forward(self, input, labels=None):
        input = input.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        results = self.batch_unloader(input, labels)
        return results
    
    def batch_unloader(self, input, labels):
        defended_images = []
        for image, label in zip(input, labels):
            defended_image = self.forward_individual(image, label)
            defended_images.append(defended_image)

        # return torch.stack(defended_images).flatten(start_dim=1, end_dim=2)
        return torch.stack(defended_images)

    def forward_individual(self, input, label=None):
        """Should be overwritten by every subclass"""
        raise NotImplementedError
    
    def forward_batch(self, input, label=None):
        """Should be overwritten by every subclass"""
        raise NotImplementedError