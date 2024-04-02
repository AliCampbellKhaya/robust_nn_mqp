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
        pert_image_batch = []
        for image, label in zip(input, labels):
            # print("x before attack")
            # print(image)
            original_image = image.detach().clone()
            defended_image = self.forward_individual(image, label)
            defended_images.append(defended_image)
            pert_image_batch.append(defended_image)

        results2 = [torch.stack(pert_image_batch), defended_images]
        
        return results2

    def forward_individual(self, input, label=None):
        """Should be overwritten by every subclass"""
        raise NotImplementedError
    
    def forward_batch(self, input, label=None):
        """Should be overwritten by every subclass"""
        raise NotImplementedError