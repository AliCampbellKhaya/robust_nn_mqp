import torch

class BaseAttack():
    """
    Base Attack super class used for calling all attacks

    Arguments:
        name (str): name and type of the attack
        model (torch.nn.Module): neural network to be used in attack
        device (torch.device): either "cpu" or "cuda", the type of processor to be used in the attack
        targeted (bool): True if targeted to a value, False if not
        loss_function (torch.nn._Loss): loss function to be used to calculate success of attack
        optimizer (torch.nn.Optimizer): optimizer used to minimze the loss
    """
    
    def __init__(self, name, model, device, targeted, loss_function, optimizer):
        self.attack = name
        self.model = model
        self.device = device
        self.targeted = targeted
        self.loss_function = loss_function
        self.optimizer = optimizer

    def forward(self, input, labels=None):
        """
        Defines the forward pass for the attack
        """
        input = input.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        results = self.batch_unloader(input, labels)
        return results

    def normalize(self, input, mean=[0.1307], std=[0.3081]):
        """
        Defines the normalization for each batch of images
        """
        mean = torch.tensor(mean).to(self.device).reshape(1, self.model.num_channels, 1, 1)
        std = torch.tensor(std).to(self.device).reshape(1, self.model.num_channels, 1, 1)
        return (input - mean) / std
    
    def denormalize(self, input, mean=0.1307, std=0.3081):
        """
        Defines the denomalization for each batch of images
        """
        mean = torch.tensor(mean).to(self.device).reshape(1, self.model.num_channels, 1, 1)
        std = torch.tensor(std).to(self.device).reshape(1, self.model.num_channels, 1, 1)
        return (input * std) + mean
    
    def batch_unloader(self, input, labels):
        """
        Unloads images from the batch for use in an attack as some attacks ran
        quicker and more succesfully when unloaded
        """

        # Dictionary for tracking individual results from the attack
        results = {
            "pert_image": [],
            "final_label": [],
            "attack_label": [],
            "iterations": [],
            "perturbations": [],
            "original_image": []
        }

        # Run the forward pass for each image
        pert_image_batch = []
        for image, label in zip(input, labels):
            original_image = image.detach().clone()
            pert_image, final_label, attack_label, iterations, pert = self.forward_individual(image, label)
            results["pert_image"].append(pert_image)
            results["final_label"].append(final_label)
            results["attack_label"].append(attack_label)
            results["iterations"].append(iterations)
            results["perturbations"].append(pert)
            results["original_image"].append(original_image)
            pert_image_batch.append(pert_image)

        results2 = [torch.stack(pert_image_batch).flatten(start_dim=1, end_dim=2), results]
        
        return results2

    def forward_individual(self, input, label):
        """
        Defines the computation used in the forward pass
        Should be overwritten by every subclass
        """
        raise NotImplementedError

    def __call__(self, inputs, labels=None):

        inputs = self.denormalize(inputs)
        perturbed_inputs = self.forward(inputs, labels)
        perturbed_inputs = self.normalize(perturbed_inputs)

        return perturbed_inputs