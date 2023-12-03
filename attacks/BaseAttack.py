import torch

class BaseAttack():
    def __init__(self, name, model, device, targeted, loss_function, optimizer):
        self.attack = name
        self.model = model
        self.device = device
        self.targeted = targeted
        self.loss_function = loss_function
        self.optimizer = optimizer

    def forward(self, input, labels=None):
        """Should be overwritten by every subclass"""
        raise NotImplementedError

    def normalize(self, input, mean=[0.1307], std=[0.3081]):
        mean = torch.tensor(mean).to(self.device).reshape(1, self.model.num_channels, 1, 1)
        std = torch.tensor(std).to(self.device).reshape(1, self.model.num_channels, 1, 1)
        return (input - mean) / std
    
    def denormalize(self, input, mean=0.1307, std=0.3081):
        mean = torch.tensor(mean).to(self.device).reshape(1, self.model.num_channels, 1, 1)
        std = torch.tensor(std).to(self.device).reshape(1, self.model.num_channels, 1, 1)
        return (input * std) + mean
    
    # All methods from here are placeholders -- Copied to provide inspiration
    
    def _set_mode_targeted(self, mode, quiet):
        if "targeted" not in self.supported_mode:
            raise ValueError("Targeted mode is not supported.")
        self.targeted = True
        self.attack_mode = mode
        if not quiet:
            print("Attack mode is changed to '%s'." % mode)

    def set_mode_targeted_by_function(self, target_map_function, quiet=False):
        """
        Set attack mode as targeted.

        Arguments:
            target_map_function (function): Label mapping function.
                e.g. lambda inputs, labels:(labels+1)%10.
                None for using input labels as targeted labels. (Default)
            quiet (bool): Display information message or not. (Default: False)

        """
        self._set_mode_targeted("targeted(custom)", quiet)
        self._target_map_function = target_map_function

    def set_mode_targeted_random(self, quiet=False):
        """
        Set attack mode as targeted with random labels.

        Arguments:
            quiet (bool): Display information message or not. (Default: False)

        """
        self._set_mode_targeted("targeted(random)", quiet)
        self._target_map_function = self.get_random_target_label

    def set_mode_targeted_least_likely(self, kth_min=1, quiet=False):
        """
        Set attack mode as targeted with least likely labels.

        Arguments:
            kth_min (str): label with the k-th smallest probability used as target labels. (Default: 1)
            num_classses (str): number of classes. (Default: False)

        """
        self._set_mode_targeted("targeted(least-likely)", quiet)
        assert kth_min > 0
        self._kth_min = kth_min
        self._target_map_function = self.get_least_likely_label

    def set_mode_targeted_by_label(self, quiet=False):
        """
        Set attack mode as targeted.

        Arguments:
            quiet (bool): Display information message or not. (Default: False)

        .. note::
            Use user-supplied labels as target labels.
        """
        self._set_mode_targeted("targeted(label)", quiet)
        self._target_map_function = "function is a string"
    
    @torch.no_grad()
    def get_output_with_eval_nograd(self, inputs):
        given_training = self.model.training
        if given_training:
            self.model.eval()
        outputs = self.get_logits(inputs)
        if given_training:
            self.model.train()
        return outputs

    def get_target_label(self, inputs, labels=None):
        """
        Function for changing the attack mode.
        Return input labels.
        """
        if self._target_map_function is None:
            raise ValueError(
                "target_map_function is not initialized by set_mode_targeted."
            )
        if self.attack_mode == "targeted(label)":
            target_labels = labels
        else:
            target_labels = self._target_map_function(inputs, labels)
        return target_labels
    
    @torch.no_grad()
    def get_least_likely_label(self, inputs, labels=None):
        outputs = self.get_output_with_eval_nograd(inputs)
        if labels is None:
            _, labels = torch.max(outputs, dim=1)
        n_classses = outputs.shape[-1]

        target_labels = torch.zeros_like(labels)
        for counter in range(labels.shape[0]):
            l = list(range(n_classses))
            l.remove(labels[counter])
            _, t = torch.kthvalue(outputs[counter][l], self._kth_min)
            target_labels[counter] = l[t]

        return target_labels.long().to(self.device)

    @torch.no_grad()
    def get_random_label(self, inputs, labels=None):
        outputs = self.get_output_with_eval_nograd(inputs)
        if labels is None:
            _, labels = torch.max(outputs, dim=1)
        n_classses = outputs.shape[-1]

        target_labels = torch.zeros_like(labels)
        for counter in range(labels.shape[0]):
            l = list(range(n_classses))
            l.remove(labels[counter])
            t = (len(l) * torch.rand([1])).long().to(self.device)
            target_labels[counter] = l[t]

        return target_labels.long().to(self.device) 

    def __call__(self, inputs, labels=None):

        inputs = self.denormalize(inputs)
        perturbed_inputs = self.forward(inputs, labels)
        perturbed_inputs = self.normalize(perturbed_inputs)

        return perturbed_inputs