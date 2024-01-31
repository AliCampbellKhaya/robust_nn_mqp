import torch
import torch.nn.functional as F
from torch.autograd import Variable

from Attacks.BaseAttack import BaseAttack

"""
TODO: Write comment explaining attack
TODO: Write attack
"""

class JSMA(BaseAttack):
    def __init__(self, model, device, targeted, max_perturb, perturb_length, max_steps):
        super(JSMA, self).__init__("JSMA", model, device, targeted)
        self.max_perturb = max_perturb
        self.perturb_length = perturb_length
        self.max_steps = max_steps

    def forward(self, input, label):
        input = input.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)
        target_label = find_target_label(input, self.model(input), label)

        for _ in self.max_steps:
            jsm = self.saliency_map(input)

            most_salient_feature = torch.argmax(jsm)

            input[0, most_salient_feature] += self.perturb_length

            perturbed_input = torch.clamp(perturbed_input, input * (1 - self.max_perturb),
                                          input * (1 + self.max_perturb))
            
            new_output = self.model(perturbed_input)
            _, pred_class = torch.max(new_output, 1)

            if pred_class.item() == target_label:
                    break
            
            return perturbed_input.squeeze()
            

    def saliency_map(self, input):
        self.model.zero_grad()

        output = self.model(input.unsqueeze(0))
        target = torch.argmax(output)

        output[target].backward()

        map = input.grad.abs()

        return map.squeeze()
    
    def calculate_label_distance(embedding1, embedding2):
         return torch.norm(embedding1 - embedding2)
    
    def find_target_label(input, input_label_idx, labels):
         with torch.no_grad():
              embeddings = omit_softmax(input).numpy()

         current_embedding = embeddings[input_label_idx]
         distances = [calculate_label_distance(current_embedding, embeddings[i]) for i in range(len(labels))]

         nearest_label_idx = torch.argmin(torch.tensor(distances))

         return labels[nearest_label_idx]
    
    def omit_softmax(self, input):
        fwd = self.model.conv_layer1(Variable(input[None, :, :, :], requires_grad=True))
        fwd = self.model.conv_layer2(fwd)
        fwd = fwd.view(fwd.size(0), -1)
        fwd = self.model.fc_layer(fwd)

        return fwd


# def jsma_adversarial_attack(input_tensor, model, target_class, max_perturbations=0.1, theta=0.1):
#     # Assume input_tensor is a PyTorch tensor representing the input features
#     # Assume model is a PyTorch model
#     # Assume target_class is the index of the target class
#     # max_perturbations is the maximum percentage of perturbations allowed
#     # theta is a parameter controlling the attack strength

#     # Copy the input tensor to track perturbations
#     perturbed_input = input_tensor.clone().detach().requires_grad_(True)

#     # Calculate the model output for the original input
#     original_output = model(input_tensor.unsqueeze(0))

#     # Iteratively perturb input features to maximize the target class probability
#     while True:
#         # Calculate the Jacobian Saliency Map
#         saliency_map = jacobian_saliency_map(perturbed_input, model)

#         # Find the index of the most salient feature
#         most_salient_feature = torch.argmax(saliency_map)

#         # Perturb the most salient feature
#         perturbed_input[0, most_salient_feature] += theta

#         # Clip perturbations to ensure they stay within a certain range
#         perturbed_input = torch.clamp(perturbed_input, input_tensor * (1 - max_perturbations),
#                                       input_tensor * (1 + max_perturbations))

#         # Check if the model's prediction has changed to the target class
#         new_output = model(perturbed_input)
#         _, predicted_class = torch.max(new_output, 1)
#         if predicted_class.item() == target_class:
#             # Successfully generated an adversarial example
#             break

#     return perturbed_input.squeeze()

# def jacobian_saliency_map(input_tensor, model):
#     # Calculate the Jacobian Saliency Map using PyTorch autograd

#     input_tensor = input_tensor.clone().detach().requires_grad_(True)
#     model.zero_grad()

#     output = model(input_tensor.unsqueeze(0))
#     target = torch.argmax(output)

#     output[target].backward()

#     saliency_map = input_tensor.grad.abs()

#     return saliency_map.squeeze()
