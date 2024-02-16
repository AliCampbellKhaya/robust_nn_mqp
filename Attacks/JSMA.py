import torch
import torch.nn.functional as F
from torch.autograd import Variable

from Attacks.BaseAttack import BaseAttack

"""
TODO: Write comment explaining attack
TODO: Write attack
"""

class JSMA(BaseAttack):
    def __init__(self, model, device, targeted, max_perturb, perturb_length, max_steps, loss_function, optimizer):
        super(JSMA, self).__init__("JSMA", model, device, targeted, loss_function, optimizer)
        self.max_perturb = max_perturb
        self.perturb_length = perturb_length
        self.max_steps = max_steps

    def forward(self, input, label):
        input = input.clone().detach().to(self.device).requires_grad_(True)
        label = label.clone().detach().to(self.device)
        perturbed_input = input.clone().detach().requires_grad_(True)
        target_label = self.find_target_label(input, self.model(input).detach(), label)

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
    
    def calculate_label_distance(self, embedding1, embedding2):
         return torch.norm(embedding1 - embedding2)
    
    def find_target_label(self, input, input_label_idx, labels):
         with torch.no_grad():
              embeddings = self.omit_softmax(input).numpy()

         current_embedding = embeddings[input_label_idx]
         distances = [self.calculate_label_distance(current_embedding, embeddings[i]) for i in range(len(labels))]

         nearest_label_idx = torch.argmin(torch.tensor(distances))

         return labels[nearest_label_idx]
    
    # Variable(input[None, :, :, :]
    def omit_softmax(self, input):
        fwd = self.model.conv_layer1(input)
        fwd = self.model.conv_layer2(fwd)
        fwd = fwd.view(fwd.size(0), -1)
        fwd = self.model.fc_layer(fwd)

        return fwd

