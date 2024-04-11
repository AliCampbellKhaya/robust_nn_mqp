#Feature squeezing by applying a gausian blur to the attacked image as a method by which to smooth the features of the image and reduce concentrated perturbations.
#Takes in originalImageArray, originalLabelArray, attackedImages
#originalImageArray is a list of tensors size [batch, channels, width, height]
#originalLabelArray is a list of "Tensor with shape torch.Size([])" with length equal to the number of attacked images
#attackedImages is a list with length equal to the number of attacked batches.  This list contains lists with length equal to batch size containing tensors with size [channels, width, height]

"""
TODO: Update defense and see what is going on here
"""

from Defenses.BaseDefense import BaseDefense
from scipy.ndimage import gaussian_filter
import torch

class FeatureSqueezing(BaseDefense):
    def __init__(self, model, device):
        super(FeatureSqueezing, self).__init__("FS", model, device)

    def forward_individual(self, input, label):
      input = input[None, :]
      input = gaussian_filter(input.detach(), sigma=0.5)
      #inputs = gaussian_filter(inputs.detach().numpy(), sigma=0.5)
      return torch.tensor(input)
