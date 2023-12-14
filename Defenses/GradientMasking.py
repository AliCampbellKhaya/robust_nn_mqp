import torch

from Defenses.BaseDefense import BaseDefense
from scipy.ndimage import gaussian_filter

class GradientMasking(BaseDefense):
    def __init__(self, base_model):
        super(GradientMasking, self).__init__()
        self.base_model = base_model