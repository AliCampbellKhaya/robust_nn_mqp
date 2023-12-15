#Feature squeezing by applying a gausian blur to the attacked image as a method by which to smooth the features of the image and reduce concentrated perturbations.
#Takes in originalImageArray, originalLabelArray, attackedImages
#originalImageArray is a list of tensors size [batch, channels, width, height]
#originalLabelArray is a list of "Tensor with shape torch.Size([])" with length equal to the number of attacked images
#attackedImages is a list with length equal to the number of attacked batches.  This list contains lists with length equal to batch size containing tensors with size [channels, width, height]

from Defenses.BaseDefense import BaseDefense
from scipy.ndimage import gaussian_filter
import torch

class FeatureSqueezing(BaseDefense):
    def __init__(self, model):
        super(FeatureSqueezing, self).__init__("FS", model)

    def forward(self, inputs, labels):
      inputs = gaussian_filter(inputs.detach().numpy(), sigma=0.5)
      return torch.tensor(inputs)


    

# printMe = []
# origImages = []
# atackImages = []
# smoothedImages = []
# for batch in originalImageArray:
#   for image in batch:
#     origImages.append(image)
# for batch in attackedImages:
#   #smoothBatch = []
#   for image in batch:
#     atackImages.append(image)
#     smoothedImages.append(gaussian_filter(image, sigma=0.5))
# for i in range(len(origImages)):
#   printMe.append((origImages[i],atackImages[i],smoothedImages[i]))


# labelMe = []
# originLabels = []
# atackLabels = []
# smoothLabels = []
# for item in printMe:
#   a = model(item[0][None, :]).argmax(dim=1)[0].item()
#   b = model(item[1][None, :]).argmax(dim=1)[0].item()
#   c = model(torch.tensor(item[2][None, :])).argmax(dim=1)[0].item()
#   labelMe.append((a, b, c))
# #print(labelMe)
# for i in range(len(labelMe)):
#   if i == 0:
#     break
#   fig = plt.figure(figsize=(10, 7))
#   fig.add_subplot(1, 3, 1)
#   plt.imshow(printMe[i][0].squeeze(), cmap='gray')
#   plt.title(f"Correct label: {originalLabelArray[i]}\noriginally Labeled as: {labelMe[i][0]}")
#   plt.axis("off")
#   fig.add_subplot(1, 3, 2)
#   plt.imshow(printMe[i][1].squeeze(), cmap='gray')
#   plt.title(f"Attacked Image\nLabel: {labelMe[i][1]}")
#   plt.axis("off")
#   fig.add_subplot(1, 3, 3)
#   plt.imshow(printMe[i][2].squeeze(), cmap='gray')
#   plt.title(f"Smoothed Image\nLabel: {labelMe[i][2]}")
#   plt.axis("off")
#   plt.show()

# total = 0
# correctAfterAttack = 0
# correctAfterDefense = 0
# defenseHelped = 0
# defenseHurt = 0
# for threeple in labelMe:

#   if threeple[0] == threeple[1]:
#     correctAfterAttack = correctAfterAttack + 1
#     if threeple[1] != threeple[2]:
#       defenseHurt += 1
#   if threeple[0] == threeple[2]:
#     correctAfterDefense += 1
#     if threeple[2] != threeple[1]:
#       defenseHelped += 1
#   total = total + 1

# print(f"The total number of images is {len(labelMe)}")
# print(f"The number of images where the attack did not change class is {correctAfterAttack}")
# print(f"The number of images where the defense had the same class as the original is {correctAfterDefense}")
# print(f"The number of images where the attack changed the class but the defense changed it back is {defenseHelped}")
# print(f"The number of images where the attack did not change the class but the defense did is {defenseHurt}")
# print(f"The accuracy of the model after feature smoothing is {correctAfterDefense/len(labelMe)}")
