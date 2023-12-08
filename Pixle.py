#Contents
#definitions  Line 7
#batch unloader  Line 63
#call pixle  Line 72
#data processing and display images  Line 88

#definitions start
import numpy as np
from torch.autograd import Variable
import copy
def pixle(image, net, attackType):
  if attackType == 0:
    return pixleSwitchbasic(image, net)
  elif attackType == 1:
    return pixleRandomRows(image, 12, 17, net)
  else:
    return pixleRandom(image, net)

def pixleSwitchbasic(image, net):
  flatImage = image.view(-1)
  #print(flatImage)
  indexes = []
  counter = 0
  for item in flatImage:
    indexes.append([item, counter])
    counter = counter + 1
  #the image is now stored in a multidimensional array where indexes[number][0] is the value of the given pixel and indexes[number][1] is the index of the pixel in the original image

  #indexes now must be sorted
  indexes = sorted(indexes)
  #print(indexes)
  attackedImage = flatImage
  swapMe = 0
  counterTwo = 0

  for counterTwo in range(len(indexes)):
    if counterTwo % 2 == 0:
      counterTwo = counterTwo + 1
      continue
    #swap pixels of indexes n and n-1
    attackedImage[indexes[counterTwo][1]] = indexes[counterTwo-1][0]
    attackedImage[counterTwo-1] = indexes[counterTwo-1][0]
    counterTwo = counterTwo + 1

  attackedImage = attackedImage.view(image.size())
  return attackedImage

def pixleRandomRows(image, startRow, endRow, net):
    section = image[:, startRow:endRow, :].view(-1)
    indexes = torch.randperm(section.size(0))
    shuffled = section[indexes]
    image[:, startRow: endRow, :] = shuffled.view(image[:, startRow:endRow, :].size())
    return image

def pixleRandom(image, net):
  flatImage = image.view(-1)
  indexes = torch.randperm(flatImage.size(0))
  attackedImage = flatImage[indexes]
  attackedImage = attackedImage.view(image.size())
  return attackedImage
#definitions end

#batch unloader start
def batchUnloader(images, net, attackType):
  perturbations = []
  for image in images:
    pertImage = pixle(image, net, attackType)
    perturbations.append((pertImage))
  return perturbations
#batch unloader end

#call pixle start
model.eval()
attackType = 0  #corresponds to hard coded ways to use the pixle attack
batchesToPert = 1
counter = 0
originalImageArray = []
pixleReturn = []
for (X, y) in val_dataloader:
  if counter >= batchesToPert: #if no more calls are required
    break
  originalImageArray.append(X)
  pixled = batchUnloader(X, model, attackType) #Calls Pixle.
  pixleReturn.append(pixled)
  counter = counter + 1
#call pixle end

#data processing and display images start
#print the pixled images
#pixled is a batch of images that have been attacked.
total = 0
correct = 0
counterOne = 0
for batch in pixleReturn:
  for item in batch:
    plt.imshow(item.squeeze().detach().cpu().numpy(), cmap='gray')
    item = item.to(device)
    #prediction = ""
    with torch.no_grad():
      prediction = model(item.unsqueeze(0).to(device)).argmax()
    plt.title("attacked label is " + str(prediction.item()))
    plt.axis("off")
    plt.show()
    if model(originalImageArray[counterOne][total % 64].unsqueeze(0)).argmax() == prediction:
      correct = correct + 1
    total = total +1
  counterOne = counterOne + 1
print(f"the total attacked images is {total}")
print(f"the correctly predicted attacked images is {correct}")
print(f"the accuracy is {correct/total}")
#data processing and display images end
