#Targeted deepfool attack
#contents
#Targeted deepfool definition Line 10
#Batch unloader Line 87
#Run targeted deepfool Line 97
#print perturbations Line 124
#print results Line 135
#data processing Line 177

#Targeted deepfool definition start
import numpy as np
from torch.autograd import Variable
import copy
#indexOfAttackedLabel refers to the index on the final output layer that we want to attack to.
def deepfoolTargeted(image, net, indexOfAttackedLabel, numClasses=10, step=0.02, maxIterations=50):
    #forward call omiting softmax.  Note to change depending on how many layers there are.
    fwd = net.conv_layer1(Variable(image[None, :, :, :], requires_grad=True))
    fwd = net.conv_layer2(fwd)
    fwd = fwd.view(fwd.size(0), -1)
    fwd = net.fc_layer(fwd)
    fImg = fwd.data.cpu().numpy().flatten()
    #replacing this line
    #fImg = net.forward(Variable(image[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()
    indexesOfClass = (np.array(fImg)).flatten().argsort()[::-1]
    indexesOfClass = indexesOfClass[0:numClasses]
    label = indexesOfClass[0]
    theShape = image.cpu().numpy().shape
    pertImage = copy.deepcopy(image)
    direction = np.zeros(theShape)
    totalPert = np.zeros(theShape)
    iLoops = 0
    x = Variable(pertImage[None, :], requires_grad=True)

    #do forward call.  Omit softmax
    fwda = net.conv_layer1(x)
    fwda = net.conv_layer2(fwda)
    fwda = fwda.view(fwda.size(0), -1)
    fwda = net.fc_layer(fwda)
    logits = fwda
    #logits = net.forward(x)
    #print(indexesOfClass)
    preds = [logits[0,indexesOfClass[k]] for k in range(numClasses)]
    attackedLabel = label
    if indexOfAttackedLabel == label:
      iLoops = maxIterations + 1
    while attackedLabel != indexOfAttackedLabel and iLoops < maxIterations:
        logits[0, indexesOfClass[0]].backward(retain_graph=True)
        origGrad = x.grad.data.cpu().numpy().copy()
        
        #start of for loop
        x.grad = None
        logits[0, indexOfAttackedLabel].backward(retain_graph=True)
        aGrad = x.grad.data.cpu().numpy().copy()
        #Set a new pertDirection and distance
        pertDirection = aGrad - origGrad
        distance = (logits[0, indexOfAttackedLabel] - logits[0, indexesOfClass[0]]).data.cpu().numpy()
        pertToClass = abs(distance)/np.linalg.norm(pertDirection.flatten())
        #Determine which pertDirection to use
        minPert = pertToClass
        direction = pertDirection
        #end of for loop

        #Compute pertOfILoop and totalPert
        #Add 1e-4 for numerical stability
        pertOfILoop =  (minPert+1e-4) * direction / np.linalg.norm(direction)
        totalPert = np.float32(totalPert + pertOfILoop)
        pertImage = image + (1+step)*torch.from_numpy(totalPert)
        x = Variable(pertImage, requires_grad=True)

        #Do forward call.  Omit softmax
        fwda = net.conv_layer1(x)
        fwda = net.conv_layer2(fwda)
        fwda = fwda.view(fwda.size(0), -1)
        fwda = net.fc_layer(fwda)
        logits = fwda
        #Replacing this line
        #logits = net.forward(x)

        attackedLabel = np.argmax(logits.data.cpu().numpy().flatten())
        iLoops += 1
    totalPert = (1+step)*totalPert
    if iLoops == maxIterations + 1:
      iLoops = 0
    return totalPert, iLoops, label, attackedLabel, pertImage
#targeted deepfool definition end

#Batch unloader start
def batchUnloader(images, net, indexOfAttackedLabel, numClasses=10, step=0.02, maxIter=50):
  perturbations = []
  for image in images:
    totalPert, iLoops, label, k_i, pertImage = deepfoolTargeted(image, net, indexOfAttackedLabel, numClasses, step, maxIter)
    perturbations.append((totalPert, iLoops, label, k_i, pertImage))
    #print("a")
  return perturbations
#Batch unloader end

#Run targeted deepfool
#begins with variable definitions for label being attacked to and maximum iterations
AttackToLabel = 1 #Attack to this label index
maximumIterations = 100 #50 for 0, more for other numbers.
model.eval()
batchesToPert = 1
counter = 0
originalImageArray = []
pertImageArray = []
deepfoolReturn = []
iloops = []
for (X, y) in val_dataloader:
  #for item in X:
    if counter >= batchesToPert: #if no more calls are required
      break
    originalImageArray.append(X)
    (X, y) = (X.to(device), y.to(device))
    deepFooled = batchUnloader(X, model, AttackToLabel, step=0.02, maxIter=maximumIterations) #Calls deepfool.
    deepfoolReturn.append(deepFooled)
    for i in range(X.size(0)):
      iAnImage = deepFooled[i][4]
      iAnNP = iAnImage.squeeze().detach().cpu().numpy()
      pertImageArray.append(iAnNP)
      iloops.append(deepFooled[i][1])
    counter = counter + 1
#Run targeted deepfool end

#print perturbations start
#Print the pertubation given to each image.  Loops through a single batch.
batchNum = 0 #batch number to show
for i, (noise, _, _, _, _) in enumerate(deepfoolReturn[batchNum]):
    noise = noise.squeeze()
    plt.imshow(noise, cmap='gray')
    plt.title(f"Perturbation of image {i+1}")
    plt.axis("off")
    plt.show()
#print perturbtaions end

#print results start
#For displaying original image, perturbed image, iloop iterations for that image, and labels for both images
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
origLabels = torch.Tensor([])
pertLabels = []
for item in originalImageArray:
  X = item
  X = X.to(device)
  with torch.no_grad():
    predictions = model(X)
  predictedLabels = predictions.argmax(dim=1)
  origLabels = torch.cat((predictedLabels,origLabels))
for itemtwo in deepfoolReturn:
  for item in itemtwo:
    _,_,_,item,_ = item

    pertLabels.append(item)

#Display original image, pert image and labels for each.
for i in range(0, 64):
    print(f"iloop iterations: {iloops[i]}")  #Print the iloop iterations for each image

    origImg = originalImageArray[int(i / 64)][i % 64].cpu().numpy()
    origImg = np.squeeze(origImg)
    origLabel = int(origLabels[i])

    pertImg = pertImageArray[i]
    pertImg = np.squeeze(pertImg)
    pertLabel = pertLabels[i]

    plt.subplot(1, 2, 1)
    plt.imshow(origImg, cmap='gray')
    plt.title(f"Original Image\nLabel: {origLabel}")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(pertImg, cmap='gray')
    plt.title(f"Perturbed Image\nLabel: {pertLabel}")
    plt.axis("off")
    plt.show()
#print results end

#data processing start
#for Data processing the iloop iterations.
totalarray = [] #Number of iloop iterations that it took to perturb all the numbers of the given index.  ie. at index zero the number is 8 than all the pertubations of zeros took eight iloops
totalarray.append(0)
totalarray.append(0)
totalarray.append(0)
totalarray.append(0)
totalarray.append(0)
totalarray.append(0)
totalarray.append(0)
totalarray.append(0)
totalarray.append(0)
totalarray.append(0)
countarray = [] #Number of images with original label of the index.
countarray.append(0)
countarray.append(0)
countarray.append(0)
countarray.append(0)
countarray.append(0)
countarray.append(0)
countarray.append(0)
countarray.append(0)
countarray.append(0)
countarray.append(0)
failedArray = [] #Number of images that were unable to be perturbed.
failedArray.append(0)
failedArray.append(0)
failedArray.append(0)
failedArray.append(0)
failedArray.append(0)
failedArray.append(0)
failedArray.append(0)
failedArray.append(0)
failedArray.append(0)
failedArray.append(0)
#process data
for i in range(len(pertLabels)): #Number of perturbed images generated
  if pertLabels[i] == origLabels[i] and iloops[i] != 0:
    #failed to perturb
    failedArray[int(origLabels[i])] = failedArray[int(origLabels[i])] + 1
    continue
  totalarray[int(origLabels[i])] = totalarray[int(origLabels[i])] + iloops[i]
  countarray[int(origLabels[i])] = countarray[int(origLabels[i])] + 1
  if iloops[i] > 25: #Used to locate especially bad images.  Change number to whatever is desired/needed
    print(f"perturbed image index {i} took {iloops[i]} iloops, and was originally labeled as a {int(origLabels[i])}")
    
print(totalarray) #total iloop iterations of index
print(countarray) #total original labels of index
print("\n")
for i in range(10): #Take and print averages
  
  print(f"Average iloop iterations for number {i} is {totalarray[i]/countarray[i]}")
print("\n")
for i in range(10): #Take and print averages
  print(f"Number of failed perturbations for number {i} is {failedArray[i]}")
#data processing end
