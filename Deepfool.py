#contents
#deepfool definition and algorithm  Line 9
#Batch unloader  Line 80
#call deepfool  Line 89
#Print Perturbations  Line 114
#Display deepfool output  Line 124
#Data processing  Line 170

#deepfool definition and algorithm start
import numpy as np
from torch.autograd import Variable
import copy
def deepfool(image, net, numClasses=10, step=0.02, maxIterations=50):
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

    preds = [logits[0,indexesOfClass[k]] for k in range(numClasses)]
    attackedLabel = label
    while attackedLabel == label and iLoops < maxIterations:
        minPert = np.inf
        logits[0, indexesOfClass[0]].backward(retain_graph=True)
        origGrad = x.grad.data.cpu().numpy().copy()
        for k in range(1, numClasses):
            x.grad = None
            logits[0, indexesOfClass[k]].backward(retain_graph=True)
            aGrad = x.grad.data.cpu().numpy().copy()
            #Set a new pertDirection and distance
            pertDirection = aGrad - origGrad
            distance = (logits[0, indexesOfClass[k]] - logits[0, indexesOfClass[0]]).data.cpu().numpy()
            pertToClass = abs(distance)/np.linalg.norm(pertDirection.flatten())
            #Determine which pertDirection to use
            if pertToClass < minPert:
                minPert = pertToClass
                direction = pertDirection
        # compute pertOfILoop and totalPert
        # Added 1e-4 for numerical stability
        pertOfILoop =  (minPert+1e-4) * direction / np.linalg.norm(direction)
        totalPert = np.float32(totalPert + pertOfILoop)
        pertImage = image + (1+step)*torch.from_numpy(totalPert)
        x = Variable(pertImage, requires_grad=True)
        
        #do forward call.  Omit softmax
        fwda = net.conv_layer1(x)
        fwda = net.conv_layer2(fwda)
        fwda = fwda.view(fwda.size(0), -1)
        fwda = net.fc_layer(fwda)
        logits = fwda
        #replacing this line
        #logits = net.forward(x)
        
        attackedLabel = np.argmax(logits.data.cpu().numpy().flatten())
        iLoops += 1
    totalPert = (1+step)*totalPert
    return totalPert, iLoops, label, attackedLabel, pertImage
#end deepfool definition and algorithm

#batch unloader start
def batchUnloader(images, net, numClasses=10, step=0.02, maxIter=50):
  perturbations = []
  for image in images:
    totalPert, iLoops, label, k_i, pertImage = deepfool(image, net, numClasses, step, maxIter)
    perturbations.append((totalPert, iLoops, label, k_i, pertImage))
  return perturbations
#Batch unloader end

#call deepfool start
#includes some data processing for perturbed images
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
    deepFooled = batchUnloader(X, model) #Calls deepfool.
    deepfoolReturn.append(deepFooled)
    for i in range(X.size(0)):
      iAnImage = deepFooled[i][4]
      iAnNP = iAnImage.squeeze().detach().cpu().numpy()
      pertImageArray.append(iAnNP)
      iloops.append(deepFooled[i][1])
    counter = counter + 1
#Call deepfool end

#Print Perturbations start
batchNum = 0 #batch number to show
for i, (noise, _, _, _, _) in enumerate(deepfoolReturn[batchNum]):
    noise = noise.squeeze()
    plt.imshow(noise, cmap='gray')
    plt.title(f"Perturbation of image {i+1}")
    plt.axis("off")
    plt.show()
#print perturbations end

#Display deepfool output start
#displays images, perturbations, labels, and iloops for each image
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
    _,_,_,_,item = item
    X = item
    X = X.to(device)
    with torch.no_grad():
      ppredictions = model(X)
    ppredictedLabels = ppredictions.argmax(dim=1)
    pertLabels.append(ppredictedLabels)

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
#Display deepfool output end

#Data processing start
#data processing for averages on iloops
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
#process data
for i in range(len(pertLabels)): #Number of perturbed images generated
  totalarray[int(origLabels[i])] = totalarray[int(origLabels[i])] + iloops[i]
  countarray[int(origLabels[i])] = countarray[int(origLabels[i])] + 1
  if iloops[i] > 5: #Used to locate especially bad images.  Change number to whatever is desired/needed
    print(iloops[i])
    print(origLabels[i])
    print(i)
print(totalarray) #total iloop iterations of index
print(countarray) #total original labels of index
print("\n")
for i in range(10): #Take and print averages
  print(f"Average iloop iterations for number {i} is {totalarray[i]/countarray[i]}")
#Data processing end
