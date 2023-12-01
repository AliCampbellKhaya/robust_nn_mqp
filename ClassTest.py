import torch
import torch.nn as nn

from NeuralNetworks.TrafficNeuralNetwork import TrafficNeuralNetwork
from Attacks.FGSM import FGSM

import numpy as np
from torch.autograd import Variable
import copy
def deepfool(image, net, numClasses=10, step=0.02, maxIterations=50):
    #forward call omiting softmax.  Note to change depending on how many layers there are.
    fwd = net.conv_layer1(Variable(image[None, :, :, :], requires_grad=True))
    #fwd = net.conv_layer2(fwd)
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
    #fwda = net.conv_layer2(fwda)
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
        #fwda = net.conv_layer2(fwda)
        fwda = fwda.view(fwda.size(0), -1)
        fwda = net.fc_layer(fwda)
        logits = fwda
        #replacing this line
        #logits = net.forward(x)
        
        attackedLabel = np.argmax(logits.data.cpu().numpy().flatten())
        iLoops += 1
    totalPert = (1+step)*totalPert
    return totalPert, iLoops, label, attackedLabel, pertImage

def main():
    # For now remove SSL certification because is not working
    # Remove when SSL cert is working
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    device = torch.device("cpu")
    model = TrafficNeuralNetwork(device, train_split=0.8, batch_size=64).to(device)

    # Between 1e-3 and 1e-5
    learning_rate = 1e-4
    epochs = 3
    loss_function = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    for e in range(epochs):
        print(f"Epoch {e+1}")
        print(model.train_model(loss_function, optimizer))
        print("-"*50)

    # cr, preds = model.test(loss_function)
    # print(cr)

    count = 0
    for (X, y) in model.test_dataloader:
        a, b, c, d, e = deepfool(X, model)
        print(f"{c} -----> {d}")
        if count >= 3:
            break
        count += 1

if __name__ == "__main__":
    main()