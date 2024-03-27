import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time

from NeuralNetworks.MNISTNeuralNetwork import MNISTNeuralNetwork
from NeuralNetworks.CifarNeuralNetwork import CifarNeuralNetwork
from NeuralNetworks.TrafficNeuralNetwork import TrafficNeuralNetwork

from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import random_split
from torch.utils.data import DataLoader

batch_size = 64

CNN_MEAN = [0.485, 0.456, 0.406]
CNN_STD = [0.229, 0.224, 0.225]
transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=CNN_MEAN, std=CNN_STD),
    # resize(32, 32)
    v2.Resize((128, 128), antialias=True),
])

train_data = datasets.GTSRB(root="data", split="train", download=True, transform=transforms)
test_data = datasets.GTSRB(root="data", split="test", download=True, transform=transforms)

train_dataloader = DataLoader(train_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

images = {}
added_images = []

count = 0
for (inputs, labels) in test_dataloader:
    for input, label in zip(inputs, labels):
        label = label.cpu().numpy().item()

        if label not in added_images:

            #images[label] = input
            images[count] = input
            added_images.append(label)
            count += 1
    
    if len(added_images) > 10:
        break

# for i in range(10):
#     fig, axs = plt.subplots(10, 1)
#     axs[i].plot(images[i][0, :, :])
#     axs[i].set_title(i)
#     plt.axis("off")

# plt.show()

fig, axs = plt.subplots(10, 1, figsize=(8, 20))

# Loop through each subplot
for i, ax in enumerate(axs):
    # Plot data on each subplot
    if i == 4:
        ax.imshow(images[13].permute(1, 2, 0).numpy())
    else: 
        ax.imshow(images[i].permute(1, 2, 0).numpy())

    ax.axis('off')

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()
    

""" MNIST Data """

# from NeuralNetworks.MNISTNeuralNetwork import MNISTNeuralNetwork

# device = torch.device("cpu")
# model = MNISTNeuralNetwork(device, train_split=0.8, batch_size=64).to(device)

# model.load_model()

# images = {}
# added_images = []

# for inputs, labels in model.test_dataloader:

#     for input, label in zip(inputs, labels):
#         label = label.cpu().numpy().item()

#         if label not in added_images:

#             images[label] = input
#             added_images.append(label)
    
#     if len(added_images) > 10:
#         break

# for i in range(10):
#     plt.subplot(1, 10, i+1)
#     plt.imshow(images[i][0, :, :], cmap='gray')
#     plt.title(i)
#     plt.axis("off")

# plt.tight_layout()
# plt.show()

# """ CIFAR Data """

# device = torch.device("cpu")
# model = CifarNeuralNetwork(device, train_split=0.8, batch_size=64).to(device)

# model.load_model()

# images = {}
# added_images = []
# classes_labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# for inputs, labels in model.test_dataloader:

#     for input, label in zip(inputs, labels):
#         label = label.cpu().numpy().item()

#         if label not in added_images:

#             images[label] = input
#             added_images.append(label)
    
#     if len(added_images) > 10:
#         break

# for i in range(10):
#     plt.subplot(1, 10, i+1)
#     plt.imshow(images[i].permute(1, 2, 0))
#     plt.title(classes_labels[i])
#     plt.axis("off")

# #plt.tight_layout()
# plt.show()


""" Traffic Data """

# device = torch.device("cpu")
# model = TrafficNeuralNetwork(device, train_split=0.8, batch_size=64).to(device)

# model.load_model()

# images = {}
# added_images = []
# #classes_labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# for inputs, labels in model.test_dataloader:

#     for input, label in zip(inputs, labels):
#         label = label.cpu().numpy().item()

#         if label not in added_images:

#             images[label] = input
#             added_images.append(label)
    
#     if len(added_images) > 10:
#         break

# print(len(added_images))
# print(added_images)

# for i in range(10):
#     plt.subplot(1, 10, i+1)
#     plt.imshow(images[i].permute(1, 2, 0))
#     #plt.title(classes_labels[i])
#     plt.axis("off")

# #plt.tight_layout()
# plt.show()

# device = torch.device("cpu")

# mnist_model = MNISTNeuralNetwork(device, train_split=0.8, batch_size=64).to(device)
# mnist_model.load_model()

# cifar_model = CifarNeuralNetwork(device, train_split=0.8, batch_size=64).to(device)
# cifar_model.load_model()

# traffic_model = TrafficNeuralNetwork(device, train_split=0.8, batch_size=64).to(device)
# traffic_model.load_model()

# # Between 1e-3 and 1e-5
# learning_rate = 1e-4
# epochs = 5
# loss_function = nn.NLLLoss()
# #optimizer = torch.optim.Adam(model.parameters(), learning_rate)

# print("Initial results for trained MNIST neural network")
# start = time.time()
# cr, preds, examples = mnist_model.test_model(loss_function)
# print(cr)
# end = time.time()
# print(f"Time to test MNIST neural network: {end-start}")
# mnist_model.display_images(examples)

# print("Initial results for trained Cifar neural network")
# start = time.time()
# cr, preds, examples = cifar_model.test_model(loss_function)
# print(cr)
# end = time.time()
# print(f"Time to test MNIST neural network: {end-start}")
# cifar_model.display_images(examples)

# print("Initial results for trained Traffic neural network")
# start = time.time()
# cr, preds, examples = traffic_model.test_model(loss_function)
# print(cr)
# end = time.time()
# print(f"Time to test MNIST neural network: {end-start}")
# traffic_model.display_images(examples)