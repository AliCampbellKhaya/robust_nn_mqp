import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time

from NeuralNetworks.MNISTNeuralNetwork import MNISTNeuralNetwork
from NeuralNetworks.CifarNeuralNetwork import CifarNeuralNetwork
from NeuralNetworks.TrafficNeuralNetwork import TrafficNeuralNetwork


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

device = torch.device("cpu")

mnist_model = MNISTNeuralNetwork(device, train_split=0.8, batch_size=64).to(device)
mnist_model.load_model()

cifar_model = CifarNeuralNetwork(device, train_split=0.8, batch_size=64).to(device)
cifar_model.load_model()

traffic_model = TrafficNeuralNetwork(device, train_split=0.8, batch_size=64).to(device)
traffic_model.load_model()

# Between 1e-3 and 1e-5
learning_rate = 1e-4
epochs = 5
loss_function = nn.NLLLoss()
#optimizer = torch.optim.Adam(model.parameters(), learning_rate)

print("Initial results for trained MNIST neural network")
start = time.time()
cr, preds, examples = mnist_model.test_model(loss_function)
print(cr)
end = time.time()
print(f"Time to test MNIST neural network: {end-start}")
mnist_model.display_images(examples)

print("Initial results for trained Cifar neural network")
start = time.time()
cr, preds, examples = cifar_model.test_model(loss_function)
print(cr)
end = time.time()
print(f"Time to test MNIST neural network: {end-start}")
cifar_model.display_images(examples)

print("Initial results for trained Traffic neural network")
start = time.time()
cr, preds, examples = traffic_model.test_model(loss_function)
print(cr)
end = time.time()
print(f"Time to test MNIST neural network: {end-start}")
traffic_model.display_images(examples)