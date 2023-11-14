import torch

import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision import transforms
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import random_split
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
import numpy as np

class TrafficNeuralNetwork(nn.Module):
    def __init__(self):
        super(TrafficNeuralNetwork, self).__init__()
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.1)
        )

# For future use, not needed for MNIST dataset
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
            # nn.Dropout2d(p=0.1)
        )

        self.fc_layer = nn.Sequential(
            #445568
            #RuntimeError: mat1 and mat2 shapes cannot be multiplied (64x968256 and 186624x2048)
            nn.Linear(in_features=968256, out_features=2048),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=2048, out_features=512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=0.1),
            nn.Linear(in_features=256, out_features=43)
        )

    def forward(self, x):
        x = self.conv_layer1(x)

        # x = self.conv_layer2(x)

        # x = self.conv_layer3(x)

        # Flatten into 1D Vector for FC Layer
        x = x.view(x.size(0), -1)

        x = self.fc_layer(x)

        return F.log_softmax(x, dim=1)
    
def train(model, train_dataloader, val_dataloader, loss_function, optimizer, device, train_steps, val_steps, history):
    model.train()

    total_train_loss = 0
    total_val_loss = 0

    total_train_correct = 0
    total_val_correct = 0

    for (X, y) in train_dataloader:
        (X, y) = (X.to(device), y.to(device))

        optimizer.zero_grad()

        pred = model(X)
        loss = loss_function(pred, y)

        loss.backward()
        optimizer.step()

        total_train_loss += loss
        total_train_correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    with torch.no_grad():
        model.eval()

        for (X, y) in val_dataloader:
            (X, y) = (X.to(device), y.to(device))

            pred = model(X)
            loss = loss_function(pred, y)

            total_val_loss += loss
            total_val_correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    avg_train_loss = total_train_loss / train_steps
    avg_val_loss = total_val_loss / val_steps

    train_correct = total_train_correct / len(train_dataloader.dataset)
    val_correct = total_val_correct / len(val_dataloader.dataset)

    history["train_loss"].append(avg_train_loss.cpu().detach().numpy())
    history["train_loss_min"] = min( (history["train_loss"][-1], history["train_loss_min"]) )
    history["train_acc"].append(train_correct)
    history["val_loss"].append(avg_val_loss.cpu().detach().numpy())
    history["val_acc"].append(val_correct)

    return history

def test(model, test_dataloader, loss_function, device, test_data):
    model.eval()

    total_test_loss = 0
    total_test_correct = 0
    preds = []
    examples = []

    for (X, y) in test_dataloader:
        (X, y) = (X.to(device), y.to(device))

        pred = model(X)
        loss = loss_function(pred, y)

        total_test_loss += loss
        total_test_correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        preds.extend(pred.argmax(1).cpu().numpy())

        if len(examples) < 5:
            examples.append( (pred, X.squeeze().detach().cpu().numpy()) )


    cr = classification_report(test_data.targets, np.array(preds), target_names=test_data.classes)

    return cr, examples

DEVICE = torch.device("cuda")
#DEVICE = torch.device("cpu")
print(f"Device: {DEVICE}")
MODEL = TrafficNeuralNetwork()
MODEL = nn.DataParallel(MODEL)
MODEL = MODEL.to(DEVICE)
print(MODEL)

LEARNING_RATE = 1e-4
BATCH_SIZE = 64
EPOCHS = 3

TRAIN_SPLIT = 0.8
VAL_SPLIT = 1 - TRAIN_SPLIT

# Optimizer can be changed (ex SGD, Adam)
OPTIMIZER = torch.optim.Adam(MODEL.parameters(), lr=LEARNING_RATE)
# Loss function can be changed (ex NLLLoss, CrossEntropyLoss)
LOSS_FUNCTION = nn.NLLLoss()

HISTORY = {
    "train_loss": [],
    "train_loss_min": np.Inf,
    "train_acc": [],
    "val_loss": [],
    "val_acc": []
}

# # For now remove SSL certification because is not working
# # Remove when SSL cert is working
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

# Dataset can be changed here
CNN_MEAN = [0.485, 0.456, 0.406]
CNN_STD = [0.229, 0.224, 0.225]
transforms = v2.Compose([
    #torchvision.transforms.Lambda(lambda x: x / 255.),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=CNN_MEAN, std=CNN_STD),
    v2.Resize((250, 250)),
])

train_data_init = datasets.GTSRB(root="data", split="train", download=True, transform=ToTensor())
test_data = datasets.GTSRB(root="data", split="test", download=True, transform=ToTensor())

train_data_init = datasets.GTSRB(root="data", split="train", download=True, transform=transforms)
test_data = datasets.GTSRB(root="data", split="test", download=True, transform=transforms)

train_sample_size = int(len(train_data_init) * TRAIN_SPLIT)
val_sample_size = len(train_data_init) - train_sample_size
train_data, val_data = random_split(train_data_init, [train_sample_size, val_sample_size], generator=torch.Generator().manual_seed(42)) # manual seed for reproducability

train_dataloader = DataLoader(train_data, batch_size = BATCH_SIZE)
val_dataloader = DataLoader(val_data, batch_size = BATCH_SIZE)
test_dataloader = DataLoader(test_data, batch_size = BATCH_SIZE)

train_steps = len(train_dataloader.dataset) // BATCH_SIZE
val_steps = len(val_dataloader.dataset) // BATCH_SIZE
test_steps = len(test_dataloader.dataset) // BATCH_SIZE

# If dataset is changed, update class labels here
#classes = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")

for e in range(EPOCHS):
    print(f"Epoch {e+1}")
    print(train(MODEL, train_dataloader, val_dataloader, LOSS_FUNCTION, OPTIMIZER, DEVICE, train_steps, val_steps, HISTORY))
    #if (HISTORY["train_loss"][-1] == HISTORY["train_loss_min"]):
    torch.save(MODEL.state_dict(), "traffic_model_2.pt")
    # if val loss < val loss min -- TODO
    print("-"*50)

MODEL.load_state_dict(torch.load("traffic_model_2.pt"))

cr, examples = test(MODEL, test_dataloader, LOSS_FUNCTION, DEVICE, test_data)
print(cr)

cnt = 0
plt.figure(figsize=(8,10))
for i in range(len(examples)):
    cnt += 1
    plt.subplot(1, 5, cnt)
    plt.xticks([], [])
    plt.yticks([], [])
    preds, img = examples[i]
    plt.title(f"{preds.argmax(1)[0]}")
    plt.imshow(img[0,:,:], cmap="gray")
plt.tight_layout()
plt.show()

#torch.save(MODEL.state_dict(), "mnist_model_1.pt")