import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from sklearn.metrics import classification_report

class BaseNeuralNetwork(nn.Module):
    def __init__(self, device, num_channels, num_features, num_out_features, batch_size, train_dataloader, val_dataloader, test_dataloader, test_data):
        super(BaseNeuralNetwork, self).__init__()

        self.device = device
        self.batch_size = batch_size

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.test_data = test_data
        self.num_channels = num_channels

        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }
        
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Layers 2 and 3 not needed - included in case of need in future 

        # self.conv_layer2 = nn.Sequential(
        #     nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Dropout2d(p=0.1)
        # )
    
        # self.conv_layer3 = nn.Sequential(
        #     nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Dropout2d(p=0.1)
        # )
    
        self.fc_layer = nn.Sequential(
            nn.Linear(in_features=num_features, out_features=512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=0.1),
            nn.Linear(in_features=256, out_features=num_out_features)
        )

    def forward(self, x):
        x = self.conv_layer1(x)

        # Layers 2 and 3 not needed - included in case of need in future 
        #x = self.conv_layer2(x)
        #x = self.conv_layer3(x)

        # Flatten input into 1D vector
        x = x.view(x.size(0), -1)

        x = self.fc_layer(x)

        return F.log_softmax(x, dim=1)
    
    def train_model(self, loss_function, optimizer):
        self.train()

        total_train_loss = 0
        total_val_loss = 0

        total_train_correct = 0
        total_val_correct = 0
    
        for (inputs, labels) in self.train_dataloader:
            (inputs, labels) = (inputs.to(self.device), labels.to(self.device))
    
            optimizer.zero_grad()
        
            pred = self(inputs)
            loss = loss_function(pred, labels)
            
            loss.backward()
            optimizer.step()

            total_train_loss += loss
            total_train_correct += (pred.argmax(1) == labels).type(torch.float).sum().item()

        with torch.no_grad():
            self.eval()

            for (inputs, labels) in self.val_dataloader:
                (inputs, labels) = (inputs.to(self.device), labels.to(self.device))

                pred = self(inputs)
                loss = loss_function(pred, labels)

                total_val_loss += loss
                total_val_correct += (pred.argmax(1) == labels).type(torch.float).sum().item()

        # Train and Val Steps
        avg_train_loss = total_train_loss / ( len(self.val_dataloader.dataset) / self.batch_size)
        avg_val_loss = total_val_loss / ( len(self.val_dataloader.dataset) / self.batch_size)

        train_correct = total_train_correct / len(self.train_dataloader.dataset)
        val_correct = total_val_correct / len(self.val_dataloader.dataset)

        self.history["train_loss"].append(avg_train_loss.cpu().detach().numpy())
        self.history["train_acc"].append(train_correct)
        self.history["val_loss"].append(avg_val_loss.cpu().detach().numpy())
        self.history["val_acc"].append(val_correct)

        # Do I need to return the history??
        return self.history
    
    def test_model(self, loss_function):
        self.eval()

        total_test_loss = 0
        total_test_correct = 0
        preds = []

        for (inputs, labels) in self.test_dataloader:
            (inputs, labels) = (inputs.to(self.device), labels.to(self.device))

            pred = self(inputs)
            loss = loss_function(pred, labels)

            total_test_loss += loss
            total_test_correct += (pred.argmax(1) == labels).type(torch.float).sum().item()

            preds.extend(pred.argmax(axis=1).cpu().numpy())

        cr = classification_report(self.test_data.targets, np.array(preds), target_names=self.test_data.classes)

        # Preds are the array of probability percentage
        return cr, preds
    
    def test_attack_model(self, loss_function, attack):
        self.eval()

        total_test_loss = 0
        total_test_correct = 0
        preds = []

        for (inputs, labels) in self.test_dataloader:
            (inputs, labels) = (inputs.to(self.device), labels.to(self.device))

            init_pred = self(inputs)
            init_loss = loss_function(init_pred, labels)

            self.zero_grad()
            init_loss.backward()

            input_attack = attack.forward(inputs, labels)

            attack_pred = self(input_attack)
            attack_loss = loss_function(attack_pred, labels)

            total_test_loss += attack_loss
            total_test_correct += (attack_pred.argmax(1) == labels).type(torch.float).sum().item()

            preds.extend(attack_pred.argmax(axis=1).cpu().numpy())

        cr = classification_report(self.test_data.targets, np.array(preds), target_names=self.test_data.classes)

        # Preds are the array of probability percentage
        return cr, preds
    
    def save_model(self, model_name):
        torch.save(self.state_dict(), f"{model_name}_model.pt")
        print(f"Model {model_name} Saved")
    
    def load_model(self, model_name):
        self.load_state_dict(torch.load(f"{model_name}_model.pt"))