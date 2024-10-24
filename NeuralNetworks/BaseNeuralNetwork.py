import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

class BaseNeuralNetwork(nn.Module):
    def __init__(self, dataset_name, device, num_channels, num_features, num_out_features, batch_size, train_dataloader, val_dataloader, test_dataloader, test_data):
        super(BaseNeuralNetwork, self).__init__()

        self.dataset_name = dataset_name

        self.device = device
        self.batch_size = batch_size

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.test_data = test_data
        self.num_channels = num_channels
        self.num_classes = num_out_features

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

        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.1)
        )
    
        self.fc_layer = nn.Sequential(
            nn.Linear(in_features=num_features, out_features=512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=256, out_features=num_out_features)
        )

    def forward(self, x):
        x = self.conv_layer1(x)

        x = self.conv_layer2(x)

        # Flatten input into 1D vector
        x = x.view(x.size(0), -1)

        x = self.fc_layer(x)

        return F.log_softmax(x, dim=1)
    
    # Same method as forward pass but returns the logits
    def forward_omit_softmax(self, x):
        x = self.conv_layer1(x)

        x = self.conv_layer2(x)

        x = x.view(x.size(0), -1)

        x = self.fc_layer(x)

        return x
    
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

        # if avg_val_loss.cpu().detach().numpy() <= min(self.history["val_loss"]):
        #     self.save_model()

        return self.history
    
    
    
    def train_model_distiller(self, loss_function, optimizer):
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

        return self.history
    
    def test_model(self, loss_function):
        self.eval()

        total_test_loss = 0
        total_test_correct = 0
        preds = []
        preds_true = []
        examples = []

        pred_probs = 0

        for (inputs, labels) in self.test_dataloader:
            (inputs, labels) = (inputs.to(self.device), labels.to(self.device))

            pred = self(inputs)
            loss = loss_function(pred, labels)

            total_test_loss += loss
            total_test_correct += (pred.argmax(1) == labels).type(torch.float).sum().item()

            preds.extend(pred.argmax(axis=1).cpu().numpy())
            preds_true.extend(labels.cpu().numpy())

            # TODO: Fix example generator
            if len(examples) < 5:
                examples.append( (pred, inputs.squeeze().detach().cpu()) )
            else:
                break

            pred_probs = pred[0].detach().cpu().numpy()

            break

        # TODO: Fix classification
        #cr1 = classification_report(self.test_data.targets, np.array(preds), target_names=self.test_data.classes)
        cr = classification_report(np.array(preds_true), np.array(preds), zero_division=0.0) # target_names =

        # Preds are the array of probability percentage
        return cr, pred_probs, examples
    
    def test_attack_model(self, loss_function, attack):
        self.eval()

        total_test_loss = 0
        total_test_correct = 0
        preds = []
        preds_true = []
        examples = []

        results = {
            "pert_image": [],
            "final_label": [],
            "attack_label": [],
            "iterations": [],
            "perturbations": [],
            "original_image": []
        }

        pred_probs = 0

        for (inputs, labels) in self.test_dataloader:
            (inputs, labels) = (inputs.to(self.device), labels.to(self.device))

            init_pred = self(inputs)
            init_loss = loss_function(init_pred, labels)

            self.zero_grad()
            init_loss.backward()

            input_attack_results = attack.forward(inputs, labels)

            attack_pred = self(input_attack_results[0])
            attack_loss = loss_function(attack_pred, labels)

            total_test_loss += attack_loss
            total_test_correct += (attack_pred.detach().argmax(1) == labels).type(torch.float).sum().item()

            preds.extend(attack_pred.detach().argmax(axis=1).cpu().numpy())
            preds_true.extend(labels.cpu().numpy())

            results["pert_image"] += input_attack_results[1]["pert_image"]
            results["final_label"] += input_attack_results[1]["final_label"]
            results["attack_label"] += input_attack_results[1]["attack_label"]
            results["iterations"] += input_attack_results[1]["iterations"]
            results["perturbations"] += input_attack_results[1]["perturbations"]
            results["original_image"] += input_attack_results[1]["original_image"]

            # TODO: A better way of generating examples
            if len(examples) < 5:
            #for i in range(5):
                #examples.append( (results["original_image"][i], results["final_label"][i], results["pert_image"][i].squeeze().detach().cpu()) )
                examples.append( (init_pred, attack_pred, input_attack_results[0].squeeze().detach().cpu()) )
            #else:
                #break

            pred_probs = attack_pred[0].detach().cpu().numpy()


            break

        #cr1 = classification_report(self.test_data.targets, np.array(preds), target_names=self.test_data.classes)
        cr = classification_report(np.array(preds_true), np.array(preds), zero_division=0.0) # target_names =
        #cr = classification_report(results["final_label"], results["attack_label"])

        # Preds are the array of probability percentage
        return cr, pred_probs, examples, results
    
    def test_defense_model(self, loss_function, attack, defense):
        self.eval()

        total_test_loss = 0
        total_test_correct = 0
        preds = []
        preds_true = []
        examples = []

        results = {
            "defended_image": [],
            "final_label": [],
            "attack_label": [],
            "original_image": []
        }

        for (inputs, labels) in self.test_dataloader:
            (inputs, labels) = (inputs.to(self.device), labels.to(self.device))

            init_pred = self(inputs)
            init_loss = loss_function(init_pred, labels)

            self.zero_grad()
            init_loss.backward()

            #input_attack = defense.forward(inputs, labels)
            #input_defense = attack.forward(input_attack[0], labels)

            input_attack = attack.forward(inputs, labels)
            input_defense = defense.forward(input_attack[0], labels)

            #attack_pred = self(torch.from_numpy(input_defense).float())
            attack_pred = self(input_defense[0])
            attack_loss = loss_function(attack_pred, labels)

            total_test_loss += attack_loss
            total_test_correct += (attack_pred.argmax(1) == labels).type(torch.float).sum().item()

            preds.extend(attack_pred.argmax(axis=1).cpu().numpy())
            preds_true.extend(labels.cpu().numpy())

            # results["defended_image"] += input_defense[1]
            # results["final_label"] += attack_pred.detach().cpu().numpy()
            # results["attack_label"] += input_attack[1]["attack_label"]
            # results["original_image"] += input_attack[1]["original_image"]

            if len(examples) < 5:
                examples.append( (init_pred, attack_pred, input_defense[0].squeeze().detach().cpu()) )

            else:
                break

            #break

        #cr1 = classification_report(self.test_data.targets, np.array(preds), target_names=self.test_data.classes)
        cr = classification_report(np.array(preds_true), np.array(preds), zero_division=0.0) # target_names =

        # Preds are the array of probability percentage
        return cr, preds, examples, results
        
    def test_baseline_defense_model(self, loss_function, defense):
        self.eval()

        total_test_loss = 0
        total_test_correct = 0
        preds = []
        preds_true = []
        examples = []

        for (inputs, labels) in self.test_dataloader:
            (inputs, labels) = (inputs.to(self.device), labels.to(self.device))

            init_pred = self(inputs)
            init_loss = loss_function(init_pred, labels)

            self.zero_grad()
            init_loss.backward()

            input_defense = defense.forward(inputs, labels)

            #attack_pred = self(torch.from_numpy(input_defense).float())
            attack_pred = self(input_defense[0])
            attack_loss = loss_function(attack_pred, labels)

            total_test_loss += attack_loss
            total_test_correct += (attack_pred.argmax(1) == labels).type(torch.float).sum().item()

            preds.extend(attack_pred.argmax(axis=1).cpu().numpy())
            preds_true.extend(labels.cpu().numpy())

            if len(examples) < 5:
                examples.append( (init_pred, attack_pred, input_defense[0].squeeze().detach().cpu()) )

            else:
                break

        #cr1 = classification_report(self.test_data.targets, np.array(preds), target_names=self.test_data.classes)
        cr = classification_report(np.array(preds_true), np.array(preds), zero_division=0.0) # target_names =

        # Preds are the array of probability percentage
        return cr, preds, examples
    
    def generate_example_images(self):
        pass
    
    def display_images(self, examples):
        cnt = 0
        plt.figure(figsize=(8,10))
        for i in range(len(examples)):
            cnt += 1
            plt.subplot(1, 5, cnt)
            plt.xticks([], [])
            plt.yticks([], [])
            preds, img = examples[i]
            plt.title(f"{preds.argmax(1)[0]}")
            #plt.imshow(img[0,:,:], cmap="gray")
            plt.imshow(img[0,:,:].permute(1, 2, 0))
        plt.tight_layout()
        plt.show()


    def display_attacked_images(self, examples):
        cnt = 0
        plt.figure(figsize=(8,10))
        for i in range(len(examples)):
            cnt += 1
            plt.subplot(1, 5, cnt)
            plt.xticks([], [])
            plt.yticks([], [])
            init_preds, attack_preds, img = examples[i]
            plt.title(f"{(init_preds.argmax(1)[0])} -> {(attack_preds.argmax(1)[0])}")
            #plt.imshow(img[0,:,:], cmap="gray")
            plt.imshow(img[0,:,:].permute(1, 2, 0))
        plt.tight_layout()
        plt.show()
    
    def save_model(self):
        torch.save(self.state_dict(), f"SavedModels/{self.dataset_name}_model.pt")
        print(f"Model {self.dataset_name} Saved")
    
    def load_model(self):
        self.load_state_dict(torch.load(f"SavedModels/{self.dataset_name}_model.pt"))

    def save_defense_model(self, defense_name):
        torch.save(self.state_dict(), f"SavedModels/{self.dataset_name}_{defense_name}_model.pt")
        print(f"Model {self.dataset_name} with Defense {defense_name} Saved")

    def load_defense_model(self, defense_name):
        self.load_state_dict(torch.load(f"SavedModels/{self.dataset_name}_{defense_name}_model.pt"))
