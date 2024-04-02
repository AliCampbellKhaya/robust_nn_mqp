import torch
import random

from Defenses.BaseDefense import BaseDefense

from Attacks.FGSM import FGSM
from Attacks.CW import CW
from Attacks.DeepFool import DeepFool
from Attacks.IFGSM import IFGSM
from Attacks.Pixle import Pixle

from NeuralNetworks.MNISTNeuralNetwork import MNISTNeuralNetwork
from NeuralNetworks.CifarNeuralNetwork import CifarNeuralNetwork
from NeuralNetworks.TrafficNeuralNetwork import TrafficNeuralNetwork


"""
TODO: Write Defense
"""

class AdverserialExamples(BaseDefense):
    def __init__(self, model, device, ifgsm, cw, deepfool, pixle, dataset, learning_rate, loss_function):
        super(AdverserialExamples, self).__init__("AE", model, device)
        random.seed(42)

        self.ifgsm = ifgsm
        self.cw = cw
        self.deepfool = deepfool
        self.pixle = pixle

        if dataset == "MNIST":
            self.model = MNISTNeuralNetwork(device, train_split=0.8, batch_size=64).to(device)
        elif dataset == "Cifar":
            self.model = CifarNeuralNetwork(device, train_split=0.8, batch_size=64).to(device)
        elif dataset == "Traffic":
            self.model = TrafficNeuralNetwork(device, train_split=0.8, batch_size=64).to(device)
        else:
            print("Please select a valid dataset: (MNIST, Cifar, Traffic)")

        self.learning_rate = learning_rate
        self.loss_function = loss_function

    def forward_batch(self, inputs, labels):
        r = random.randint(0, 1)

        #if r == 0:
        attack_results = self.ifgsm.forward(inputs, labels)

        #elif r == 1:
        #    attack_results = self.deepfool.forward(inputs, labels)

        # elif r == 2:
        #     attack_results = self.cw.forward(inputs, labels)

        # else:
        #     attack_results = self.pixle.forward(inputs, labels)

        return attack_results[0]
    
    def train_model_adverserial_examples(self, loss_function, optimizer):
        self.model.load_model()
        self.model.train()

        total_train_loss = 0
        total_val_loss = 0

        total_train_correct = 0
        total_val_correct = 0

        count = 0
    
        for (inputs, labels) in self.model.train_dataloader:
            (inputs, labels) = (inputs.to(self.device), labels.to(self.device))
    
            optimizer.zero_grad()

            inputs_attack = self.forward_batch(inputs, labels)
        
            pred = self.model(inputs_attack)
            loss = loss_function(pred, labels)
            
            loss.backward()
            optimizer.step()

            total_train_loss += loss
            total_train_correct += (pred.argmax(1) == labels).type(torch.float).sum().item()

            print("iter:", count)
            count += 1


        # with torch.no_grad():
        #     self.model.eval()

        #     for (inputs, labels) in self.model.val_dataloader:
        #         (inputs, labels) = (inputs.to(self.device), labels.to(self.device))

        #         inputs_attack = self.forward_batch(inputs, labels)

        #         pred = self(inputs_attack)
        #         loss = loss_function(pred, labels)

        #         total_val_loss += loss
        #         total_val_correct += (pred.argmax(1) == labels).type(torch.float).sum().item()
            
        self.model.save_defense_model("Adverserial_Defense")

        # Train and Val Steps
        avg_train_loss = total_train_loss / ( len(self.model.val_dataloader.dataset) / self.model.batch_size)
        avg_val_loss = total_val_loss / ( len(self.model.val_dataloader.dataset) / self.model.batch_size)

        train_correct = total_train_correct / len(self.model.train_dataloader.dataset)
        val_correct = total_val_correct / len(self.model.val_dataloader.dataset)

        self.model.history["train_loss"].append(avg_train_loss)
        self.model.history["train_acc"].append(train_correct)
        self.model.history["val_loss"].append(avg_val_loss)
        self.model.history["val_acc"].append(val_correct)

        #if avg_val_loss.cpu().detach().numpy() <= min(self.history["val_loss"]):
        

        # Do I need to return the history??
        return self.model.history
    

    def test_adverserials(self, loss_function):
        self.model.load_defense_model("Adverserial_Defense")

        cr, preds, examples = self.model.test_model(loss_function)

        return cr, preds, examples
    
    def test_attack_adversarials(self, loss_function, attack):
        self.model.load_defense_model("Adverserial_Defense")

        cr, preds, examples, results = self.model.test_attack_model(loss_function, attack)

        return cr, preds, examples, results
    

            