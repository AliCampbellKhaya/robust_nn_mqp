import torch
from torch import optim
from torch.nn import functional as F
from Defenses.BaseDefense import BaseDefense

from NeuralNetworks.MNISTNeuralNetwork import MNISTNeuralNetwork
from NeuralNetworks.CifarNeuralNetwork import CifarNeuralNetwork
from NeuralNetworks.TrafficNeuralNetwork import TrafficNeuralNetwork

"""
TODO: Write Defense
TODO: convert into actual code
"""

class Distiller(BaseDefense):
    def __init__(self, model, device, dataset, learning_rate, loss_function):
        super(Distiller, self).__init__("Distiller", model, device)
        if dataset == "MNIST":
            self.teacher = MNISTNeuralNetwork(device, train_split=0.8, batch_size=64).to(device)
            self.student = MNISTNeuralNetwork(device, train_split=0.8, batch_size=64).to(device)
        elif dataset == "Cifar":
            self.teacher = CifarNeuralNetwork(device, train_split=0.8, batch_size=64).to(device)
            self.student = CifarNeuralNetwork(device, train_split=0.8, batch_size=64).to(device)
        elif dataset == "Traffic":
            self.teacher = TrafficNeuralNetwork(device, train_split=0.8, batch_size=64).to(device)
            self.student = TrafficNeuralNetwork(device, train_split=0.8, batch_size=64).to(device)
        else:
            print("Please select a valid dataset: (MNIST, Cifar, Traffic)")

        self.learning_rate = learning_rate
        self.loss_function = loss_function

    def train_distillation(self):
        # self.teacher.load_defense_model("Teacher_Distiller")
        # self.student.load_defense_model("Student_Distiller")

        teacher_optimizer = optim.Adam(self.teacher.parameters(), lr=self.learning_rate)
        teacher_history = self.teacher.train_model_distiller(loss_function=self.loss_function, optimizer=teacher_optimizer)

        for (inputs, labels) in self.teacher.train_dataloader:
            (inputs, labels) = (inputs.to(self.device), labels.to(self.device))
            soft_labels = F.log_softmax(self.teacher(inputs), dim=1)
            labels = soft_labels

        student_optimizer = optim.Adam(self.student.parameters(), lr=self.learning_rate)
        student_history = self.student.train_model_distiller(loss_function=self.loss_function, optimizer=student_optimizer)

        self.teacher.save_defense_model("Teacher_Distiller")
        self.student.save_defense_model("Student_Distiller")

        return teacher_history, student_history

    def test_distillation(self):
        self.teacher.load_defense_model("Teacher_Distiller")
        self.student.load_defense_model("Student_Distiller")

        cr, preds, examples = self.student.test_model(self.loss_function)

        return cr, preds, examples
    
    def test_attack_distillation(self, attack):
        self.teacher.load_defense_model("Teacher_Distiller")
        self.student.load_defense_model("Student_Distiller")

        cr, preds, examples, results = self.student.test_attack_model(self.loss_function, attack)

        return cr, preds, examples, results
