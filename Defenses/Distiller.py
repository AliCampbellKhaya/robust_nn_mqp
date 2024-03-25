import torch
from torch import optim
from torch import functional as F
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


#     def __init__(self, student, teacher):
#         self.teacher = teacher
#         self.student = student
        
#         def compile(
#             self,
#             optimizer,
#             metrics,
#             student_loss_fn,
#             distillation_loss_fn,
#             alpha=0.1,
#             temperature=3,
#         ):
#             """ Configure the distiller.student_loss_fn: Loss function of difference between student
#                     predictions and ground-truth
#                 distillation_loss_fn: Loss function of difference between soft
#                                 student predictions and soft teacher predictions
#                 alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
#                 temperature: Temperature for softening probability distributions.
#                                 Larger temperature gives softer distributions.
#             """
#         super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
#         self.student_loss_fn=student_loss_fn
#         self.distillation_loss_fn= distillation_loss_fn
#         self.temperature= temperature
#         self.alpha= alpha
        
#     def train_step(self, data):
#         x,y=data
        
#         # Forward pass of teacher
#         teacher_prediction=self.teacher(x, training=False)
#         print("Tecaher prediction   ...", teacher_prediction)
#         with tf.GradientTape() as tape:
#             # Forward pass of student
#             student_predcition= self.student(x, training=True)
#             # Compute losses
#             student_loss= self.student_loss_fn(y, student_predcition)
            
#             distillation_loss=self.distillation_loss_fn(
#             tf.nn.softmax(teacher_prediction/self.temperature, axis=1),
#             tf.nn.softmax(student_predcition/self.temperature, axis=1)
#             )
#             loss= self.alpha* student_loss + (1-self.alpha)* distillation_loss
#             print("Loss in distiller :",loss)
#             # Compute gradients
#             trainable_vars= self.student.trainable_variables
#             gradients=tape.gradient(loss, trainable_vars)
#             gradients = [gradient * (self.temperature ** 2) for gradient in gradients]
#             # Update weights
#             self.optimizer.apply_gradients(zip(gradients, trainable_vars))
            
#             # Update the metrics configured in `compile()`
#             self.compiled_metrics.update_state(y, student_predcition)
            
#             # Return a dict of performance
#             results={ m.name: m.result()  for m in self.metrics}
#             results.update({"student_loss": student_loss, "distillation_loss": distillation_loss})
#             print("Train...", results)
#             return results
        
#     def test_step(self, data):
#         # Unpack the data
#         x, y = data
        
#         ## Compute predictions
#         y_prediction= self.student(x, training=False)
        
#         # calculate the loss
#         student_loss= self.student_loss_fn(y, y_prediction)
        
#         # Update the metrics.
#         self.compiled_metrics.update_state(y, y_prediction)
        
#         # Return a dict of performance
#         results ={m.name: m.result() for m in self.metrics}
#         results.update({"student_loss": student_loss})
#         print("Test...", results)
#         return results# Initialize  distiller
    
# distiller= Distiller(student=student, teacher=teacher)

# #compile distiller
# distiller.compile(optimizer=keras.optimizers.Adam(),
#                  metrics=[keras.metrics.SparseCategoricalAccuracy()],
#                  student_loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#                  distillation_loss_fn=keras.losses.KLDivergence(),
#                  alpha=0.3,
#                  temperature=7)