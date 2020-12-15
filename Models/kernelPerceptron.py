import Models.kernelFunctions as kernelFunctions
import numpy as np
from datetime import datetime
import pickle
import time

class KernelPerceptron():
    def __init__(self, class_label, kernel, hyperparameters=None):
        self.kernel = kernel
        self.class_label = class_label
        self.hyperparameters = hyperparameters

    def train(self, x_train, y_train, x_val, y_val, kernel_matrix_train=None, kernel_matrix_val=None):     
        print(f"Training Class {self.class_label}")   
        # Set up training variables
        n_samples, _ = x_train.shape
        alpha_training = np.zeros(n_samples)
        y_train = self.classify_labels(y_train)
        y_val = self.classify_labels(y_val)
        epochs_since_val_acc_improvement = 0
        best_val_accuracy = -1
        epoch = 0
        
        # Calc kernel matrices if not given in function call
        if kernel_matrix_train is None:
            kernel_matrix_train = kernelFunctions.calc_kernel_matrix(self.kernel, self.hyperparameters, x_train, x_train)
        if kernel_matrix_val is None:
            kernel_matrix_val = kernelFunctions.calc_kernel_matrix(self.kernel, self.hyperparameters, x_train, x_val)

        # Table header
        print("Epoch|Train Acc|Val Acc|Best Val Acc|")
        
        while(epochs_since_val_acc_improvement < 2):
            m = 0
            epoch += 1
            for t in range(n_samples):
                # Make prediction
                y_hat = 1 if self.distance_to_hyperplane(alpha_training, kernel_matrix_train, t) > 0 else -1 
                #Update weights
                if y_hat != y_train[t]:
                    alpha_training[t] += y_train[t]
                    m +=1
            training_accuracy = 1 - (float(m)/n_samples)
            # After each training loop check if val accuracy has increased
            epoch_val_acc = self.val_accuracy(kernel_matrix_val, y_val, alpha_training)
            if epoch_val_acc > best_val_accuracy:
                # If it has save updated model
                epochs_since_val_acc_improvement = 0
                best_val_accuracy = epoch_val_acc
                self.saved_alpha = alpha_training
            else:
                epochs_since_val_acc_improvement += 1
            print(f"{epoch} {training_accuracy:.4f} {epoch_val_acc:.4f} {best_val_accuracy:.4f}")

        self.set_support_vectors(self.saved_alpha, x_train)

    def classify_labels(self, labels):
        return np.where(labels == self.class_label, 1, -1)
    
    @staticmethod
    def distance_to_hyperplane(alpha, kernel_matrix, t):
        return np.sum(alpha*kernel_matrix[:, t])

    def set_support_vectors(self, alpha_training, training_X):
        # Non zero alphas and corresponding training image stored in object
        support_vector_indicies = np.nonzero(alpha_training)
        self.saved_alpha = alpha_training[support_vector_indicies]
        self.support_vectors = training_X[support_vector_indicies]

    def calc_certainites(self, kernel_matrix, alpha):
        # Calc kernel image based off support vectors and predict using alphas
        n_samples = kernel_matrix.shape[1]
        certainties = np.zeros(n_samples)
        for t in range(n_samples):
            certainties[t] = KernelPerceptron.distance_to_hyperplane(alpha, kernel_matrix, t)
        return certainties

    def predict(self, X, mapToClassLabels = True):
        kernel_matrix = kernelFunctions.calc_kernel_matrix(self.kernel, self.hyperparameters, self.support_vectors, X)
        certainties = self.calc_certainites(kernel_matrix, self.saved_alpha)
        return np.where(certainties > 0, self.class_label, -1) if mapToClassLabels else certainties

    def val_accuracy(self, kernel_matrix_val, y_val, alpha_training):
        y_pred = np.where(self.calc_certainites(kernel_matrix_val, alpha_training) > 0, 1, -1)
        return np.count_nonzero(y_pred==y_val) / float(len(y_val))

    def saveModel(self):
        timestampStr = datetime.now().strftime("%d_%b_%H_%M_%S")
        with open(f"savedModels/kernelPerceptron_{timestampStr}", 'wb') as pickleFile:
            pickle.dump(self, pickleFile)

    @staticmethod
    def loadModel(savedModelFname):
        with open(savedModelFname, 'rb') as pickleFile:      
            return pickle.load(pickleFile)

    
