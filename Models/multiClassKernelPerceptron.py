from Models.kernelPerceptron import KernelPerceptron
import Models.kernelFunctions as kernelFunctions

import time
import numpy as np

class MultiClassKernelPerceptron():
    """One vs all multi class kernel perceptron"""

    def __init__(self, kernel, hyperparameters=None):
        self.perceptrons = []
        self.kernel = kernel
        self.hyperparameters = hyperparameters

    def train(self, x_train, y_train, x_val, y_val):
        # Create 1 perceptron for each unique label
        for label in np.unique(y_train):
            self.perceptrons.append(KernelPerceptron(label, self.kernel, self.hyperparameters))
        # Kernel matrix are same for all classes so calc once and pass around
        kernel_matrix_train = kernelFunctions.calc_kernel_matrix(self.kernel, self.hyperparameters, x_train)
        kernel_matrix_val = kernelFunctions.calc_kernel_matrix(self.kernel, self.hyperparameters, x_train, x_val)
        for model in self.perceptrons:
            model.train(x_train, y_train, x_val, y_val, kernel_matrix_train, kernel_matrix_val)

    def predict(self, X):
        # Each model gives certainty that image belongs to its class
        perceptron_certainities = np.zeros((len(X), len(self.perceptrons)))
        for i, perceptron in enumerate(self.perceptrons):
            perceptron_certainities[:, i] = perceptron.predict(X, mapToClassLabels=False)
        # Index of perceptron with max certainty
        index_max_certainity_perceptron = np.argmax(perceptron_certainities, axis = 1)
        # Return class label for most certain perceptron for each image 
        return np.array([self.perceptrons[i].class_label for i in index_max_certainity_perceptron])

    def saveModel(self):
        timestampStr = datetime.now().strftime("%d_%b_%H_%M_%S")
        with open(f"savedModels/multiClassKernelPerceptron_{timestampStr}", 'wb') as pickleFile:
            pickle.dump(self, pickleFile)

    @staticmethod
    def loadModel(savedModelFname):
        with open(savedModelFname, 'rb') as pickleFile:      
            return pickle.load(pickleFile)