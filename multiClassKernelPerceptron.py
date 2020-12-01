from kernelPerceptron import KernelPerceptron
from mnistDigitLoader import MnistDigits
import kernelFunctions

import time
import numpy as np

class MultiClassKernelPerceptron():
    """One vs all multi class kernel perceptron"""

    def __init__(self, class_labels, kernel, hyperparameters=None):
        self.perceptrons = []
        self.kernel = kernel
        self.hyperparameters = hyperparameters
        for label in class_labels:
            self.perceptrons.append(KernelPerceptron(label, kernel, hyperparameters))

    def calc_symetric_kernel_matrix(self, data):
        # Kernels used are symetric so at training time so is the kernel matrix
        # Hence calculate top half of matrix and then reflect
        n_samples = len(data)
        kernel_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i, n_samples):
                kernel_matrix[i,j] = self.kernel(data[i], data[j], self.hyperparameters)
        # Reflection
        kernel_matrix = kernel_matrix + kernel_matrix.T - np.diag(kernel_matrix.diagonal())
        return kernel_matrix

    def train(self, data):
        kernel_matrix = self.calc_symetric_kernel_matrix(data.images)
        for model in self.perceptrons:
            model.train(data, kernel_matrix)

    def predict(self, test_images):
        # Each model gives certainty that image belongs to its class
        perceptron_certainities = np.zeros((len(test_images), len(self.perceptrons)))
        for i, perceptron in enumerate(self.perceptrons):
            perceptron_certainities[:, i] = perceptron.calc_certainites(test_images)
        # Index of perceptron with max certainty
        index_max_certainity_perceptron = np.argmax(perceptron_certainities, axis = 1)
        # Return class label for most certain perceptron for each image 
        return [self.perceptrons[i].class_label for i in index_max_certainity_perceptron]

    def saveModel(self):
        timestampStr = datetime.now().strftime("%d_%b_%H_%M_%S")
        with open(f"savedModels/multiClassKernelPerceptron_{timestampStr}", 'wb') as pickleFile:
            pickle.dump(self, pickleFile)

    @staticmethod
    def loadModel(savedModelFname):
        with open(savedModelFname, 'rb') as pickleFile:      
            return pickle.load(pickleFile)

if __name__ == "__main__":
    t1 = time.time()
    data = MnistDigits(r"Data\dtrain123.dat")
    model = MultiClassKernelPerceptron([1,2,3], kernelFunctions.polynomial_kernel, 3)
    model.train(data)
    print(model.predict(data.images))
    print(data.labels)
    print(time.time() - t1)