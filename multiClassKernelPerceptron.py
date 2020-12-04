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

    def train(self, x_train, y_train, x_val, y_val):
        for label in np.unique(y_train):
            self.perceptrons.append(KernelPerceptron(label, self.kernel, self.hyperparameters))

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
    data = MnistDigits(r"Data\dtrain123.dat").get_split_datasets()
    model = MultiClassKernelPerceptron(np.arange(0,10), kernelFunctions.polynomial_kernel, 3)
    model.train(data["images_train"], data["labels_train"], data["images_val"], data["labels_val"])
    predict = model.predict(data["images_test"])
    print(np.count_nonzero(data["labels_test"]==predict) / float(len(data["images_test"])))
    print(time.time() - t1)