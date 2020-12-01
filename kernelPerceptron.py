from mnistDigitLoader import MnistDigits
import kernelFunctions
import numpy as np
from datetime import datetime
import pickle
import time

class KernelPerceptron():
    def __init__(self, class_label, kernel, hyperparameters=None):
        self.kernel = kernel
        self.class_label = class_label
        self.hyperparameters = hyperparameters
        self.alpha =  []
        self.support_vectors = []

    def calc_kernel_matrix(self, data_1, data_2):
        kernel_matrix = np.zeros((len(data_1), len(data_2)))
        for i in range(len(data_1)):
            for j in range(len(data_2)):
                kernel_matrix[i,j] = self.kernel(data_1[i], data_2[j], self.hyperparameters)
        return kernel_matrix

    def train(self, data, kernel_matrix=None):
        num_epochs = 10
        
        # Set up training variables
        n_samples, _ = data.images.shape
        alpha_training = np.zeros(n_samples)
        if kernel_matrix is None:
            kernel_matrix = self.calc_kernel_matrix(data.images, data.images)

        for epoch in range(num_epochs):
            m = 0
            for t in range(n_samples):
                # Get sample
                x_t = data.images[t]
                y = 1 if data.labels[t] == self.class_label else -1
                # Make prediction
                y_hat = 1 if self.distance_to_hyperplane(alpha_training, kernel_matrix, t) > 0 else -1 
                #Update weights
                if y_hat != y:
                    alpha_training[t] += y
                    m +=1
            print(t, m, float(m)*100/t)

        self.set_support_vectors(alpha_training, data.images)

    def set_support_vectors(self, alpha_training, training_images):
        # Non zero alphas and corresponding training image stored in object
        support_vector_indicies = np.nonzero(alpha_training)
        self.alpha = alpha_training[support_vector_indicies]
        self.support_vectors = training_images[support_vector_indicies]

    @staticmethod
    def distance_to_hyperplane(alpha, kernel_matrix, t):
        # Base kernel perceptron algorithm
        return np.sum(alpha*kernel_matrix[:, t])

    def calc_certainites(self, test_images):
        certainties = np.zeros(len(test_images))
        # Calc kernel image based off support vectors and predict using alphas
        kernel_matrix = self.calc_kernel_matrix(self.support_vectors, test_images)
        for t in range(len(test_images)):
            certainties[t] = KernelPerceptron.distance_to_hyperplane(self.alpha, kernel_matrix, t)
        return certainties

    def predict(self, test_images):
        return np.where(self.calc_certainites(test_images) > 0, self.class_label, -1)

    def saveModel(self):
        timestampStr = datetime.now().strftime("%d_%b_%H_%M_%S")
        with open(f"savedModels/kernelPerceptron_{timestampStr}", 'wb') as pickleFile:
            pickle.dump(self, pickleFile)

    @staticmethod
    def loadModel(savedModelFname):
        with open(savedModelFname, 'rb') as pickleFile:      
            return pickle.load(pickleFile)

if __name__ == "__main__":
    t1 = time.time()
    data = MnistDigits(r"Data\dtrain123.dat")
    model = KernelPerceptron(2, kernelFunctions.polynomial_kernel, 3)
    model.train(data)
    print(model.predict(data.images))
    print(data.labels)
    print(time.time() - t1)
    # model.saveModel()
    # print(model.infer(data.images))

    
