from mnistDigitLoader import MnistDigits
import numpy as np
from datetime import datetime
import pickle

class KernelPerceptron():
    def __init__(self, kernel, savedWeightsFname=None):
        self.kernel = kernel
        self.alpha =  []
        self.support_vectors = []

    def calc_kernel_matrix(self, data_1, data_2):
        kernel_matrix = np.zeros((len(data_1), len(data_2)))
        for i in range(len(data_1)):
            for j in range(len(data_2)):
                kernel_matrix[i,j] = self.kernel(data_1[i], data_2[j])
        return kernel_matrix

    def train(self, data):
        num_epochs = 10
        
        # Set up training variables
        n_samples, _ = data.images.shape
        alpha_training = np.zeros(n_samples)
        kernel_matrix = self.calc_kernel_matrix(data.images, data.images)

        for epoch in range(num_epochs):
            m = 0
            for t in range(n_samples):
                # Get sample
                x_t = data.images[t]
                y = data.labels[t]

                # Make prediction
                y_hat = self.predict(alpha_training, kernel_matrix, t)

                #Update weights
                if y_hat != y:
                    alpha_training[t] += y
                    m +=1
            print(t, m, float(m)*100/t)

        self.set_support_vectors(alpha_training, data.images)
        self.saveModel()

    def set_support_vectors(self, alpha_training, training_images):
        # Non zero alphas and corresponding training image stored in object
        support_vector_indicies = np.nonzero(alpha_training)
        self.alpha = alpha_training[support_vector_indicies]
        self.support_vectors = training_images[support_vector_indicies]

    @staticmethod
    def predict(alpha, kernel_matrix, t):
        # Base kernel perceptron algorithm
        return -1 if np.sum(alpha*kernel_matrix[:, t]) < 0 else 1

    def infer(self, test_images):
        predictions = np.zeros(len(test_images))
        # Calc kernel image based off support vectors and predict using alphas
        kernel_matrix = self.calc_kernel_matrix(self.support_vectors, test_images)
        for t in range(len(test_images)):
            predictions[t] = KernelPerceptron.predict(self.alpha, kernel_matrix, t)
        return predictions

    def saveModel(self):
        timestampStr = datetime.now().strftime("%d_%b_%H_%M_%S")
        with open(f"savedModels/kernelPerceptron_{timestampStr}", 'wb') as pickleFile:
            pickle.dump(self, pickleFile)

    @staticmethod
    def loadModel(savedModelFname):
        with open(savedModelFname, 'rb') as pickleFile:      
            return pickle.load(pickleFile)

def dot_product(x_i, x_j):
    return x_i.T@x_j

if __name__ == "__main__":
    data = MnistDigits(r"Data\dtrain123.dat")
    model = KernelPerceptron(dot_product)
    model.train(data)
    # model.saveModel()
    print(model.infer(data.images))

    # model2 = KernelPerceptron(r"savedModels\kernelPerceptron_30_Nov_17_07_41.npy")
    # print(model2.inference(data.get_image(0)))

    
