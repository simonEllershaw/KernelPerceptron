from mnistDigitLoader import MnistDigits
import numpy as np
from datetime import datetime
import time

class KernelPerceptron():
    def __init__(self, savedWeightsFname=None):
        # Either load or zero weights
        if savedWeightsFname: 
            self.w = self.loadModel(savedWeightsFname) 
        else: 
            self.w = np.zeros((MnistDigits.image_size**2))

    def loadModel(self, savedWeightsFname):
        return np.load(savedWeightsFname)

    def saveModel(self):
        timestampStr = datetime.now().strftime("%d_%b_%H_%M_%S")
        np.save(f"savedModels/kernelPerceptron_{timestampStr}", self.w)

    def train(self, data):
        num_epochs = 2

        for i in range(num_epochs):
            for t in range(len(data)):
                # Get sample
                x_t = data.get_image(t)
                y = data.get_label(t)
                # Make prediction
                y_hat = -1 if self.w.T@x_t < 0 else 1

                if y_hat != y:
                    self.w += y*x_t
        self.saveModel()

if __name__ == "__main__":
    model = KernelPerceptron()
    data = MnistDigits(r"Data\dtest123.dat")
    model.train(data)

    model2 = KernelPerceptron(r"C:\Users\SIMON\Documents\Code\SupervisedLearning\mnistExperiments\savedModels\kernelPerceptron_30_Nov_16_51_33.npy")
    
