import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import time

from mnistDigitLoader import MnistDigits

class SupportVectorMachine():
    def __init__(self, class_label, kernel, hyperparameters=None):
        self.class_label = class_label
        self.model = SVC(kernel='poly', degree=8)

    def train(self, x_train, y_train):     
        masked_labels = self.classify_labels(y_train)
        self.model.fit(x_train, masked_labels)

    def classify_labels(self, labels):
        return np.where(labels == self.class_label, 1, -1)

    def predict(self, X, mapToClassLabels = True):
        predictions = self.model.predict(X)
        return np.where(predictions == 1, self.class_label, -1) if mapToClassLabels else predictions 

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
    data = MnistDigits(r"Data\dtrain123.dat").get_split_datasets()
    model = SupportVectorMachine(3, kernel='poly', hyperparameters=3)
    model.train(data["images_train"], data["labels_train"])
    predict = model.predict(data["images_test"])
    test_gt = np.where(data["labels_test"] == 3, 3, -1)
    print("Test Acc:", np.count_nonzero(test_gt==predict) / float(len(test_gt)))
    print("Time taken:", time.time() - t1)