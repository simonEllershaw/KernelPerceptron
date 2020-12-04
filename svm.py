import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import time

from mnistDigitLoader import MnistDigits

class SVM():
    def __init__(self, class_label, kernel, hyperparameters=None):
        self.model = SVC(kernel='poly', degree=8)

    def train(self, x_train, y_train):     
        self.model.fit(x_train, y_train)

    def predict(self, X, mapToClassLabels = True):
        return self.model.predict(X) 

    def saveModel(self):
        timestampStr = datetime.now().strftime("%d_%b_%H_%M_%S")
        with open(f"savedModels/SVM_{timestampStr}", 'wb') as pickleFile:
            pickle.dump(self, pickleFile)

    @staticmethod
    def loadModel(savedModelFname):
        with open(savedModelFname, 'rb') as pickleFile:      
            return pickle.load(pickleFile)

if __name__ == "__main__":
    data = MnistDigits(r"Data\dtrain123.dat").get_split_datasets()

    model = SVM(3, 'poly', hyperparameters=3)
    model.train(data["images_train"], data["labels_train"])
    y_pred = model.predict(data["images_test"])

    print(confusion_matrix(data["labels_test"], y_pred))
    print(classification_report(data["labels_test"], y_pred))