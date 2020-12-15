import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import time

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