import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class MnistDigits:
    image_size = 16

    def __init__(self, fname):
        data = np.genfromtxt(fname)
        self.images = data[:, 1:]
        self.labels = data[:, 0]
        self.labels
        self.length = data.shape[0]
    
    def __len__(self):
        return self.length
    
    def visualise_sample(self, index):
        label = self.labels[index]
        image = self.images[index].reshape(self.image_size, self.image_size)

        plt.imshow(image)
        plt.title(f"Index: {index}, Label: {label}")
        plt.show()
        

if __name__ == "__main__":
    data = MnistDigits(r"Data\dtest123.dat")
    data.visualise_sample(4)