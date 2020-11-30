import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class MnistDigits:
    def __init__(self, fname):
        self.data = np.genfromtxt(fname)
        self.image_size = 16
    
    def get_image(self, index):
        return self.data[index, 1:]
    
    def get_label(self, index):
        return self.data[index, 0]
    
    def visualise_sample(self, index):
        label = self.get_label(index)
        image = self.get_image(index).reshape(self.image_size, self.image_size)
        
        plt.imshow(image)
        plt.title(f"Index: {index}, Label: {label}")
        plt.show()
        

if __name__ == "__main__":
    data = MnistDigits(r"Data\dtest123.dat")
    data.visualise_sample(454)