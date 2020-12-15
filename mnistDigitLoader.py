import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math

class MnistDigits():
    image_size = 16
    label_column = 0

    def __init__(self, fname):
        self.data = np.genfromtxt(fname)
        
        self.images = self.data[:, MnistDigits.label_column+1:]
        self.labels = self.data[:, MnistDigits.label_column]
    

    def get_split_datasets(self, fraction_test=0.2, fraction_val=0.2):
        num_samples = self.data.shape[0]
        divider_test = math.floor(fraction_test * num_samples)
        divider_val = divider_test + math.floor(fraction_val * num_samples)

        np.random.shuffle(self.data)
        return {
            "images_test": self.images[:divider_test],
            "images_val": self.images[divider_test:divider_val],
            "images_train": self.images[divider_val:],
            "labels_test": self.labels[:divider_test],
            "labels_val": self.labels[divider_test:divider_val],
            "labels_train": self.labels[divider_val:]
        }

    @staticmethod
    def visualise_sample(image, label):
        image = image.reshape(MnistDigits.image_size, MnistDigits.image_size)
        plt.imshow(image)
        plt.title(f"Label: {label}")
        plt.axis('off')
        plt.savefig("Output/digit.png")

if __name__ == "__main__":
    datasets = MnistDigits(r"Data\zipcombo.dat").get_split_datasets()
    MnistDigits.visualise_sample(datasets["images_train"][0], datasets["labels_train"][0])