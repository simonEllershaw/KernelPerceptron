import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math


class MnistDigits:
    image_size = 16

    def __init__(self, fname):
        self.data = np.genfromtxt(fname)
        self.label_column = 0

    def get_split_datasets(self, fraction_test=0.2, fraction_val=0):
        num_samples = self.data.shape[0]
        divider_test = math.floor(fraction_test * num_samples)
        divider_val = divider_test + math.floor(fraction_val * num_samples)

        np.random.shuffle(self.data)
        images = self.data[:, self.label_column+1:]
        labels = self.data[:, self.label_column]

        return {
            "images_test": images[:divider_test],
            "images_val": images[divider_test:divider_val],
            "images_train": images[divider_val:],
            "labels_test": labels[:divider_test],
            "labels_val": labels[divider_test:divider_val],
            "labels_train": labels[divider_val:]
        }

    @staticmethod
    def visualise_sample(image, label):
        image = image.reshape(MnistDigits.image_size, MnistDigits.image_size)
        plt.imshow(image)
        plt.title(f"Label: {label}")
        plt.show()

    


if __name__ == "__main__":
    datasets = MnistDigits(r"Data\dtest123.dat").get_split_datasets()
    MnistDigits.visualise_sample(datasets["images_train"][0], datasets["labels_train"][0])