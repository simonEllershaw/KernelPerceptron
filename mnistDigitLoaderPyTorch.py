from mnistDigitLoader import MnistDigits
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class MnistDigitsPytorch(Dataset):
    def __init__(self, fname, mode):
        datasets = MnistDigits(fname).get_split_datasets()
        if mode == "train":
            self.images = datasets["images_train"]
            self.labels = datasets["labels_train"]
        elif mode == "val":
            self.images = datasets["images_val"]
            self.labels = datasets["labels_val"]
        elif mode == "test":
            self.images = datasets["images_test"]
            self.labels = datasets["labels_test"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.images[idx].reshape(1, MnistDigits.image_size, MnistDigits.image_size)
        # label = self.get_one_hot_encode_label(idx)
        return image.astype(np.float32), self.labels[idx]
    
    # def get_one_hot_encode_label(self, idx):
    #     # Hard coded that there are 10 digits sorry :(
    #     label = np.zeros(10)
    #     class_label = int(self.labels[idx])
    #     label[class_label] = 1.0
    #     return label
    
    @staticmethod 
    def getDataLoader(batchSize, fname):
        MnistDigitsPytorchData = {mode: MnistDigitsPytorch(fname, mode) for mode in ["train", "val", "test"]}

        dataloaders = {
            "train": torch.utils.data.DataLoader(
                MnistDigitsPytorchData["train"],
                batch_size=batchSize,
                shuffle=True,
                drop_last= True,
            ),
            "val": torch.utils.data.DataLoader(
                MnistDigitsPytorchData["val"],
                batch_size=batchSize,
                shuffle=False,
                drop_last= True
            ),
            "test": torch.utils.data.DataLoader(
                MnistDigitsPytorchData["test"],
                batch_size=batchSize,
                shuffle=False,
                drop_last= True
            ),
        }

        return dataloaders
    
if __name__ == "__main__":
    data = MnistDigitsPytorch("Data\dtrain123.dat", "train")
    print(data[0])