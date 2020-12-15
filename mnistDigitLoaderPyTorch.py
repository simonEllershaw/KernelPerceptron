from mnistDigitLoader import MnistDigits
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class MnistDigitsPytorch(Dataset):
    def __init__(self, data, mode):
        if mode == "train":
            self.images = data["images_train"]
            self.labels = data["labels_train"]
        elif mode == "val":
            self.images = data["images_val"]
            self.labels = data["labels_val"]
        elif mode == "test":
            self.images = data["images_test"]
            self.labels = data["labels_test"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.images[idx].reshape(1, MnistDigits.image_size, MnistDigits.image_size)
        return image.astype(np.float32), self.labels[idx]

    
    @staticmethod 
    def getDataLoader(data, batch_size):
        MnistDigitsPytorchData = {mode: MnistDigitsPytorch(data, mode) for mode in ["train", "val", "test"]}
        
        dataloaders = {
            "train": torch.utils.data.DataLoader(
                MnistDigitsPytorchData["train"],
                batch_size=batch_size,
                shuffle=True,
                drop_last= True,
            ),
            "val": torch.utils.data.DataLoader(
                MnistDigitsPytorchData["val"],
                batch_size=batch_size,
                shuffle=False,
                drop_last= True
            ),
            "test": torch.utils.data.DataLoader(
                MnistDigitsPytorchData["test"],
                batch_size=batch_size,
                shuffle=False,
                drop_last= True
            ),
        }

        return dataloaders
    
if __name__ == "__main__":
    data = MnistDigits(r"Data\zipcombo.dat").get_split_datasets()
    dataset = MnistDigitsPytorch(data, "train")
    print(dataset[0])