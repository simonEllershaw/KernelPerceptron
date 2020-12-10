import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import time
import os
from datetime import datetime

from mnistDigitLoaderPyTorch import MnistDigitsPytorch

class LeNet5(nn.Module):
    """
        Reference https://github.com/bollakarthikeya/LeNet-5-PyTorch/blob/master/lenet5_cpu.py
    """
    def __init__(self):   
        super(LeNet5, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=1, padding=2, bias=True)
        self.max_pool_1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=0, bias=True)
        self.max_pool_2 = torch.nn.MaxPool2d(kernel_size=2)
        self.fc1 = torch.nn.Linear(144, 120)   # Modifiction
        self.fc2 = torch.nn.Linear(120, 84)      
        self.fc3 = torch.nn.Linear(84, 10)
        
    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))  
        x = self.max_pool_1(x)
        x = torch.nn.functional.relu(self.conv2(x))
        x = self.max_pool_2(x)
        # Modification
        x = x.view(-1, 144)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train(model, dataloaders, device, criterion, optimizer):#, directory,
    dataset_sizes = {x: len(dataloaders[x])* dataloaders[x].batch_size for x in ["train", "val"]}
    best_loss = float("inf")
    numberEpochsWtNoImprovement = 0
    epoch = 0
    timestampStr = datetime.now().strftime("%d_%b_%H_%M_%S")
    checkpoint_fname = os.path.join("savedModels", f"LeNet5_{timestampStr}.tar")

    while numberEpochsWtNoImprovement < 2:
        epoch += 1
        trainLoss = 0
        valLoss = 0
        epochStart = time.time()

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()

            running_loss = 0.0
            for source, target in dataloaders[phase]:
                source = source.to(device)
                target = target.to(device).long()
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(source)
                    loss = criterion(outputs, target)
                    # backward + optimize only if in training phase
                    if phase == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item()

            epoch_loss = running_loss / dataset_sizes[phase]
            if phase == "train":
                trainLoss = epoch_loss
            elif phase == "val":
                valLoss = epoch_loss
                # Improvement has to be at least by 0.1%
                if epoch_loss < best_loss * 0.999:
                    best_loss = epoch_loss
                    torch.save(model.state_dict(), checkpoint_fname)
                    numberEpochsWtNoImprovement = 0
                else:
                    numberEpochsWtNoImprovement += 1

        epochEnd = time.time()
        epochTime = epochEnd - epochStart

        # Write stats to file at end of each epoch
        # with open(os.path.join(directory, "metrics.txt"), "a+") as outfile:
        print(f"{epoch} {trainLoss} {valLoss} {best_loss} {epochTime} \n")



def evaluate(model, dataset):
    print(len(dataset))
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in dataset:
            output = model(data)
            _, pred = torch.max(output, dim=1)
            correct += (pred == target).sum().item()
    num_samples = len(dataset) * dataset.batch_size
    return correct / num_samples

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = LeNet5().to(device).float()
    learning_rate = 0.01
    momentum = 0.5
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                      momentum=momentum)
    criterion = nn.CrossEntropyLoss()
    dataloaders = MnistDigitsPytorch.getDataLoader(12, "Data\zipcombo.dat")
    train(model, dataloaders, device, criterion, optimizer,)
    # model.load_state_dict(torch.load("savedModels\LeNet5_10_Dec_17_55_51.tar"))
    # test_acc = evaluate(model, dataloaders["test"])
    # print(test_acc)