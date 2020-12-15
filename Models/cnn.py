import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import time
import os
from datetime import datetime

class LeNet5(nn.Module):
    """
        Reference https://github.com/bollakarthikeya/LeNet-5-PyTorch/blob/master/lenet5_cpu.py
    """
    def __init__(self):   
        """ LeNet architecture modified for 16x16 input """
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
        x = x.view(-1, 144) # Modification
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train(model, dataloaders):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device).float()

    # Init training objects and parameters
    # These have not been optimised by a hyperparemeter search
    learning_rate = 0.01
    momentum = 0.5
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                      momentum=momentum)
    criterion = nn.CrossEntropyLoss()
    train_model(model, dataloaders, device, criterion, optimizer)

def train_model(model, dataloaders, device, criterion, optimizer):
    # Initialise training parameters
    dataset_sizes = {x: len(dataloaders[x])* dataloaders[x].batch_size for x in ["train", "val"]}

    best_loss = float("inf")
    numberEpochsWtNoImprovement = 0
    epoch = 0
    print("|Epoch|Train Loss|Val Loss|Best Loss|Epoch Time|")

    # Convergence criteria
    while numberEpochsWtNoImprovement < 2:
        # Epoch parameters
        epoch += 1
        trainLoss = 0
        valLoss = 0
        epochStart = time.time()

        # Alternate train and val
        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()
            running_loss = 0.0
            # Iterate through data
            for source, target in dataloaders[phase]:
                source = source.to(device)
                target = target.to(device).long()
                # Track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(source)
                    loss = criterion(outputs, target)
                    # Backward + optimize only if in training phase
                    if phase == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item()

            # Track losses
            epoch_loss = running_loss / dataset_sizes[phase]
            if phase == "train":
                trainLoss = epoch_loss
            elif phase == "val":
                valLoss = epoch_loss
                # If improved save model and log update
                # Improvement has to be at least by 0.1%
                if epoch_loss < best_loss * 0.999:
                    best_loss = epoch_loss
                    best_model_state_dict = model.state_dict()
                    numberEpochsWtNoImprovement = 0
                else:
                    numberEpochsWtNoImprovement += 1

        epochEnd = time.time()
        epochTime = epochEnd - epochStart

        # Write metrics to file at end of each epoch
        print(f"{epoch} {trainLoss:.4f} {valLoss:.4f} {best_loss:.4f} {epochTime:.4f}")
    model.load_state_dict(best_model_state_dict)

def predict(model, dataset):
    model.eval()
    preds = []
    # Iterate over data and make prediction
    with torch.no_grad():
        for data, target in dataset:
            output = model(data)
            _, pred_batch = torch.max(output, dim=1)
            preds.append(pred_batch)
    # Turns list of batch predictions to 1D numpy array
    # This conforms with the other numpy implemented models
    return torch.flatten(torch.cat(preds)).detach().numpy()
