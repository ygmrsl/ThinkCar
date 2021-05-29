# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 17:36:52 2021

@author: ygmr
"""
import os
import pandas as pd
pth = r"E:\GitHub\ThinkCar/"
#%%
file = "ethics.xlsx"
df = pd.read_excel(pth+file)
np = []
for p in df["pth"]:
    temp = os.path.split(p)[1][:-4]+"-mask.png"
    np.append(temp)
df["pth"] = np
#df.to_csv(pth+"ethics.csv", index=False)
#%%
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms,utils
import numpy as np
import matplotlib.pyplot as plt
import cv2
#%%
class EthicalDecision(Dataset):
    def __init__(self, directory, filename, classes, transform=None):
        self.dataset = pd.read_csv(filename)
        self.classes = classes
        self.path = directory
        self.transform = transform
        if self.transform == None:
            self.transform = transforms.Compose([transforms.Resize(256),transforms.ToTensor()])
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        row = self.dataset.loc[index,:]
        image = cv2.imread(self.path+row["pth"])
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        speed = row["speed"]
        label = row["label"]
        pil = transforms.ToPILImage()
        image = pil(image)
        image = self.transform(image)
        speed = torch.as_tensor(speed, dtype=torch.uint8)
        label = torch.as_tensor(label, dtype=torch.long)
        
        return (image, speed), label
#%%
directory = r"E:\carla\CARLA_0.9.10\WindowsNoEditor\PythonAPI\examples\_out/"
filename = pth + "ethics.csv"
dataset = EthicalDecision(directory, filename, None)
#%%
"""
data, target = dataset[0]
print(target, data[1])
im = data[0].permute(1,2,0).cpu().detach().numpy()
cv2.imshow("", im)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""


#%%
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
class EthicsCaseNet(torch.nn.Module):
    def __init__(self):
        super(EthicsCaseNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3,64,3)
        self.conv2 = torch.nn.Conv2d(64,32,3)
        self.conv3 = torch.nn.Conv2d(32,32,3)
        self.pool = torch.nn.MaxPool2d(2)
        self.fc1 = torch.nn.Linear(32*30*55+1, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 2)
        self.dropout = torch.nn.Dropout(0.2)
        
    def forward(self, x1,x2):
        x1 = self.pool(F.relu(self.conv1(x1)))
        x1 = self.pool(F.relu(self.conv2(x1)))
        x1 = self.pool(F.relu(self.conv3(x1)))
        #print(x1.shape)
        x1 = x1.view(-1,32*30*55)
        x2 = torch.unsqueeze(x2, dim=1)
        #print(x1.shape, x2.shape)
        x = torch.cat((x1, x2), dim=1)
        #print(x.shape)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = F.relu(self.fc3(x))
        
        return x
#%%
def train_test_split(dataset,testRatio=0.2,batchSize=16,shuffleDataset=True,seed=10):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(testRatio * dataset_size))
    if shuffleDataset :
        np.random.seed(seed)
        np.random.shuffle(indices)
    #print(indices)
    trainIndices, valIndices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    trainSampler = SubsetRandomSampler(trainIndices)
    validSampler = SubsetRandomSampler(valIndices)

    trainDataLoader = DataLoader(dataset, batch_size=batchSize,
                                               sampler=trainSampler)
    validationDataLOoader = DataLoader(dataset, batch_size=batchSize,
                                                    sampler=validSampler,
                                                    shuffle=False)
    return trainDataLoader , validationDataLOoader
#%%
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = EthicsCaseNet()
model.to(device)
trainLoad, valLoad = train_test_split(dataset, batchSize=16, shuffleDataset=True)
import torch.optim as optim

# specify loss function
criterion = torch.nn.CrossEntropyLoss()

# specify optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

#%%
# number of epochs to train the model
n_epochs = 10 # you may increase this number to train a final model

valid_loss_min = np.Inf # track change in validation loss

for epoch in range(1, n_epochs+1):

    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0
    
    ###################
    # train the model #
    ###################
    model.train()
    for data, target in trainLoad:
        images, speeds = data[0], data[1]
        images = images.to(device)
        speeds = speeds.to(device)
        target = target.to(device)
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(images, speeds)
        # calculate the batch loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss
        train_loss += loss.item()*data[0].size(0)
        

    
    # calculate average losses
    train_loss = train_loss/len(trainLoad.dataset)
    
        
    # print training/validation statistics 
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
    
    # save model if validation loss has decreased
#%%
torch.save(model.state_dict(), pth+"ethic_model.pth")
#%%
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = EthicsCaseNet()
model.load_state_dict(torch.load(pth+"ethic_model.pth"))
trainLoad, valLoad = train_test_split(dataset, batchSize=8, shuffleDataset=True)
criterion = torch.nn.CrossEntropyLoss()
valLoad = DataLoader(dataset, batch_size=8, shuffle=True)
#%%
######################    
# validate the model #
######################
train_on_gpu = torch.cuda.is_available()
classes = ["Ethical", "Not Ethical"]
test_loss = 0.0
class_correct = list(0. for i in range(2))
class_total = list(0. for i in range(2))
model.to(device)
model.eval()
# iterate over test data
with torch.no_grad():
    for data, target in valLoad:
        # move tensors to GPU if CUDA is available
        images, speeds = data[0], data[1]
        images = images.to(device)
        speeds = speeds.to(device)
        target = target.to(device)
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(images, speeds)
        # calculate the batch loss
        loss = criterion(output, target)
        # update test loss 
        test_loss += loss.item()*data[0].size(0)
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        # compare predictions to true label
        correct_tensor = pred.eq(target.data.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
        # calculate test accuracy for each object class
        for i in range(data[0].size(0)): #batch size
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1
    
    test_loss = test_loss/len(valLoad.dataset)
    print('Test Loss: {:.6f}\n'.format(test_loss))
    
    for i in range(2):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                classes[i], 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))
    
    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))




