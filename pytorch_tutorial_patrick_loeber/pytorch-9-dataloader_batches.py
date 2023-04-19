# training loop

# for epoch in range(1000):
#     #loop over all batches
#     for i in range(total_batches):
#         x_batch, y_batch = ...



# Use Dataset and dataloader to load wine.csv


# Important terms
# epoch = 1 forward and backward pass of ALL training samples
# batch_size = number of training samples in one forward and backward pass

# number of iterations = number of passes, each pass using [batch_size] number of samples

# e.g. 100 samples, batch_size = 20 --> 100/20 = 5 iterations for 1 epoch


import torch
import torchvision
from torch.utils.data import Dataset, Dataloader
import numpy as np
import math

class WineDataset(Dataset):

    def __init__(self):
        # data loading
        xy = np.loadtxt('./wine.csv', delimiter="," dtype=npfloat32, skiprows=1)
        self.x = xy[:, 1:]
        self.y = xy[:, [0]] #n_samples, 1
        self.n_samples = xy.shape[0]


    def __getitem__(self, index):
        return self.x[index], self.y[index]
        #dataset[0]

    def __len__(self):
        return self.n_samples
        #len(dataset)

dataset = WineDataset()
# first_data = dataset[0]
# features, labels = first_data
# print(features, labels)
dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=2)



# dataiter = iter(dataloader)
# data = dataiter.next()
# features, labels = data
# print(features, labels)


# training loop
num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/4)
print(total_samples, n_iterations)

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        # forward, backward, update
        if (i+1) % 5 == 0'
            print(f'epoch{epoch+1}/{num_epochs}, step{i+1}/{n_iterations}, inputs {inputs.shape}')

torchvision.datasets.MNIST()
# fashion-mnist, cifar, coco 


