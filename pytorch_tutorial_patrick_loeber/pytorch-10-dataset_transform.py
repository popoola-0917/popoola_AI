import torch
import torchvision
from torch.utils.data import Dataset, Dataloader
import numpy as np
import math

class WineDataset(Dataset):

    def __init__(self, transform=None):
        # data loading
        xy = np.loadtxt('./wine.csv', delimiter="," dtype=npfloat32, skiprows=1)

        # note that we do not conver to tensor here
        # we perform slicing here
        self.x = xy[:, 1:]
        self.y = xy[:, [0]] #n_samples, 1
        # self.n_samples = xy.shape[0]

        self.transform = transform


    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        #dataset[0]

        if self.transform:
            sample = self.transform(sample)
        
        return sample



    def __len__(self):
        return self.n_samples
        #len(dataset)


class ToTensor()
    def __call__(self, sample):
        inputs, tragets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)

class MulTransform:
    def __init__(self, factor):
        self.factor = factor
    def __call__(self, sample):
        inputs, target = sample
        inputs *= self.factor
        return inputs, target


dataset = WineDataset(transform=ToTensor())
first_data = dataset[0]
features, labels = first_data
print(features)
print(type(features), type(labels))


composed = torchvision.transforms.Compose([ToTensor(), MulTransform(2)])
datset = WineDataset(transform=composed)
first_data = dataset[0]
features, labels = first_data
print(features)
print(type(features), type(labels))















# first_data = dataset[0]
# # features, labels = first_data
# # print(features, labels)
# dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=2)



# # dataiter = iter(dataloader)
# # data = dataiter.next()
# # features, labels = data
# # print(features, labels)


# # training loop
# num_epochs = 2
# total_samples = len(dataset)
# n_iterations = math.ceil(total_samples/4)
# print(total_samples, n_iterations)

# for epoch in range(num_epochs):
#     for i, (inputs, labels) in enumerate(dataloader):
#         # forward, backward, update
#         if (i+1) % 5 == 0'
#             print(f'epoch{epoch+1}/{num_epochs}, step{i+1}/{n_iterations}, inputs {inputs.shape}')

# torchvision.datasets.MNIST()
# fashion-mnist, cifar, coco 


