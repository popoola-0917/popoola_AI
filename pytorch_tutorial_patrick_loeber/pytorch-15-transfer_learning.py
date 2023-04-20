# Transfer learning is a machine learning technique where a model developed for a particular task is been re-used as a starting point of a particular task.
# For Example we can train a model to classify "birds" and "cat" and then use thesame model, modified a little bit at the last layer  to classify "bees" and "dogs"
# This is a good because training a new model from scratch can be time-consuming.



# In this example we are using pretraind "RESNET-CNN"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt 
import time 
import os 
import copy 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        tranforms.ToTensor(),
        tranforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        transforms.Resize(256)
        transforms.CenterCrop(224)
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

# import data
# we are using inbuilt datasets
data_dir = 'data/hymenoptera_data'
sets = ['train', 'val']
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                        data_transforms[x])}
                for x in ['train', 'val']
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=0)}
                for x in ['train', 'val']

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes 
print(class_names)

def train_model(model, criterion, optimizer, scheduler, num_epochs=2):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs-1}')
        print('-'* 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train() #set model to training mode

            else:
                model.eval() # set model to evaluate mode
            
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)


                #forward
                #track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                
                #statistics
                running_loss += loss.item() * inputs*size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epochloss:.4f} Acc: {epoch acc: .4f}')


            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        
        print()
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model



# Using transfer learning
# we will be using a technique called 'fine-tuning'
model = model.resnet18(pretrained=True)
num_ftrs = model.fc.in_features

model.fc = nn.Linear(num_ftrs, 2)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# scheduler
step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1) #every 7 epochs our learning rate will by multiplied by gamma

# for epoch in range(100):
#     train() #optimizer.step()
#     evaluate()
#     scheduler.step()

model = train_model(model, criterion, optimizer, scheduler, num_epochs=20)




#####################################################################
# Another option is to freeze the layers at the beginning of the model
# and only train the last layers
model = model.resnet18(pretrained=True)

# this will freeze all the layers in the beginning
for param in model.parameters():
    param.requires_grad = False


num_ftrs = model.fc.in_features

# setting up a new last layer
model.fc = nn.Linear(num_ftrs, 2)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# scheduler
step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1) #every 7 epochs our learning rate will by multiplied by gamma

# for epoch in range(100):
#     train() #optimizer.step()
#     evaluate()
#     scheduler.step()

model = train_model(model, criterion, optimizer, scheduler, num_epochs=20)





