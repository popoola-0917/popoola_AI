# MNIST
# DataLoader, Transformation
# Multilayer Neural Net, activation function
# Loss and Optimizer
# Training Loop (batch training)
# Model evaluation
# GPU support

import torch
import torch.nn as nn 
import torchvision 
import torchvision.transforms as transforms 
import sys
import torch.nn.functional as F
import matplotlib.pyplot as plt 


from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/mnist")

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# hyper parameters

# 28*28 = 784
input_size = 784 
hidden_size = 100
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001
# learning_rate = 0.01


#MNIST
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)

test_dataset = torchvision.datasets.MNIST(root='./data', train=False,
transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

examples = iter(train_loader)
samples, labels = examples.next()
print(samples.shape, labels.shape)

for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(samples[i][0],cmap='gray')
# plt.show()

# adding images to our tensorboard
imag_grid = torchvision.utils.make_grid(example_data)
writer.add_image('mnist_images', img_grid)
writer.close()
# sys.exit()

# Setting up a fully connected neural layer to perform classification
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self). __init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.li(x) #output of the first linear layer
        out = self.relu(out) #output of the relu activation layer
        out = self.l2(out) #output of the second linear layer
        return out 

model = NeuralNet(input_size, hidden_size, num_classes)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adm(model.parameters(), lr=learning_rate)

writer.add_graph(mode, example_data.reshape(-1, 28*28))
writer.close()
# sys.exit()

#training loop
n_total_steps = len(train_loader)

running_loss = 0.0
running_correct = 0

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):

        # reshaping our images
        # 100 , 1, 28, 28
        # 100, 784
        images = images.reshape(-1, 28*28.to(device))
        labels = labels.to(device)

        #forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)



        #backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)

        running_correct += (predicted += labels).sum().item()

        # Every 100 steps we print out loss
        if (i+1) % 100 == 0:
            print(f'epoch{epoch+1}/{num_epochs}, step{i+1}/{n_total_steps}, loss ={loss.item():.4f}')
            writer.add_scalar('training loss', running-_loss / 100, epoch = n_total_steps + i)
            writer.add_scalar('accuracy', running_correct/100, epoch * n_total_steps + i)
            running_loss = 0.0
            running_correct = 0
        
    # testing
    # we don't want to compute the gradients for all the step we do


    labels = []
    preds = []
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in test_loader:
            images = images.reshape(-1, 28*28).to(device)
            labels1 = labels1.to(device)
            outputs = model(images)

            #value, index
            _, predicted = torch.max(outputs.data, 1)
            n_samples += labels1.shape[0]
            n_correct = (predicted == labels1).sum().item()

            class_predictions = [F.softmax(output, dim=0) for output in outputs]

            preds.append(class_predictions)
            labels1.append(predicted)
        
        preds = torch.cat([torch.stack(batch) for batch in preds])
        labels = torch.cat(labels)

        # calculating the total accuracy
        acc = 100.0 * n_correct / n_samples
        print(f'accuracy ={acc}')

        classes = range(10)
        for i in classes:
            labels_i = labels == i
            preds_i = preds[:, i]
            writer.add_pr_curve(str(i), labels_i, preds_i, globa_step=0)
            wtier.close()
        



