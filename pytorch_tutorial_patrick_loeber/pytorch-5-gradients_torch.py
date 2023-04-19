# Design model(input, output size, forward pass)
# Consruct loss and optimizer
# Training loop
#     forward pass: compute prediction
#     backward pass: gradients
#     update weights


import torch

# function = weight * input
# f = w * x   

# f = 2 * x
X = torch.tensor([1,2,3,4], dtype=torch.float32)
Y = torch.tensor([2,4,6,8], dtype=torch.float32)


w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# model prediction
def forward(x):
    return w * x

# loss = MSE
def loss(y, y_predicted):
    return ((y_predicted-y)**2).mean()




print(f'Prediction before training: f(5) = {forward(5):.3f}')



# Training
learning_rate = 0.01
n_iters = 100

# training loop
for epoch in range(in_iters):
    # prediction = forward pass
    y_pred = forward(X)

    # loss
    l = loss(Y, y_pred)

    # gradients = backward pass
    l.backward() #dl/dw
    

    # update weights
    with torch.no_grad():
        w -= learning_rate * w.grad


    # zero gradients
    w.grad.zero_()

    #  print our updates every "10" steps
    if epoch % 10 == 0:
        print(f'epoch {epoch + 1}: w = {w:.3f}, loss = {l:.8f}')


print(f'Prediction after training: f(5) = {forward(5):.3f}') # as our weight increases, our loss decreases which implies that our model is performing well.

