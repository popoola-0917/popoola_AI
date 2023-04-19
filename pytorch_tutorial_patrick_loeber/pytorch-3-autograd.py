import torch

x = torch.randn(3) #creating a tensor with 3 random values
print(x)

# Later when we want to calculate the gradient wrt to "x"
x = torch.randn(3, requires_grad=True) 
print(x)



# let's say
y = x + 2;
print(y)
z = y*y*2
z = z.mean()
print(z)
# to calculate our gradient
z.backward() #dz/dx
print(x.grad)

v = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float32)
z.backward() #dz/dx
print(x.grad)










# HOW TO PREVENT PYTORCH FROM TRACKING THE HISTORY
# In a situation when we want to update our weights 
# then we want some previous weights to be left out of our computation
import torch
x = torch.randn(3, requires_grad=True) 
print(x)

# different approaches to prevent pytorch from tracking the history of our code
x.requires_grad_(False) #first method
x.detach() #second method
with torch.no_grad() #third method

# for example

#1
x = torch.randn(3, requires_grad=True) 
x.requires_grad_(False)
print(x)


#2
x = torch.randn(3, requires_grad=True) 
y= x.detach()
print(x)

#3
x = torch.randn(3, requires_grad=True) 
with torch.no_grad():
    y = x + 2;
    print(x)






# TRAINING EXAMPLE
import torch
weights = torch.ones(4, requires_grad=True) #tensor filled with 1's on 4 rows.

for epoch in range(1):
    model_output = (weights*3).sum()
    model_output.backward()
    print(weights.grad)

    # to perform next iteration we must empty the gradients
    # because we are updating our weights.
    weights.grad.zero_()






# OPTIMIZATION
import torch
weights = torch.ones(4, requires_grad=True)

optimizer = torch.optim.SGD(weights, lr=0.01)
optimizer.step()
optimizer.zero_grad()



# THINGS TO NOTE IS THAT BEFORE OUR NEXT ITERATION WE MUST EMPTY THE GRADIENTS
import torch
weights = torch.ones(4, requires_grad=True)
z.backward()
weights.grad.zero_()




