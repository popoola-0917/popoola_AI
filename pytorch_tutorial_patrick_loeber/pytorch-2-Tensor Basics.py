import torch

# 1 dimenision with three elements
x = torch.empty(3)

# 2 dimenision with three elements
x = torch.empty(2,3)

# 3 dimension
x = torch.empty(2,2,3)

# torch with random values
x = torch.rand(2,2)
print(x)

# fill with zeroes
x = torch.zeroes(2,2)
print(x)

# fill with ones
x = torch.ones(2,2)
print(x)


# specific datatype
x = torch.ones(2,2 dtype=torch.int)
x = torch.ones(2,2 dtype=torch.double)
x = torch.ones(2,2 dtype=torch.float)
x = torch.ones(2,2 dtype=torch.float16)

print(x.dtype)
print(x.size())


# constructing tensor from data
x = torch.tensor([2.5, 0.1])
print(x)


# Tensor Operation
x = torch.rand(2,2)
y = torch.rand(2,2)
print(x)
print(y)

# perform element-wise addition
z = x + y

# same operation as above
z = torch.add(x,y)
print(z)


# An inplance addition
x = torch.rand(2,2)
print(x)
y = torch.rand(2,2)
print(y)

# In pytorch any operation that have trailing underscore will perform inplace operation 
y.add_(x)
print(y)


# Subtraction operation
z = x - y
print(z)
z = torch.sub(x,y)
print(z)


# multiplication operation
z = x*y
print(z)
z = torch.mul(x,y)
print(z)


# division operation
z = x/y
print(z)
z = torch.div(x,y)
print(z)


# slicing operation
x = torch.rand(5,3)
print(x)
print(x[:, 0])
print(x[1, 1])
print(x[1, :])


# Reshaping a tensor
# by calling the view method.
x = torch.rand(4,4)
print(x)
y = x.view(16)
print(y)



x = torch.rand(4,4)
print(x)
y = x.view(-1, 8)
print(y.size())







# Converting from numpy to tensor
import torch
import numpy as np

a = torch.ones(5) #creating a tensor
print(a)

b = a.numpy()
print(b) #however, we should be careful if we are running the code on a cpu, because "a" and "b" will share same memory location.

# for example
a.add_(1)
print(a)
print(b)

# otherwise
a = np.ones(5)
print(a)
b = torch.from_numpy(a)
print(b)

a += 1
print(a)
print(b)

if torch.cuda.is_available():
    device = torch.device("cuda")

    # creating a tensor on the GPU
    x = torch.ones(5, device=device)

    # creating tensor on the cpu
    y = torch.ones(5)
    # then moving the operation to the GPU
    y = y.to(device)
    z = x + y
    # moving the tensor back to cpu
    z = z.to("cpu")






# A lot of times when a tensor is created
# "requires_grad=True" tells the machine to calculate the gradients
# this will be useful for optimization
x = torch.ones(5, requires_grad=True) 
print(x)