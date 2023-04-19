# Three steps
# 1 Forward pass: Compute Loss
# 2. Compute local gradients
# 3. Backward pass: Compute dLoss/ dWeights using the Chain Rule

import torch

x = torch.tensor(1.0)
y = torch.tensor(2.0)

w = torch.tensor(1.0, requires_grad=True)

#forward pass and compute the loss
y_hat = w * x
loss = (y_hat - y) ** 2

print(loss)

# backward pass
# pytorch will automatically compute backward pass
loss.backward()
print(w.grad) # the first gradient in this code


# update weights
# next forward and backwards pass