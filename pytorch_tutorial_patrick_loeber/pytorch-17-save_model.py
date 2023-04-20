import torch
import torch.nn as nn


# Method 1 to save models

# COMPLETE MODEL
torch.save(model, PATH)
# model class must be defined somewhere
model = torch.load(PATH)
model.eval()



# Method 2 to sav models
# State DICT
torch.save(model.state_dict(), PATH)

# model must be created again with parameters
model = Model (*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()
