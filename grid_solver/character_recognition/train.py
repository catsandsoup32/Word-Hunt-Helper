import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
from model import SmallCNN, get_transform
import numpy

model = SmallCNN()
dummy_input = torch.zeros((1, 1, 32, 32))
output = model(dummy_input)

# Use imageFolder 

num_epochs = 5
for epoch in range(1, num_epochs):
    model.train()
    
    

