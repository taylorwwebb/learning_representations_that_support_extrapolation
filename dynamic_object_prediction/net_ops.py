#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 14:20:07 2019

@author: Zack
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    # Encoder CNN
    # input shape with 1 channel = (batch, 1, 64, 64), pixel values 0-1
    # 3 convolutional layers, then two FC relu linear layers
    # then FC linear compression layer to 10-vector
    
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)
        self.layer2 = nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.layer3 = nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.fc1 = nn.Linear(8 * 8 * 32, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = x.view(-1, 8 * 8 * 32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
    

class DeConvNet(nn.Module):
    # Transposed conv-net - takes (batch,32,8,8) as input
    # 3 deconv layers, first two maintain 32 channels
    # last layer outputs (batch, 1, 64, 64) as original input
    
    def __init__(self):
        super(DeConvNet, self).__init__()
        self.layer1 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.layer2 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer1(x))
        x = torch.sigmoid(self.layer2(x)) # sigmoid puts pixels back to 0-1 range

        
        return x



class Combine(nn.Module):
    # INPUT:  (batch, timestep, 1, 64, 64)
    # Stacks encoder -> LSTM -> deconv 
    # LSTM takes 10-vector as input, has hidden size 256
    # outputs to two linear layers, 256 then 2048 neurons
    # Then reshaped before passed to deconv to produce image
    # Overall output is same format as input - (batch, timestep, 1, 64, 64)
    
    def __init__(self):
        super(Combine, self).__init__()
        self.cnn = ConvNet()
        self.rnn = nn.LSTM(
            input_size=10, 
            hidden_size=256, 
            num_layers=1,
            batch_first=True)
        self.fc1 = nn.Linear(256,256)
        self.fc2 = nn.Linear(256,2048)
        self.deconv = DeConvNet()

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        # combines time and batch dimensions in order to feed whole batch into encoder
        x = x.reshape(batch_size * timesteps, C, H, W) 
        x = self.cnn(x)
        # separates batch and time dimensions again to feed into LSTM
        x = x.view(batch_size, timesteps, -1)
        x, (h_n, h_c) = self.rnn(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.view(-1,32,8,8) # effectively combines batch/time dims again
        x = self.deconv(x)
        # separates batch and time dimensions for final output
        x = x.view(batch_size, timesteps, C, H, W)
        
        return x

