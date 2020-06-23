#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 14:34:56 2019

@author: Zack
"""
# Training network

import torch
import torch.nn as nn
from torch.autograd import Variable
#import numpy as np
import net_ops as nets
import poly_ops as poly
from generator import generator


epochs = 1  # really just number of batches, since training data generated, no real 'epochs'
batch_size = 32  # number of sequences to generate in each training batch
seq_length = 20  # how many frames in polygon sequence
learning_rate = 0.0005 
test_check, sample = 0, 10  # change to 1 if you want an example prediction from the training batch (choose sample from 0 to 31)

net = nets.Combine()  # encoder - LSTM - decoder
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

for epoch in range(epochs):
    
        # This generates a batch of polygon sequences, new training set is generated on the fly for each batch
        train_batch = torch.tensor(generator(num_ex = batch_size, n = 64, sequence_length = seq_length, display = 0))
        
        # Adds channel, converts to float, re-formats to input dimensions
        a = train_batch[:,:,:,:].unsqueeze(1).float().permute(0,4,1,2,3) 
    
        tn = Variable(a[:,0:18,:,:,:])       # Input sequence
        tn_1 = Variable(a[:,1:19,:,:,:])     # Sequence offset at t = t+1
        
        optimizer.zero_grad()                # Intialize the hidden weight to all zeros
        outputs = net(tn)                    # Forward pass: compute predictsion at t+1
        loss = criterion(outputs, tn_1)      # Compute loss (MSE)
        print('loss:', loss.item())
        loss.backward()                      # Backward pass
        optimizer.step()                     # update the weights of hidden nodes
        
 
# test - display input sequence and prediction for a sample training sequence       
if test_check == 1: 
    with torch.no_grad():
        pred = net(a)
        c = pred.permute(3,4,1,0,2)
        b = c[:,:,:,sample,0]
        b = b.detach().numpy()
        
        d = a[sample,:,0,:,:]
        d = d.permute(1,2,0)
        poly.display_sequence(d)
        poly.display_sequence(b)
    