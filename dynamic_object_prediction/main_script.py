#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math as math
import matplotlib.pyplot as plt
from matplotlib.path import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os


def make_polygon(n, numEdge, r, rot, xoff, yoff, intensity, distortion, r_mod):
    # Makes single polygon 
    
    #Initialize grid space
    x, y = np.meshgrid(np.arange(n), np.arange(n))
    x, y = x.flatten(), y.flatten()
    grid_points = np.vstack((x,y)).T

    # Initialize array for vertices
    points = []
    current_edge = 0 # parameter rmod re-scales the radius of every vertex to increase irregularity    
    
    # Get vertices of polygon    
    for vertex in np.arange(numEdge) + distortion:

        points.append([xoff + r_mod[current_edge]*r*math.cos(((rot+vertex/numEdge)*2*math.pi)), yoff + r_mod[current_edge]*r*math.sin(((rot+vertex/numEdge)*2*math.pi))])
        current_edge += 1

    # Create binary mask for polygon by checking if it contains each grid point  
    p = Path(points)
    grid = p.contains_points(grid_points)
    mask = intensity*(grid.reshape(n,n))
    #mask[mask==0] = np.random.rand(1)

    return mask

def make_polygon_sequence(n, numEdge, line, r, rot, xoff, yoff, intensity, sequence_length, distort, outline, noise, rmod):
   # returns sequence of polygons in one 3D matrix. Can vary along any specified dimension.
   # r, rot, xoff, yoff, intensity should be inputted as 2-value array. If same, will not vary.

    sequence = np.empty([sequence_length,n,n])
    distortion = np.random.uniform(-distort,distort,numEdge)
    gaussian_noise = noise*np.random.randn(sequence_length,n,n)
    r_mod = np.random.uniform(rmod[0],rmod[1],numEdge)


    r = np.linspace(r[0], r[1], num=sequence_length)
    rot = np.linspace(rot[0], rot[1], num=sequence_length)
    xoff = np.linspace(xoff[0], xoff[1], num=sequence_length)
    yoff = np.linspace(yoff[0], yoff[1], num=sequence_length)
    intensity = np.linspace(intensity[0], intensity[1], num=sequence_length)
    
    for i in range(sequence_length):

        if outline == True:
          outer = make_polygon(n, numEdge, r[i], rot[i], xoff[i], yoff[i], intensity[i], distortion, r_mod)
          inner = make_polygon(n, numEdge, line*r[i], rot[i], xoff[i], yoff[i], intensity[i], distortion, r_mod)
          sequence[i,:,:] = outer - inner
        else:
          sequence[i,:,:] = make_polygon(n, numEdge, r[i], rot[i], xoff[i], yoff[i], intensity[i], distortion, r_mod)

    seq = sequence + gaussian_noise
    normed = (seq - np.min(seq))/(np.max(seq)-np.min(seq))
    return normed

def make_polygon_batch(num_ex, n, sequence_length, numEdge_vec, distort_vec, outline_vec, r1_vec, r2_vec, rot1_vec, rot2_vec, xoff1_vec, xoff2_vec, yoff1_vec, yoff2_vec, intensity1_vec, intensity2_vec, outline, noise_vec, rmod):
# Creates a training set of sequences

    train_set = np.empty([num_ex,sequence_length,n,n])

    for i in range(num_ex):

        noise = noise_vec[i]
        numEdge = numEdge_vec[i]
        distort = distort_vec[i]
        line = outline_vec[i]
        r = np.array([r1_vec[i],r2_vec[i]])
        rot = np.array([rot1_vec[i],rot2_vec[i]])
        xoff = np.array([xoff1_vec[i],xoff2_vec[i]])
        yoff = np.array([yoff1_vec[i],yoff2_vec[i]])
        intensity = np.array([intensity1_vec[i],intensity2_vec[i]])

        train_set[i,:,:,:] = make_polygon_sequence(n, numEdge, line, r, rot, xoff, yoff, intensity, sequence_length, distort, outline, noise, rmod)
        
    return train_set


def context_normalize_full(x):
    # inputs: batch of sequences in low-dimensional embedding space
    # outputs: normalized vector, along with mean and sd used for normalization
    #for i in range(x.size()):
    mean = x.mean(axis = 1,keepdim=True)
    var = x.var(axis = 1,keepdim=True) + 10**-15
    std = torch.sqrt(var)
    zscore = torch.div((x - mean),std)

    return zscore, mean, std

def batch_norm(x):
    # inputs: batch of sequences in low-dimensional embedding space
    # outputs: normalized vector, along with mean and sd used for normalization
    #for i in range(x.size()):
    mean = x.mean(axis = 0,keepdim=True)
    var = x.var(axis = 0,keepdim=True) + 10**-15
    std = torch.sqrt(var)
    zscore = torch.div((x - mean),std)

    return zscore, mean, std


class decoder(nn.Module):
    # Transposed conv-net - takes (batch,bottle) as input
    # Then two fc layers with 256 units each, then fc with 2048 units
    # then reshape to (batch,32,8,8)
    # 3 deconv layers, first two maintain 32 channels
    # last layer outputs (batch, 1, 64, 64) as original input
    
    def __init__(self, bottle):
        super(decoder, self).__init__()

        self.fc1 = nn.Linear(bottle,256)
        self.fc2 = nn.Linear(256,256)
        self.fc3 = nn.Linear(256,2048)
        self.trans1 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.trans2 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.trans3 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x)) # 2048-d FC layer
        x = x.view(-1,32,8,8) # reshape to 32 channel 2d array before passing to deconv
        x = F.relu(self.trans1(x))
        x = F.relu(self.trans2(x))
        x = torch.sigmoid(self.trans3(x)) # sigmoid puts pixels back to 0-1 range
        
        return x

class encoder(nn.Module):
    # Encoder CNN
    # input shape with 1 channel = (batch, 1, 64, 64), pixel values 0-1
    # 3 convolutional layers, then two FC relu linear layers
    # then FC linear compression layer to 10-vector
    
    def __init__(self, bottle):
        super(encoder, self).__init__()
        self.layer1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)
        self.layer2 = nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.layer3 = nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.fc1 = nn.Linear(8 * 8 * 32, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, bottle)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = x.view(-1, 8 * 8 * 32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x


class auto_encoder(nn.Module):
  # simple autoencoder - consists of CNN -> bottleneck linear -> deCNN
  def __init__(self, bottle):
        super(auto_encoder, self).__init__()

        self.encode = encoder(bottle)
        self.decode = decoder(bottle)

  def forward(self, x):
    x = self.encode(x)
    x = self.decode(x)

    return x

class norm_lstm(nn.Module):
  def __init__(self, bottle, hidden_dim, n_layers):
        super(norm_lstm, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

         # LSTM Layer
        self.lstm = nn.LSTM(bottle, hidden_dim, n_layers, batch_first=True)   
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, bottle)

        #scale and shift parameters
        self.scale = nn.Parameter(torch.ones(bottle))
        self.shift = nn.Parameter(torch.zeros(bottle))

  def forward(self, x):

    x = torch.mul(x,self.scale) + self.shift
    x, (h_n, h_c) = self.lstm(x)
    x = self.fc(x)

    return x


def generator(num_ex = 32, n = 64, sequence_length = 20, display = 0, outline = False, noise = 0, rmod = [1,1], rot = [1/8,1/8], num_edge = [4,5], distort = 0, r=[2,8]):
# Creates a training set of sequences


    noise_vec = np.random.uniform(0,noise,num_ex) # adds varying levels of noise to sequences
    numEdge_vec = np.random.randint(num_edge[0],num_edge[1],num_ex)   # number of edges in polygon
    distort_vec = np.random.uniform(distort,distort,num_ex)  # how much distortion
    outline_vec = np.random.uniform(0,0.8,num_ex)  # 0 = filled in, 
    r1_vec, r2_vec = np.random.uniform(r[0],r[1],num_ex), np.random.uniform(r[0],r[1],num_ex)
    rot1_vec, rot2_vec = np.random.uniform(rot[0],rot[1],num_ex), np.random.uniform(rot[0],rot[1],num_ex)
    xoff1_vec, xoff2_vec = np.random.uniform(n/4,3*n/4,num_ex), np.random.uniform(n/4,3*n/4,num_ex)
    yoff1_vec, yoff2_vec = np.random.uniform(n/4,3*n/4,num_ex), np.random.uniform(n/4,3*n/4,num_ex)
    intensity1_vec, intensity2_vec = np.random.uniform(1,1,num_ex), np.random.uniform(1,1,num_ex)
    
    train = make_polygon_batch(num_ex, n, sequence_length, numEdge_vec, distort_vec, outline_vec, r1_vec, r2_vec, rot1_vec, rot2_vec, xoff1_vec, xoff2_vec, yoff1_vec, yoff2_vec, intensity1_vec, intensity2_vec, outline, noise_vec, rmod)

    return train

def create_test_set(r,name):
    train = generator(num_ex = 500, sequence_length = 20, r = r)
    test_path = os.getcwd() + name
    np.save(test_path,train)


def train_autoencoder(epochs = 50000):
    

    batch_size = 32  # number of sequences to generate in each training batch
    seq_length = 2  # how many frames in polygon sequence
    learning_rate = 0.0005 
    bottleneck = 10 # bottleneck in CNN encoder
    test_check, sample = 0, 10  # change to 1 if you want an example prediction from the training batch (choose sample from 0 to 31)
    losses = []
    outline = False # true for outlined shapes, false for filled shapes
    rot = [1/8,1/8] #keep rotation constant for squares
    r = [2,31] #radius range

    path = os.getcwd() + '/Autoencode_squares'
    net = auto_encoder(bottleneck)  # encoder - LSTM - decoder
    #net.load_state_dict(torch.load(path))
    #losses = torch.load(path+'losses')
    net.cuda()


    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        
            # This generates a batch of polygon sequences, new training set is generated on the fly for each batch
            train_batch = torch.tensor(generator(num_ex = batch_size, n = 64, sequence_length = seq_length, display = 0, outline = False, noise = 0, rmod = [1,1], rot = rot, num_edge = [4,5], distort = 0, r = r))

            
            # converts type to float
            a = train_batch.float()
        
            tn = Variable(a[:,0:1,:,:])       # Image to be reconstructed
            tn = tn.cuda()
            
            optimizer.zero_grad()                # Intialize the hidden weight to all zeros
            out = net(tn)                    # Forward pass: compute predictsion at t+1
            loss = criterion(out, tn)     # Compute loss (MSE)
            loss.backward()                      # Backward pass
            optimizer.step()                     # update the weights of hidden nodes

             # print and save loss
            if epoch % 100 == 0:
              print(epoch, 'loss:', loss.item())
              losses.append([epoch,loss.item()])
              torch.save(losses,path+'_losses')

             # saves and loads network every 1000 epochs
            if epoch % 10000 == 0:
              torch.save(net.state_dict(), path )
              net.load_state_dict(torch.load(path ))
              net.cuda()

def train_rnn(epochs = 50000):
    
    for array_id in range(9):
        
        # Creates 9 models with normalization as in 'norm_array'
        norm_array = ['context','context','context','batch','batch','batch',None,None,None]
        
        batch_size = 32  # number of sequences to generate in each training batch
        seq_length = 20  # how many frames in polygon sequence
        learning_rate = 0.0005 
        bottleneck = 10 # bottleneck in CNN encoder
        test_check, sample = 0, 10  # change to 1 if you want an example prediction from the training batch (choose sample from 0 to 31)
        losses = [] # train embedding losses
        te_losses = [] # test embedding losses
        tp_losses = [] # test pixel losses
        train_p_losses = [] #train pixel losses
        outline = False # true for outlined shapes, false for filled shapes
        normalization = norm_array[array_id] # if None will not normalize
        hidden = 20 # number of hidden units in LSTM
        r = [2,8] #radius range


        path = os.getcwd() + f'/model_{array_id}'
        
        auto_net = auto_encoder(bottleneck)  # encoder - decoder
        path_a = os.getcwd() + '/Autoencode_squares'
        auto_net.load_state_dict(torch.load(path_a))
        auto_net.cuda()

        # loads testing set and prepares input and output sequences
        test_path = os.getcwd() + '/square_test_set_20.npy'
        test_set = torch.tensor(np.load(test_path)).unsqueeze(2).float()
        seq_out_pixel = test_set[:,1:seq_length,:,:,:].cuda()

        a = test_set.view(-1,1,64,64).cuda()
        test_embedding = auto_net.encode(a)
        test_embedding = test_embedding.view(500,seq_length,bottleneck)

        # if performing context or batch normalization
        if normalization == 'context':
          test_embedding, test_mean, test_std = context_normalize_full(test_embedding)
        if normalization == 'batch':
          test_embedding, test_mean, test_std = batch_norm(test_embedding)

        seq_in = Variable(test_embedding[:,0:seq_length-1,:]).cuda()      # Input sequence
        seq_out = Variable(test_embedding[:,1:seq_length,:]).cuda()     # Sequence offset at t = t+1

        # Creates RNN architecture and sets path
        rnn_net = norm_lstm(bottleneck,hidden,1)  
        #rnn_net.load_state_dict(torch.load(path))
        #losses = torch.load(path+'losses')
        rnn_net.cuda()

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(rnn_net.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            
                # This generates a batch of polygon sequences, new training set is generated on the fly for each batch
                train_batch = torch.tensor(generator(num_ex = batch_size, n = 64, sequence_length = seq_length, display = 0, outline = False, r=r))
                
                # converts type to float
                a = train_batch.float()

                # Makes target array in pixel space (frames 1-20)
                pixel_target = a[:,1:seq_length,:,:]
                pixel_target = pixel_target.view(batch_size,seq_length-1,1,64,64).float().cuda()

                # combines batch and time dimensions for encoder
                a = a.view(-1,1,64,64).cuda()


                # gets embeddings from encoder, separates batch/time, context normalizes
                with torch.no_grad():
                  embedding = auto_net.encode(a)
                  embedding = embedding.view(batch_size,seq_length,bottleneck)

                  # normalizes embeddings according to context or batch
                  if normalization == 'context':
                    embedding, mean, std = context_normalize_full(embedding)
                  if normalization == 'batch':
                    embedding, mean, std = batch_norm(embedding)

                  #Gets predicted embeddings, and calculates loss against target embeddings
                  out_test = rnn_net(seq_in)
                  test_loss_embedding = criterion(out_test,seq_out)

                  # if doing normalization, unnormalizes predicted test embeddings according to input norms
                  if normalization == 'context':
                    out_test = torch.mul(out_test,test_std) + test_mean
                  if normalization == 'batch':
                    out_test = torch.mul(out_test,test_std[:,0:seq_length-1,:]) + test_mean[:,0:seq_length-1,:]

                  out_test_pixel = auto_net.decode(out_test)
                  out_test_pixel = out_test_pixel.view(500,seq_length-1,1,64,64)
                  test_loss_pixel = criterion(out_test_pixel,seq_out_pixel)


                tn = Variable(embedding[:,0:seq_length-1,:])       # Input sequence
                tn_1 = Variable(embedding[:,1:seq_length,:])     # Sequence offset at t = t+1
                tn = tn.cuda()
                tn_1 = tn_1.cuda()
                
                optimizer.zero_grad()           # Intialize the hidden weight to all zeros
                out = rnn_net(tn)               # Forward pass: compute predictsion at t+1
                loss = criterion(out, tn_1)     # Compute train loss embedding space (MSE)
                loss.backward()                 # Backward pass
                optimizer.step()                # update the weights of hidden nodes

                with torch.no_grad():
                  # if doing normalization, unnormalizes predicted train embeddings
                  if normalization == 'context':
                    out = torch.mul(out,std) + mean
                  if normalization == 'batch':
                    out = torch.mul(out,std[:,1:seq_length,:]) + mean[:,1:seq_length,:]

                  #train loss in pixel space
                  out_batch_pixel = auto_net.decode(out)
                  out_batch_pixel = out_batch_pixel.view(batch_size,seq_length-1,1,64,64)
                  train_loss_pixel = criterion(out_batch_pixel,pixel_target)

                 # print and save loss
                if epoch % 100 == 0:
                  print(epoch, 'Train loss e:', loss.item())
                  losses.append([epoch,loss.item()])
                  torch.save(losses,path+'_train_e_losses')

                  print(epoch, 'Train loss p:', train_loss_pixel.item())
                  train_p_losses.append([epoch,train_loss_pixel.item()])
                  torch.save(train_p_losses,path+'_train_p_losses')

                  print(epoch, 'Test loss e:', test_loss_embedding.item())
                  te_losses.append([epoch,test_loss_embedding.item()])
                  torch.save(te_losses,path+'_te_losses')
                  
                  print(epoch, 'Test loss p:', test_loss_pixel.item())
                  tp_losses.append([epoch,test_loss_pixel.item()])
                  torch.save(tp_losses,path+'_tp_losses')

                 # saves and loads network every 1000 epochs
                if epoch % 10000 == 0:
                  torch.save(rnn_net.state_dict(), path )
                  rnn_net.load_state_dict(torch.load(path ))
                  rnn_net.cuda()


def get_pixel_loss():
    ### GETS PIXEL LOSS FOR ALL MODELS ####

    #hyperparameters
    bottleneck = 10
    hidden = 20
    seq_length = 20
    test_loss_pixel = np.zeros((3,15))
    test_batch = 32

    #paths
    path_a = os.getcwd() + '/Autoencode_squares'
    test_path_1 = os.getcwd() + '/square_test_set_20.npy'
    test_path_2 = os.getcwd() + '/square_test_set_20_2.npy' #second test set, uncomment to use
    test_path_train = os.getcwd() + '/square_test_set_20_train.npy'
    norm = ['context','context','context','batch','batch','batch',None,None,None]

    #loading
    #load autoencoder
    auto_net = auto_encoder(bottleneck)  # encoder - LSTM - decoder
    auto_net.load_state_dict(torch.load(path_a,map_location=torch.device('cpu')))

    #load rnn
    rnn_net = norm_lstm(bottleneck,hidden,1)

    # batch norm stuff (load train-distribution test set)
    test_set_train = torch.tensor(np.load(test_path_train)).unsqueeze(2).float()
    minibatch = np.random.choice(500, (15,32), replace=False) #for batch norm

    # MSE loss, sets up dict to store results
    criterion = nn.MSELoss()
    test_loss_pixel_dict = {'context': [], 'batch_test': [], 'batch_train': [], 'none': []}

    #loops through 9 models with normalization: context 0-2, batch 3-5, None 6-8

    for test_path in [test_path_1, test_path_2]:
        test_set = torch.tensor(np.load(test_path)).unsqueeze(2).float()

        for i in range(9):
            normalization = norm[i]
            path = os.getcwd() + f'/model_{i}'
            rnn_net.load_state_dict(torch.load(path,map_location=torch.device('cpu')))

            seq_out_pixel = test_set[:,0:seq_length-1,:,:,:]
            a = test_set.view(-1,1,64,64)
            test_embedding = auto_net.encode(a)
            test_embedding = test_embedding.view(500,seq_length,bottleneck)

            if i >=0 and i<3:
                test_embedding, test_mean, test_std = context_normalize_full(test_embedding)
                seq_in = Variable(test_embedding[:,0:seq_length-1,:])      # Input sequence
                seq_out = Variable(test_embedding[:,1:seq_length,:])   # Sequence offset at t = t+1

                with torch.no_grad():

                    out_test = rnn_net(seq_in)
                    out_test = torch.mul(out_test,test_std) + test_mean
                    out_test_pixel = auto_net.decode(out_test)
                    out_test_pixel = out_test_pixel.view(500,seq_length-1,1,64,64)
                    test_loss_pixel_dict['context'].append(criterion(out_test_pixel,seq_out_pixel).item())

            if i >=3 and i<6:

                # using test statistics in batches of 32
                pixel_loss = 0
                for iteration in range(15):
                    test_set_min = test_set[minibatch[iteration],:,:,:]
                    seq_out_pixel_min = test_set_min[:,1:seq_length,:,:,:]
                    a_min = test_set_min.view(-1,1,64,64)
                    test_embedding_min = auto_net.encode(a_min)
                    test_embedding_min = test_embedding_min.view(test_batch,seq_length,bottleneck)
                    test_embedding_min, test_mean, test_std = batch_norm(test_embedding_min)

                    seq_in = Variable(test_embedding_min[:,0:seq_length-1,:])      # Input sequence
                    seq_out = Variable(test_embedding_min[:,1:seq_length,:])   # Sequence offset at t = t+1

                    with torch.no_grad():

                        out_test = rnn_net(seq_in)
                        out_test = torch.mul(out_test,test_std[:,0:seq_length-1,:]) + test_mean[:,0:seq_length-1,:]
                        out_test_pixel = auto_net.decode(out_test)
                        out_test_pixel = out_test_pixel.view(test_batch,seq_length-1,1,64,64)
                        pixel_loss += criterion(out_test_pixel,seq_out_pixel_min).item()/15
                        
                test_loss_pixel_dict['batch_test'].append(pixel_loss)

                # using training-domain statistics from batch of 500
                a_train = test_set_train.view(-1,1,64,64)
                test_train_embedding = auto_net.encode(a_train)
                test_train_embedding = test_train_embedding.view(500,seq_length,bottleneck)
                _, train_mean, train_std = batch_norm(test_train_embedding)
                test_embedding = torch.div((test_embedding - train_mean),train_std)

                seq_in = Variable(test_embedding[:,0:seq_length-1,:])      # Input sequence
                seq_out = Variable(test_embedding[:,1:seq_length,:])   # Sequence offset at t = t+1
         
                with torch.no_grad():
                    out_test = rnn_net(seq_in)
                    out_test = torch.mul(out_test,train_std[:,0:seq_length-1,:]) + train_mean[:,0:seq_length-1,:]
                    out_test_pixel = auto_net.decode(out_test)
                    out_test_pixel = out_test_pixel.view(500,seq_length-1,1,64,64)
                    test_loss_pixel_dict['batch_train'].append(criterion(out_test_pixel,seq_out_pixel).item())

            if i >= 6:
                seq_in = Variable(test_embedding[:,0:seq_length-1,:])      # Input sequence
                seq_out = Variable(test_embedding[:,1:seq_length,:])   # Sequence offset at t = t+1

                with torch.no_grad():

                    out_test = rnn_net(seq_in)
                    out_test_pixel = auto_net.decode(out_test)
                    out_test_pixel = out_test_pixel.view(500,seq_length-1,1,64,64)
                    test_loss_pixel_dict['none'].append(criterion(out_test_pixel,seq_out_pixel).item())

    return test_loss_pixel_dict

if __name__ == "__main__":
    create_test_set([8,21],'/square_test_set_20')
    create_test_set([8,21],'/square_test_set_20_2')
    create_test_set([2,8],'/square_test_set_20_train')
    train_autoencoder()
    train_rnn()
    pixel_loss = get_pixel_loss()
    np.save('pixel_loss',pixel_loss)
