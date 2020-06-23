#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 17:04:49 2019

@author: Zack
"""

import numpy as np
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point
import math as math
import matplotlib.pyplot as plt
from IPython.display import clear_output
from time import sleep
from matplotlib.path import Path



def make_polygon(n, numEdge, r, rot, xoff, yoff, intensity, distortion):
    # Should accept either a scalar or a tuple for all parameters. If no argument, randomly initialize
    # If tuple, then create sequence to produce a 28x28xsequence_length tensor

    #Initialize grid space
    x, y = np.meshgrid(np.arange(n), np.arange(n))
    x, y = x.flatten(), y.flatten()
    grid_points = np.vstack((x,y)).T

    # Initialize array for vertices
    points = []    
    
    # Get vertices of polygon    
    for index in np.arange(numEdge) + distortion:
        points.append([xoff + r*math.cos(((rot+index/numEdge)*2*math.pi)), yoff + r*math.sin(((rot+index/numEdge)*2*math.pi))])
    #polygon = Polygon(points) 

    # Crease binary mask for polygon by checking if it contains each grid point
    #mask = intensity*np.reshape([int(polygon.contains(Point(i))) for i in grid_points], (n,n))
    
    p = Path(points) # make a polygon
    grid = p.contains_points(grid_points)
    mask = intensity*(grid.reshape(n,n))

    return mask

def make_polygon_sequence(n, numEdge, r, rot, xoff, yoff, intensity, sequence_length, distort):
   # returns sequence of polygons in one 3D matrix. Can vary along any specified dimension.
   # r, rot, xoff, yoff, intensity should be inputted as 2-value array. If same, will not vary.

    sequence = np.empty([n,n,sequence_length])
    distortion = distort*np.random.standard_normal(numEdge)

    r = np.linspace(r[0], r[1], num=sequence_length)
    rot = np.linspace(rot[0], rot[1], num=sequence_length)
    xoff = np.linspace(xoff[0], xoff[1], num=sequence_length)
    yoff = np.linspace(yoff[0], yoff[1], num=sequence_length)
    intensity = np.linspace(intensity[0], intensity[1], num=sequence_length)
    
    for i in range(sequence_length):

        sequence[:,:,i] = make_polygon(n, numEdge, r[i], rot[i], xoff[i], yoff[i], intensity[i], distortion)

    return sequence



def display_sequence(sequence):
# Produces animated video of a given sequence
# Ideally should display train and test frames in different colour

    plt.figure()
    length = np.shape(sequence)[2]
    
    for i in range(length):
        plt.matshow(sequence[:,:,i],vmin=0, vmax=10, cmap = 'viridis')
        plt.show()
        sleep(0.0001)
        clear_output(wait=True)

def make_polygon_batch(num_ex, n, sequence_length, numEdge_vec, distort_vec, r1_vec, r2_vec, rot1_vec, rot2_vec, xoff1_vec, xoff2_vec, yoff1_vec, yoff2_vec, intensity1_vec, intensity2_vec):
# Creates a training set of sequences

    train_set = np.empty([num_ex,n,n,sequence_length])

    for i in range(num_ex):

        numEdge = numEdge_vec[i]
        distort = distort_vec[i]
        r = np.array([r1_vec[i],r2_vec[i]])
        rot = np.array([rot1_vec[i],rot2_vec[i]])
        xoff = np.array([xoff1_vec[i],xoff2_vec[i]])
        yoff = np.array([yoff1_vec[i],yoff2_vec[i]])
        intensity = np.array([intensity1_vec[i],intensity2_vec[i]])

        train_set[i,:,:,:] = make_polygon_sequence(n, numEdge, r, rot, xoff, yoff, intensity, sequence_length, distort)
        
    return train_set
