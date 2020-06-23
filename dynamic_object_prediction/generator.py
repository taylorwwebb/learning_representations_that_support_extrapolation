#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 17:03:20 2019

@author: Zack
"""
import poly_ops as poly
import numpy as np

def generator(num_ex = 10, n = 64, sequence_length = 20, display = 0):
# Creates a single sequence
# Creates a training set of sequences

    numEdge_vec = np.random.randint(3,7,num_ex)
    distort_vec = np.random.uniform(0,0.5,num_ex)
    r1_vec, r2_vec = np.random.uniform(n/8,n/2,num_ex), np.random.uniform(n/8,n/2,num_ex)
    rot1_vec, rot2_vec = np.random.uniform(0,2,num_ex), np.random.uniform(0,2,num_ex)
    xoff1_vec, xoff2_vec = np.random.uniform(n/4,3*n/4,num_ex), np.random.uniform(n/4,3*n/4,num_ex)
    yoff1_vec, yoff2_vec = np.random.uniform(n/4,3*n/4,num_ex), np.random.uniform(n/4,3*n/4,num_ex)
    intensity1_vec, intensity2_vec = np.random.uniform(1,10,num_ex), np.random.uniform(1,10,num_ex)
    
    train = poly.make_polygon_batch(num_ex, n, sequence_length, numEdge_vec, distort_vec, r1_vec, r2_vec, rot1_vec, rot2_vec, xoff1_vec, xoff2_vec, yoff1_vec, yoff2_vec, intensity1_vec, intensity2_vec)
    
    
    if display == 1:
        poly.display_sequence(train[1,:,:,:])    

    return train