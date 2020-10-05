# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 17:23:28 2020

@author: josea
"""
import torch

def encode(st, stoi):
    """
    Output: List of <SOS> + char indices + <EOS>
    """
    return [0] + [stoi[c] for c in st] + [1]

def letter2onehot(letter, n_letters, stoi):
    '''
    Returns one-hot encoding of a letter in shape (1, n_letters)
    '''
    onehot = torch.zeros(1, n_letters)
    onehot[0][stoi[letter]] = 1
    return onehot


def line2tensor(st, n_letters):
    '''
    Returns one-hot Tensor of a string in shape (len(st), 1, n_letters)
    '''
    tensor = torch.zeros(len(st), 1, n_letters)
    for i, l in enumerate(st):
        tensor[i][0][l] = 1
    return tensor

def activ2color(value):
    colors = ['#85c2e1', '#89c4e2', '#95cae5', '#99cce6', '#a1d0e8'
    		'#b2d9ec', '#baddee', '#c2e1f0', '#eff7fb', '#f9e8e8',
    		'#f9e8e8', '#f9d4d4', '#f9bdbd', '#f8a8a8', '#f68f8f',
    		'#f47676', '#f45f5f', '#f34343', '#f33b3b', '#f42e2e']
    value = int((value*100)/5)
    return colors[value]