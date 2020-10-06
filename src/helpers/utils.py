# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 17:23:28 2020

@author: josea
"""
import torch
from torch.utils.data import DataLoader
import numpy as np

# def encode(st, stoi):
#     """
#     Output: List of <SOS> + char indices + <EOS>
#     """
#     return [0] + [stoi[c] for c in st] + [1]

# def letter2onehot(letter, n_letters, stoi):
#     '''
#     Returns one-hot encoding of a letter in shape (1, n_letters)
#     '''
#     onehot = torch.zeros(1, n_letters)
#     onehot[0][stoi[letter]] = 1
#     return onehot


# def line2tensor(st, n_letters):
#     '''
#     Returns one-hot Tensor of a string in shape (len(st), 1, n_letters)
#     '''
#     tensor = torch.zeros(len(st), 1, n_letters)
#     for i, l in enumerate(st):
#         tensor[i][0][l] = 1
#     return tensor

# def randomExample(data, n_letters, stoi, device, langs, ref='ESP', p=[0.5, 0.5]):
#     lang = np.random.choice(langs, p=p)
#     word = np.random.choice(data[lang], size=1)[0]
#     word = encode(word, stoi)
#     in_ = line2tensor(word[:-1], n_letters).to(device)
#     out_ = torch.LongTensor(word[1:]).to(device)
#     lang_ = torch.tensor(0.0 if lang == ref else 1.0).view(1, -1).to(device)
#     return in_, out_, lang_

def activ2color(value):
    colors = ['#85c2e1', '#89c4e2', '#95cae5', '#99cce6', '#a1d0e8'
    		'#b2d9ec', '#baddee', '#c2e1f0', '#eff7fb', '#f9e8e8',
    		'#f9e8e8', '#f9d4d4', '#f9bdbd', '#f8a8a8', '#f68f8f',
    		'#f47676', '#f45f5f', '#f34343', '#f33b3b', '#f42e2e']
    value = int((value*100)/5)
    return colors[value]

def generate_batches(dataset, batch_size, shuffle=True,
                     drop_last=True, device="cpu"):
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                             shuffle=shuffle, drop_last=drop_last)
    
    for data_dict in dataloader:
        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name].to(device)
        yield out_data_dict