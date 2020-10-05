# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 14:18:44 2020

@author: josea
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class PolyChar(nn.Module):
    def __init__(self, n_in, n_hidden, n_layers, n_out, drop_p=0.1):
        super(PolyChar, self).__init__()
        self.drop_p = drop_p
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        
        self.lstm = nn.LSTM(
            input_size=n_in,
            hidden_size=n_hidden,
            num_layers=n_layers,
            dropout=drop_p,
            )
        self.fc = nn.Linear(n_hidden, n_out)
        # Predict language class after every letter
        self.lang_pred = nn.Linear(n_hidden, 1) 
        
    def forward(self, x, hidden):
        in_ = x.view(1, 1, -1)
        out, hidden = self.lstm(in_, hidden)
        out1 = F.dropout(out, p=self.drop_p)
        out = self.fc(out1.view(1, -1))
        l_pred = self.lang_pred(out1)
        return out, hidden, l_pred
        
    def initHidden(self, device):
        return (torch.zeros(self.n_layers, 1, self.n_hidden).to(device),
                torch.zeros(self.n_layers, 1, self.n_hidden).to(device))