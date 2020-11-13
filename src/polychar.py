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
        self.n_out = n_out

        self.lstm = nn.LSTM(
            input_size=n_in,
            hidden_size=n_hidden,
            num_layers=n_layers,
            dropout=drop_p if n_layers > 1 else 0,
            batch_first=True
        )
        self.fc = nn.Linear(n_hidden, n_out)

    def forward(self, x, hidden, apply_sig=False, apply_sfmx=False):
        out, hidden = self.lstm(x, hidden)

        # # Reshape output
        batch_size, seq_size, n_hidden = out.shape
        out1 = out.contiguous().view(batch_size*seq_size, n_hidden)

        chars_out = self.fc(F.dropout(out1, p=self.drop_p))

        if apply_sfmx:
            chars_out = F.softmax(chars_out, dim=1)

        return chars_out.view(batch_size, seq_size, self.n_out), hidden

    def initHidden(self, batch_size, device):
        return (torch.zeros(self.n_layers, batch_size, self.n_hidden).to(device),
                torch.zeros(self.n_layers, batch_size, self.n_hidden).to(device))
