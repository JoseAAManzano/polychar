# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 09:11:19 2020

@author: josea
"""
import utils
import torch
import math
import torch.nn.functional as F
import seaborn as sns
import pandas as pd
import collections
import torch.nn as nn
import numpy as np

from argparse import Namespace 
from collections import defaultdict

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn import metrics as mtr
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from scipy.spatial import distance
import numpy as np


args = Namespace(
    csv='../processed_data/',
    model_checkpoint_file='/models/checkpoints/',
    save_file='hidden/',
    model_save_file='models/',
    datafiles = ['ESP-ENG.csv', 'ESP-EUS.csv'],
    modelfiles = ['ESEN_', 'ESEU_'],
    probs = [1, 20, 40, 50, 60, 80, 99],
    n_runs = 5,
    hidden_dim=128,
    batch_size=256,
    learning_rate=0.001,
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    seed=404
    )


hidd_cols = [str(i) for i in range(args.hidden_dim)]

for data, category in zip(args.datafiles, args.modelfiles):
    for prob in [50, 99]:
        end = f"{prob:02}-{100-prob:02}"
        m_name = f"{category}{end}"
        
        dataset = pd.DataFrame()
        for run in range(args.n_runs):
            tmp = pd.read_json(f"{args.save_file}/{m_name}_{run}.json",
                               encoding='utf-8')
            dataset = pd.concat([dataset, tmp], axis=0, ignore_index=True)
        
        dataset['len'] = dataset.word.map(len)
        
                
        rdm_data = dataset[(dataset.char == 6) & (dataset.run == 1)]
        rdm_data = rdm_data.sort_values(by='label')
        ticks = list(rdm_data.label)
                    
        rdm_data = np.array(rdm_data[hidd_cols].values)
        rdm_data = (rdm_data - rdm_data.mean()) / rdm_data.std()
        
        RDM = distance.squareform(distance.pdist(rdm_data, 'euclidean'))
        RDM = (RDM - RDM.min()) / (RDM.max() - RDM.min())
        
        plt.figure()
        plt.imshow(RDM, cmap='coolwarm')
        plt.axis('off')
        plt.title(f"MinMax euclidean distance {m_name}_{0}_5")
        plt.colorbar()
        plt.show()
