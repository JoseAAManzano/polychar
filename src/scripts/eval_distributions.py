# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 11:02:36 2020

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
import string

from argparse import Namespace 
from collections import defaultdict

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn import metrics as mtr
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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
    learning_rate=0.001,
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    seed=404
    )

#%% Empirical evaluation
def kl_divergence(dist1, dist2, adjustment=0):
    if adjustment:
        dist1 += adjustment
        dist1 /= dist1.sum()
        dist2 += adjustment
        dist2 /= dist2.sum()
        pos = [True] * len(dist1)
    else:
        pos = (dist1 != 0.) & (dist2 != 0.)
    return np.sum(dist1[pos] * (np.log2(dist1[pos]) - np.log2(dist2[pos])))

def cosine_distance(dist1, dist2):
    return dist1.dot(dist2) / (np.linalg.norm(dist1) * np.linalg.norm(dist2))

def sse(dist1, dist2):
    return np.square(dist2 - dist1).sum()
        
def get_distribution_from_context(model, context, vectorizer, softmax=True):
    hidden = model.initHidden(1, args.device)
    
    for i, (f_v, t_v) in vectorizer.vectorize_single_char(context):
        f_v = f_v.to(args.device)
        out, hidden = model(f_v.unsqueeze(1), hidden)
    dist = torch.flatten(out.detach()).to('cpu')
    dist = dist[:-2] + dist[-1] # Take only valid continuations (letters + SOS)
    
    if softmax:
        dist = F.softmax(dist, dim=0)
    
    return dist.numpy()

def empirical_evaluation(args, metrics):

    for data, category in zip(args.datafiles, args.modelfiles):
        
        df = pd.read_csv(args.csv + data)
        vectorizer = utils.Vectorizer.from_df(df)
    
        trie = utils.Trie()
        trie.insert_many(list(df.data))
        
        for prob in [50, 99]:
            
            end = f"{prob:02}-{100-prob:02}"
            m_name = f"{category}{end}"
            
            dataset = utils.TextDataset.load_dataset_and_make_vectorizer(
                    args.csv + data,
                    p=prob/100, seed=args.seed)
            
            model = torch.load()



df = pd.read_csv(args.csv + 'ESP-ENG.csv')

trie = utils.Trie()
trie.insert_many([list(w) + ['</s>'] for w in list(df.data)])
trie.print_empirical_distribution()

model = torch.load('models/ESEN_50-50/ESEN_50-50_0.pt')

metrics = {'KL': kl_divergence, 'cosine': cosine_distance,
           'sse': sse}


plt.bar(list(string.ascii_lowercase) + ['</s>'], get_distribution_from_context(model, "loc", vectorizer))
plt.bar(list(string.ascii_lowercase) + ['</s>'], np.float32(trie.get_distribution_from_context("loc")))

dist2 = get_distribution_from_context(model, "loc", vectorizer)
dist1 = np.float32(trie.get_distribution_from_context("loc"))

print(sse(dist1, dist2))
print(cosine_distance(dist1, dist2))
print(kl_divergence(dist1, dist2))
print(lrt(dist1, dist2))
