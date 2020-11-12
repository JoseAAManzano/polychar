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
    device=torch.device('cpu'),
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
    res = defaultdict(list)
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
            
            train_words = list(dataset.train_df.data)
            
            for run in range(args.n_runs):
                print(f"\n{data}: {m_name}_{run}\n")
                    
                ngrams = {}
                for n in range(2, 6):
                    ngrams[f"{n}-gram"] = utils.CharNGram(train_words, n,
                                                          laplace=(run+1)*0.2)
                
                lstm_model = torch.load(args.model_save_file +\
                                        f"{m_name}/{m_name}_{run}.pt")
                lstm_model.to(args.device)
                lstm_model.eval()
                
                ngrams['LSTM'] = lstm_model
                
                q = []
                q.append(trie.root)
                while q:
                    p = []
                    curr = q.pop(0)
                    cnt = 0
                    for ch in range(27):
                        if curr.children[ch]:
                            q.append(curr.children[ch])
                            p.append(curr.children[ch].prob)
                        else:
                            cnt += 1
                            p.append(0)
                    if cnt < 27:
                        e_dist = np.float32(p)
                        context = curr.prefix
                        
                        for model, m in ngrams.items():
                            if isinstance(m, utils.CharNGram):
                                p_dist = m.get_distribution_from_context(context).values()
                                p_dist = np.float32(list(p_dist))
                            else:
                                p_dist = get_distribution_from_context(lstm_model,
                                                                       context,
                                                                       vectorizer)
   
                
                # TODO append here by dividing by the total
                res['model'].append(model)
                res['dataset'].append(category[:-1])
                res['prob'].append(prob)
                res['run'].append(run)
                for metric, func in metrics.items():
                    res[metric].append(func(e_dist, p_dist))
                        
    return res
                            
metrics = {'KL': kl_divergence, 'cosine': cosine_distance,
           'sse': sse}

results = empirical_evaluation(args, metrics)

results = pd.DataFrame(results)

sns.catplot(x='model', y='KL', hue='prob', row='dataset', data=results,
            kind='bar', palette='Reds')
