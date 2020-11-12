# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 15:14:44 2020

@author: josea
"""
#%% Impports
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

#%% Set-up paramenters
args = Namespace(
    csv='../../processed_data/',
    vectorizer_file="../../processed_data/vectorizer.json",
    model_checkpoint_file='../models/checkpoints/',
    model_save_file='../models/param_search/',
    hidden_dims=[64, 128, 256, 512],
    batch_size=256,
    datafiles = ['ESP-ENG.csv', 'ESP-EUS.csv'],
    modelfiles = ['ESEN_', 'ESEU_'],
    probs = [1, 20, 40, 50, 60, 80, 99],
    n_runs = 5,
    plotting=False,
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    seed=404
    )

#%% 
res = defaultdict(list)

for data, category in zip(args.datafiles, args.modelfiles):
    for prob in [50, 99]:
        end = f"{prob:02}-{100-prob:02}"
        
        df = pd.read_csv(args.csv + data)
        vectorizer = utils.Vectorizer.from_df(df)
        
        trie = utils.Trie()
        trie.insert_many([list(w) + ['</s>'] for w in df.data])

        # Set-up dataset, vectorizer, and model
        dataset = utils.TextDataset.make_text_dataset(df, vectorizer,
            p=prob/100, seed=args.seed)
        
        for hidden_units in args.hidden_dims:
            m_name = f"{category}_{hidden_units}_{end}"

            run = 0
            print(f"\n{data}: {m_name}_{run}\n")
            model = torch.load(args.model_save_file +\
                                       f"{m_name}/{m_name}_{run}" + ".pt")
                
            loss_func1 = nn.CrossEntropyLoss(ignore_index=vectorizer.data_vocab.PAD_idx)

            dataset.set_split('train')
            batch_generator = utils.generate_batches(dataset,
                                         batch_size=args.batch_size,
                                         device=args.device)
            running_loss = 0.
            running_acc = 0.
            
            model.eval()
            for batch_id, batch_dict in enumerate(batch_generator):
                    hidden = model.initHidden(args.batch_size, args.device)
                    
                    out, hidden = model(batch_dict['X'], hidden)
                    
                    loss = loss_func1(*utils.normalize_sizes(out, batch_dict['Y']))
                    
                    running_loss += (loss.item() - running_loss) / (batch_id + 1)
                    acc_chars = utils.compute_accuracy(
                                    out, batch_dict['Y'], vectorizer.data_vocab.PAD_idx
                                    )
                    running_acc += (acc_chars - running_acc) / (batch_id + 1)
            
            res['dataset'].append(category[:-1])
            res['prob'].append(end)
            res['hidden'].append(hidden_units)
            res['loss'].append(running_loss)
            res['accuracy'].append(running_acc)
            res['perplexity'].append(math.exp(running_loss))
            
            dataset.set_split('test')
            batch_generator = utils.generate_batches(dataset,
                                         batch_size=args.batch_size,
                                         device=args.device)
            running_loss = 0.
            running_acc = 0.
            
            model.eval()
            for batch_id, batch_dict in enumerate(batch_generator):
                    hidden = model.initHidden(args.batch_size, args.device)
                    
                    out, hidden = model(batch_dict['X'], hidden)
                    
                    loss = loss_func1(*utils.normalize_sizes(out, batch_dict['Y']))
                    
                    running_loss += (loss.item() - running_loss) / (batch_id + 1)
                    acc_chars = utils.compute_accuracy(
                                    out, batch_dict['Y'], vectorizer.data_vocab.PAD_idx
                                    )
                    running_acc += (acc_chars - running_acc) / (batch_id + 1)
            

            res['test_loss'].append(running_loss)
            res['test_accuracy'].append(running_acc)
            res['test_perplexity'].append(math.exp(running_loss))
            
results = pd.DataFrame(res)

results_acc = pd.melt(results, id_vars=['dataset', 'prob', 'hidden'],
                      value_vars=['accuracy','test_accuracy'],
                      var_name='split', value_name='ACC')
results_acc['split'] = np.where(results_acc.split == 'accuracy', 'train', 'test')

g= sns.catplot(x='hidden', y='ACC', hue='split', hue_order=['train', 'test'],
               row='dataset', col='prob', kind='bar', data=results_acc, palette='Reds')


results_loss = pd.melt(results, id_vars=['dataset', 'prob', 'hidden'],
                      value_vars=['loss','test_loss'],
                      var_name='split', value_name='LOSS')
results_loss['split'] = np.where(results_loss.split == 'loss', 'train', 'test')

g= sns.catplot(x='hidden', y='LOSS', hue='split', hue_order=['train', 'test'],
               row='dataset', col='prob', kind='bar', data=results_loss, palette='Reds')