# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 15:14:44 2020

@author: josea
"""
# %% Impports
import sys
import os

sys.path.append(os.path.abspath(".."))

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

# %% Set-up paramenters
args = Namespace(
    csv='../../processed_data/',
    vectorizer_file="../../processed_data/vectorizer.json",
    model_checkpoint_file='../models/checkpoints/',
    model_save_file='../models/param_search/',
    hidden_dims=[64, 128, 256, 512],
    batch_size=256,
    datafiles=['ESP-ENG.csv', 'ESP-EUS.csv'],
    modelfiles=['ESEN_', 'ESEU_'],
    probs=[1, 20, 40, 50, 60, 80, 99],
    n_runs=5,
    plotting=False,
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    seed=404
)

# %% Helper
def get_distribution_from_context(model, context, vectorizer, device='cpu',
                                  softmax=True):
    model.to(device)
    hidden = model.initHidden(1, device)

    for i, (f_v, t_v) in vectorizer.vectorize_single_char(context):
        f_v = f_v.to(device)
        out, hidden = model(f_v.unsqueeze(1), hidden)
    dist = torch.flatten(out.detach())
    # Take only valid continuations (letters + SOS)
    dist = dist[:-2] + dist[-1]

    if softmax:
        dist = F.softmax(dist, dim=0)

    return dist.numpy()

def kl_divergence(dist1, dist2):
    pos = (dist1 != 0.) & (dist2 != 0.)
    return np.sum(dist1[pos] * (np.log2(dist1[pos]) - np.log2(dist2[pos])))

def eval_distributions(model, trie, vectorizer, vocab_len=27):
    total_kl = 0
    total_eval = 0
    q = [trie.root]
    while q:
        p = []
        curr = q.pop(0)
        cnt = 0
        for ch in range(vocab_len):
            if curr.children[ch]:
                q.append(curr.children[ch])
                p.append(curr.children[ch].prob)
            else:
                cnt += 1
                p.append(0)
        if cnt < vocab_len:
            e_dist = np.float32(p)
            context = curr.prefix
            
            p_dist = get_distribution_from_context(model, context, vectorizer)
            
            total_kl += kl_divergence(e_dist, p_dist)
            
            total_eval += 1
    return total_kl / total_eval
    
# %%
res = defaultdict(list)

for data, category in zip(args.datafiles, args.modelfiles):
    for prob in [50, 99]:
        end = f"{prob:02}-{100-prob:02}"

        dataset = utils.TextDataset.load_dataset_and_make_vectorizer(
            args.csv + data,
            p=prob/100, seed=args.seed)
        vectorizer = dataset.get_vectorizer()

        train_words = list(dataset.train_df.data)
        val_words = list(dataset.val_df.data)

        train_trie = utils.Trie()
        train_trie.insert_many([list(w) + ['</s>'] for w in train_words])
        
        val_trie = utils.Trie()
        val_trie.insert_many([list(w) + ['</s>'] for w in val_words])

        for hidden_units in args.hidden_dims:
            m_name = f"{category}_{hidden_units}_{end}"

            run = 0
            print(f"\n{data}: {m_name}_{run}\n")
            model = torch.load(args.model_save_file +
                               f"{m_name}/{m_name}_{run}" + ".pt")

            loss_func1 = nn.CrossEntropyLoss(
                ignore_index=vectorizer.data_vocab.PAD_idx)

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

            dataset.set_split('val')
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

            res['val_loss'].append(running_loss)
            res['val_accuracy'].append(running_acc)
            res['val_perplexity'].append(math.exp(running_loss))
            res['KL'].append(eval_distributions(model, train_trie, vectorizer))
            res['val_KL'].append(eval_distributions(model, val_trie,
                                                    vectorizer))

results = pd.DataFrame(res)

results_acc = pd.melt(results, id_vars=['dataset', 'prob', 'hidden'],
                      value_vars=['accuracy', 'val_accuracy'],
                      var_name='split', value_name='ACC')
results_acc['split'] = np.where(
    results_acc.split == 'accuracy', 'train', 'val')

g = sns.catplot(x='hidden', y='ACC', hue='split', hue_order=['train', 'val'],
                row='dataset', col='prob', kind='bar',
                data=results_acc, palette='Reds')


results_loss = pd.melt(results, id_vars=['dataset', 'prob', 'hidden'],
                       value_vars=['loss', 'val_loss'],
                       var_name='split', value_name='LOSS')
results_loss['split'] = np.where(results_loss.split == 'loss', 'train', 'val')

g = sns.catplot(x='hidden', y='LOSS', hue='split', hue_order=['train', 'val'],
                row='dataset', col='prob', kind='bar',
                data=results_loss, palette='Reds')


results_kl = pd.melt(results, id_vars=['dataset', 'prob', 'hidden'],
                       value_vars=['KL', 'val_KL'],
                       var_name='split', value_name='kl')
results_kl['split'] = np.where(results_kl.split == 'KL', 'train', 'val')

g = sns.catplot(x='hidden', y='kl', hue='split', hue_order=['train', 'val'],
                row='dataset', col='prob', kind='bar',
                data=results_kl, palette='Reds')
