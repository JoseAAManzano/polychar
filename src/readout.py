# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 14:54:12 2020

@author: josea
"""
#%% Readout of hidden layer for ever
import os
import utils
import torch
import json

from argparse import Namespace 
from datetime import datetime
from collections import defaultdict

#%% Set-up paramenters
args = Namespace(
    csv='../processed_data/',
    save_file='hidden/',
    model_checkpoint_file='/models/checkpoints/',
    model_save_file='models/',
    datafiles = ['ESP-ENG.csv', 'ESP-EUS.csv'],
    modelfiles = ['ESEN_', 'ESEU_'],
    probs = [1, 20, 40, 50, 60, 80, 99],
    n_runs = 5,
    hidden_dim=128,
    batch_size=256,
    device=torch.device('cpu'),
    seed=404
    )

curr_dir = os.getcwd()

try:
    os.stat(os.path.join(curr_dir, args.save_file))
except:
    os.mkdir(os.path.join(curr_dir, args.save_file))

#%% Hidden representation for each time-point for each word
for data, category in zip(args.datafiles, args.modelfiles):
    for prob in args.probs:
        end = f"{prob:02}-{100-prob:02}"
        m_name = f"{category}{end}"
        
        dataset = utils.TextDataset.load_dataset_and_make_vectorizer(
                args.csv + data,
                p=prob/100, seed=args.seed)
        vectorizer = dataset.get_vectorizer()
    
        test_df = dataset.test_df
        
        for run in range(args.n_runs):
            t0 = datetime.now()
            print(f"\n{data}: {m_name}_{run}\n")
            cols = ['dataset','prob','run','word','label','char']
            hidd_cols = [str(i) for i in range(args.hidden_dim)]
            tmp = defaultdict(list)
            
            model = torch.load(args.model_save_file +\
                                    f"{m_name}/{m_name}_{run}.pt")
            model.to(args.device)
            model.eval()
            for w, l in zip(test_df.data, test_df.label):
                hidden = model.initHidden(1, args.device)
    
                for i, (f_v, t_v) in vectorizer.vectorize_single_char(w):
                    f_v, t_v = f_v.to(args.device), t_v.to(args.device)
                    _, hidden = model(f_v.unsqueeze(1), hidden)
                    tmp['dataset'].append(category[:-1])
                    tmp['prob'].append(end)
                    tmp['run'].append(run)
                    tmp['char'].append(i)
                    tmp['word'].append(w)
                    tmp['label'].append(l)
                    hid = torch.flatten(hidden[0].detach()).to('cpu').numpy()
                    for k, v in zip(hidd_cols, hid):
                        tmp[k].append(str(v))

            with open(f"{args.save_file}/{m_name}_{run}.json", 'w',
                      encoding='utf-8') as f:
                json.dump(tmp, f)
            print(f"{(datetime.now() - t0).seconds}s")

