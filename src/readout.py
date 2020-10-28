# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 14:54:12 2020

@author: josea
"""
#%% Readout of hidden layer for ever
import os
import utils
import torch
import pandas as pd
import torch.multiprocessing as mp

from argparse import Namespace 
from datetime import datetime

#%% Set-up paramenters
args = Namespace(
    csv='../processed_data/',
    save_file='/hidden/',
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

for data, category in zip(args.datafiles, args.modelfiles):
    for prob in args.probs:
        end = f"{prob:02}-{100-prob:02}"
        m_name = f"{category}{end}"
        
        dataset = utils.TextDataset.load_dataset_and_make_vectorizer(
                args.csv + data,
                p=prob/100, seed=args.seed)
        vectorizer = dataset.get_vectorizer()
        
        test_df = dataset.test_df
        
        sema = mp.Semaphore(5)
        sema.acquire()
        all_processes = []
        for run in range(args.n_runs):
            t0 = datetime.now()
            print(f"\n{data}: {m_name}_{run}\n")
            cols = ['dataset','prob','run','word','label','char']
            hidd_cols = [str(i) for i in range(args.hidden_dim)]
            tmp = pd.DataFrame(columns=cols+hidd_cols)
            
            model = torch.load(args.model_save_file +\
                                    f"{m_name}/{m_name}_{run}.pt")
            model.to(args.device)
            model.eval()
            for w, l in zip(test_df.data, test_df.label):
                hidden = model.initHidden(1, args.device)
                for i, (f_v, t_v) in vectorizer.vectorize_single_char(w):
                    _, hidden = model(f_v.unsqueeze(1), hidden)
                    tmp2 = pd.DataFrame()
                    tmp2['dataset'] = category[:-1]
                    tmp2['prob'] = end
                    tmp2['run'] = run
                    tmp2['char'] = i
                    tmp2['word'] = w
                    tmp2['label'] = l
                    tmp2['char'] = i
                    tmp2[hidd_cols] =  torch.flatten(hidden[0].detach()).numpy()
                    tmp = pd.concat([tmp, tmp2], axis=0, ignore_index=True)
            
            tmp.to_csv(f"{args.save_file}{m_name}_{run}.csv", index=False,
                       encoding='utf-8')
            t1 = datetime.now()
            print(f"{(t1-t0).total_seconds:.2f}")
            
                