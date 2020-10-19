# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 13:48:18 2020

@author: josea
"""
#%% Imports
import utils
import torch
import torch.nn as nn

from polychar import PolyChar
from argparse import Namespace 

#%% Set-up paramenters
args = Namespace(
    csv='../processed_data/ESP-ENG.csv',
    vectorizer_file="../processed_data/vectorizer.json",
    model_checkpoint_file='/models/checkpoints/',
    model_save_file='/models/bil_50-50.pt',
    hidden_dim=128,
    n_lstm_layers=1,
    n_epochs=20,
    learning_rate=0.001,
    batch_size=64,
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    seed=404
    )

utils.set_all_seeds(args.seed, args.device)

#%% Set-up dataset, vectorizer, and model
dataset = utils.TextDataset.load_dataset_and_make_vectorizer(args.csv, p=0.5)
# dataset.save_vectorizer(args.vectorizer_file)
vectorizer = dataset.get_vectorizer()

train_words = list(dataset.df.data[dataset.df.split == 'train'])
test_words = list(dataset.df.data[dataset.df.split == 'test'])

ngram_models = {}
for n in range(2, 5):
    ngram_models[f"{n}gram"] = utils.CharNGram(train_words, n)
    
lstm_model = torch.load(args.model_save_file)
lstm_model.eval()
    
#%% Compare accuracy across models
# Get accuracy from the different models


#%% Compare distribution of last character in test words
# Create a list of distributions for each word [len(test_words), len(vocab)]
# Average the KL divergence/other distance metric over all the words in test set
# Plot
    
#%% Compare word generation