# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 13:48:18 2020

@author: josea
"""
#%% Imports
import utils
import torch
import math
import torch.nn as nn
import seaborn as sns
import pandas as pd
import collections

from argparse import Namespace 

#%% Set-up paramenters
args = Namespace(
    csv='../processed_data/',
    model_checkpoint_file='/models/checkpoints/',
    model_save_file='models/',
    datafiles = ['ESP-ENG.csv', 'ESP-EUS.csv'],
    modelfiles = ['ESEN_', 'ESEU_'],
    probs = [0.01, 0.3, 0.5, 0.7, 0.99],
    batch_size=256,
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    seed=404
    )

utils.set_all_seeds(args.seed, args.device)

#%% Set-up experiments
results = pd.DataFrame()

for data, model in zip(args.datafiles, args.modelfiles):
    for prob in args.probs:
        end = f"{int(prob*100)}-{int((1-prob)*100)}"
        m_name = f"{model}{end}"
        print(f"{data}: {m_name}")
        
        dataset = utils.TextDataset.load_dataset_and_make_vectorizer(
            args.csv + data,
            p=prob)
        vectorizer = dataset.get_vectorizer()
        
        train_words = list(dataset.train_df.data)
        test_words = list(dataset.train_df.data)
        
        ngrams = {}
        for n in range(2, 5):
            ngrams[f"{n}-gram"] = utils.CharNGram(train_words, n)
                
        tmp = collections.defaultdict(list)
        
        for name, m in ngrams.items():
            tmp["model"].append(name)
            tmp["prob"].append(end)
            tmp["data"].append(model[:-1])
            tmp["accuracy"].append(m.calculate_accuracy(test_words))
            tmp["perplexity"].append(m.perplexity(test_words))
        
        dataset.set_split('test')
        batch_generator = utils.generate_batches(dataset, 
                                   batch_size=args.batch_size, 
                                   device=args.device)
        
        lstm_model = torch.load(args.model_save_file + m_name + ".pt")
        lstm_model.eval()
        
        acc = 0.0
        perp = 0.0
        loss = nn.CrossEntropyLoss(ignore_index=vectorizer.data_vocab.PAD_idx)
        
        for batch_id, batch_dict in enumerate(batch_generator):
            hidden = lstm_model.initHidden(args.batch_size, args.device)
            
            out, hidden = lstm_model(batch_dict['X'], hidden)
            
            acc_chars = utils.compute_accuracy(out,
                                   batch_dict['Y'],
                                   vectorizer.data_vocab.PAD_idx)
            perp_chars = loss(*utils.normalize_sizes(out, batch_dict['Y']))
            acc += (acc_chars - acc) / (batch_id + 1)
            perp += (perp_chars.item() - perp) / (batch_id + 1)
            
        tmp["model"].append("LSTM")
        tmp["prob"].append(end)
        tmp["data"].append(model[:-1])
        tmp["accuracy"].append(acc)
        tmp["perplexity"].append(math.exp(perp))
        
        results = pd.concat([results, pd.DataFrame(tmp)], axis=0)
        
sns.catplot(x="prob", y="accuracy", hue="model", col="data", kind='point',
            data=results, palette="Reds")

sns.catplot(x="prob", y="perplexity", hue="model", col="data", kind='point',
            data=results, palette="Reds")

#%% Compare distribution of last character in test words
# Create a list of distributions for each word [len(test_words), len(vocab)]
# Average the KL divergence/other distance metric over all the words in test set
# Plot
test_words_pad = [list(word) + ["</s>"] for word in test_words]

trie = utils.Trie()
trie.insert_many(test_words_pad)

def calculate_empirical_distribution(trie):
    """Calculates empirical distribution for the entire Trie"""
    q = []
    q.append(trie.root)
    while q:
        p = []
        curr = q.pop()
        cnt = 0
        for i in range(trie.vocab_len):
            if curr.children[i]:
                q.append(curr.children[i])
                p.append(curr.children[i].prob)
            else:
                cnt += 1
                p.append(0)

#%% Separation of langauges in hidden layers

