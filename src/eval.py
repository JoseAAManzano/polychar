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
import matplotlib.pyplot as plt

from argparse import Namespace 

#%% Set-up paramenters
args = Namespace(
    csv='../processed_data/ESP-ENG.csv',
    vectorizer_file="../processed_data/vectorizer.json",
    model_checkpoint_file='/models/checkpoints/',
    model_save_file='models/bil_50-50.pt',
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

train_words = list(dataset.train_df.data)
test_words = list(dataset.test_df.data)

ngram_models = {}
for n in range(2, 6):
    ngram_models[f"{n}-gram"] = utils.CharNGram(train_words, n)
    
lstm_model = torch.load(args.model_save_file)
lstm_model.eval()

lstm_model3 = torch.load("models/bil_50-50_3layer.pt")
lstm_model3.eval()
    
#%% Compare accuracy across models
# Get accuracy from the different models
# TODO compare accuracy on val data as well
accs = {}
perps = {}

for name, m in ngram_models.items():
    accs[name] = m.calculate_accuracy(test_words)
    perps[name] = m.perplexity(test_words)
    
dataset.set_split('test')
batch_generator = utils.generate_batches(dataset, 
                                   batch_size=args.batch_size, 
                                   device=args.device)
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
    
accs['LSTM'] = acc
perps['LSTM'] = math.exp(perp)

acc = 0.0
perp = 0.0
loss = nn.CrossEntropyLoss(ignore_index=vectorizer.data_vocab.PAD_idx)
batch_generator = utils.generate_batches(dataset, 
                                   batch_size=args.batch_size, 
                                   device=args.device)
for batch_id, batch_dict in enumerate(batch_generator):
    hidden = lstm_model3.initHidden(args.batch_size, args.device)
    
    out, hidden = lstm_model3(batch_dict['X'], hidden)
    
    acc_chars = utils.compute_accuracy(out,
                                       batch_dict['Y'],
                                       vectorizer.data_vocab.PAD_idx)
    perp_chars = loss(*utils.normalize_sizes(out, batch_dict['Y']))
    acc += (acc_chars - acc) / (batch_id + 1)
    perp += (perp_chars.item() - perp) / (batch_id + 1)

accs['LSTM3'] = acc
perps['LSTM3'] = math.exp(perp)

plt.figure()
plt.bar(accs.keys(), accs.values(), label="Accuracy")
plt.bar(perps.keys(), perps.values(), label="Perplexity")
plt.legend()
plt.show()

#%% Compare distribution of last character in test words
# Create a list of distributions for each word [len(test_words), len(vocab)]
# Average the KL divergence/other distance metric over all the words in test set
# Plot
test_words_pad = [list(word) + ["</s>"] for word in test_words]

trie = utils.Trie()
trie.insert_many(test_words_pad)
trie.get_probabilities()

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

