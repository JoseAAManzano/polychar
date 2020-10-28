# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 13:48:18 2020

@author: josea
"""
#%% Imports
import utils
import torch
import math
import torch.nn.functional as F
import seaborn as sns
import pandas as pd
import collections
import json

from argparse import Namespace 
from sklearn.manifold import TSNE
from datetime import datetime
from collections import defaultdict

#%% Helper
def get_acc(model, vectorizer, data, device):
    acc = 0.0
    model.eval()
    for i, word in enumerate(data):
        from_v, to_v = vectorizer.vectorize(word)
        from_v, to_v = from_v.to(device), to_v.to(device)
        from_v = from_v.unsqueeze(0)
        
        hidden = model.initHidden(1, device)
        
        out, _ = model(from_v, hidden)
          
        char_acc = utils.compute_accuracy(out, to_v,
                                          vectorizer.data_vocab.PAD_idx)
        acc += (char_acc - acc) / (i+1)
        
    return acc

def get_hidden_representation(model, vectorizer, df, device):
    ret = pd.DataFrame()
    words = df.data
    model.eval()
    for i, word in enumerate(words):
        from_v, to_v = vectorizer.vectorize(word)
        from_v, to_v = from_v.to(device), to_v.to(device)
        from_v = from_v.unsqueeze(0)
        
        hidden = model.initHidden(1, device)
        
        out, hidden = model(from_v, hidden)
        
        ret[word] = torch.flatten(hidden[0].detach()).to('cpu').numpy()
    ret = ret.T
    ret['data'] = ret.index
    
    return df.merge(ret, on='data')

#%% Set-up paramenters
args = Namespace(
    csv='../processed_data/',
    model_checkpoint_file='/models/checkpoints/',
    model_save_file='models/',
    datafiles = ['ESP-ENG.csv', 'ESP-EUS.csv'],
    modelfiles = ['ESEN_', 'ESEU_'],
    probs = [1, 20, 40, 50, 60, 80, 99],
    n_runs = 5,
    hidden_dim=128,
    batch_size=256,
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    seed=404
    )

#%% Set-up experiments
results = pd.DataFrame()
results_similarity = pd.DataFrame()

eval_words = pd.read_csv(args.csv + 'exp_words.csv')
es0_words = list(eval_words[eval_words.condition == 'ES-']['word'])
es1_words = list(eval_words[eval_words.condition == 'ES+']['word'])

for data, category in zip(args.datafiles, args.modelfiles):
    for prob in args.probs:
        end = f"{prob:02}-{100-prob:02}"
        m_name = f"{category}{end}"
        
        dataset = utils.TextDataset.load_dataset_and_make_vectorizer(
                args.csv + data,
                p=prob/100, seed=args.seed)
        vectorizer = dataset.get_vectorizer()
        
        train_words = list(dataset.train_df.data)
        test_words = list(dataset.test_df.data)
        
        tmp = collections.defaultdict(list)
        
        for run in range(args.n_runs):
            print(f"\n{data}: {m_name}_{run}\n")   
            
            # Get N-gram models with different laplace constant
            ngrams = {}
            for n in range(2, 5):
                ngrams[f"{n}-gram"] = utils.CharNGram(train_words, n,
                                                      laplace=(run+1)*0.2)
            
            for name, m in ngrams.items():
                tmp["model"].append(name)
                tmp["prob"].append(end)
                tmp["data"].append(category[:-1])
                tmp["run"].append(run)
                tmp["accuracy_train"].append(m.calculate_accuracy(train_words))
                tmp["perplexity_train"].append(m.perplexity(train_words))
                tmp["accuracy"].append(m.calculate_accuracy(test_words))
                tmp["perplexity"].append(m.perplexity(test_words))
                tmp["ES+"].append(m.calculate_accuracy(es1_words))
                tmp["ES-"].append(m.calculate_accuracy(es0_words))
            
            lstm_model = torch.load(args.model_save_file +\
                                    f"{m_name}/{m_name}_{run}.pt")
            lstm_model.eval()
            lstm_model.to(args.device)
            
            dataset.set_split('train')
            batch_generator = utils.generate_batches(dataset, 
                                       batch_size=args.batch_size, 
                                       device=args.device)
            
            acc = 0.0
            perp = 0.0
            
            for batch_id, batch_dict in enumerate(batch_generator):
                hidden = lstm_model.initHidden(args.batch_size, args.device)
                
                out, hidden = lstm_model(batch_dict['X'], hidden)
                
                acc_chars = utils.compute_accuracy(out,
                                       batch_dict['Y'],
                                       vectorizer.data_vocab.PAD_idx)
                perp_chars = F.cross_entropy(*utils.normalize_sizes(
                                    out,
                                    batch_dict['Y']),
                                    ignore_index=vectorizer.data_vocab.PAD_idx
                                    )
                acc += (acc_chars - acc) / (batch_id + 1)
                perp += (perp_chars.item() - perp) / (batch_id + 1)
            
            tmp["model"].append("LSTM")
            tmp["prob"].append(end)
            tmp["data"].append(category[:-1])
            tmp['run'].append(run)
            tmp["accuracy_train"].append(acc)
            tmp["perplexity_train"].append(math.exp(perp))
            
            dataset.set_split('test')
            batch_generator = utils.generate_batches(dataset, 
                                       batch_size=args.batch_size, 
                                       device=args.device)
            
            acc = 0.0
            perp = 0.0
            
            for batch_id, batch_dict in enumerate(batch_generator):
                hidden = lstm_model.initHidden(args.batch_size, args.device)
                
                out, hidden = lstm_model(batch_dict['X'], hidden)
                
                acc_chars = utils.compute_accuracy(out,
                                       batch_dict['Y'],
                                       vectorizer.data_vocab.PAD_idx)
                perp_chars = F.cross_entropy(*utils.normalize_sizes(
                                    out,
                                    batch_dict['Y']),
                                    ignore_index=vectorizer.data_vocab.PAD_idx
                                    )
                acc += (acc_chars - acc) / (batch_id + 1)
                perp += (perp_chars.item() - perp) / (batch_id + 1)
                
            tmp["accuracy"].append(acc)
            tmp["perplexity"].append(math.exp(perp))
            tmp["ES-"].append(get_acc(lstm_model, vectorizer,
                                      es0_words, args.device))
            tmp["ES+"].append(get_acc(lstm_model, vectorizer,
                                      es1_words, args.device))
            
        results = pd.concat([results, pd.DataFrame(tmp)], axis=0,
                            ignore_index=True)
        
sns.catplot(x="prob", y="accuracy", hue="model", row="data", kind='point',
            data=results, palette="Reds")

sns.catplot(x="prob", y="perplexity", hue="model", row="data", kind='point',
            data=results, palette="Reds")

sns.catplot(x="prob", y="accuracy_train", hue="model", row="data", kind='point',
            data=results, palette="Reds")

sns.catplot(x="prob", y="perplexity_train", hue="model", row="data", kind='point',
            data=results, palette="Reds")

exp_data = results[['model', 'prob', 'data', 'run', 'ES-', 'ES+']]
exp_data = pd.melt(exp_data, id_vars=['model', 'prob', 'data', 'run'],
                   value_vars=['ES-', 'ES+'])

sns.catplot(x="prob", y="value", hue="variable", row="data", col="model",
            data=exp_data, kind="bar", palette="Reds")

#%% Distribution of last character
# TODO Compare distribution of last character in test words
# Create a list of distributions for each word [len(test_words), len(vocab)]
# Average the KL divergence/other distance metric over all the words in test
# Plot

#%% Hidden representation for each time-point for each word
# First is necessary to run the readout.py file to produce the representations
hidd_cols = [str(i) for i in range(args.hidden_dim)]
tmp = pd.read_json(f"{args.save_file}/ESEN_50-50_0.json", encoding='utf-8')
tmp[hidd_cols] = tmp[hidd_cols].astype('float32')



from sklearn import LogisticRegression




#%% Clustering of hidden representations
# Can just use the previous dataframe and take the last character for each word
# Or get RDMs for each character
        
#         tsne = TSNE(n_components=2, n_jobs=-1, random_state=args.seed)
        
#         repr_df = get_hidden_representation(lstm_model, vectorizer,
#                                             test_df, args.device)
        
#         repr_df[['tsne1','tsne2']] = tsne.fit_transform(repr_df.iloc[:, 3:])
#         repr_df['model'] = end
#         repr_df['dataset'] = data
        
#         hidden_repr = pd.concat([hidden_repr, repr_df], ignore_index=True)
        
#         eval_df = get_hidden_representation(lstm_model, vectorizer,
#                                             eval_words, args.device)
        
#         eval_df[['tsne1','tsne2']] = tsne.fit_transform(eval_df.iloc[:, 2:])
#         eval_df['model'] = end
#         eval_df['dataset'] = data
        
#         eval_repr = pd.concat([eval_repr, eval_df], ignore_index=True)
        
# g = sns.FacetGrid(hidden_repr, col="model", row="dataset", hue="label")
# g.map(sns.scatterplot, "tsne1", "tsne2", alpha=0.7)
# g.add_legend()

# g = sns.FacetGrid(eval_repr, col="model", row="dataset", hue="label")
# g.map(sns.scatterplot, "tsne1", "tsne2", alpha=0.7)
# g.add_legend()
