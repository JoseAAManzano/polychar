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

from argparse import Namespace 
from sklearn.manifold import TSNE

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
    probs = [0.01, 0.3, 0.5, 0.7, 0.99],
    batch_size=256,
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    seed=404
    )

utils.set_all_seeds(args.seed, args.device)

#%% Set-up experiments
results = pd.DataFrame()
results_similarity = pd.DataFrame()

eval_words = pd.read_csv(args.csv + 'exp_words.csv')
es0_words = list(eval_words[eval_words.condition == 'ES-']['word'])
es1_words = list(eval_words[eval_words.condition == 'ES+']['word'])

for data, model in zip(args.datafiles, args.modelfiles):
    for prob in args.probs:
        end = f"{int(prob*100)}-{int((1-prob)*100)}"
        m_name = f"{model}{end}"
        print(f"{data}: {m_name}")
        
        dataset = utils.TextDataset.load_dataset_and_make_vectorizer(
            args.csv + data,
            p=prob, seed=args.seed)
        vectorizer = dataset.get_vectorizer()
        
        train_words = list(dataset.train_df.data)
        test_words = list(dataset.test_df.data)
        
        ngrams = {}
        for n in range(2, 6):
            ngrams[f"{n}-gram"] = utils.CharNGram(train_words, n)
                
        tmp = collections.defaultdict(list)
        
        for name, m in ngrams.items():
            tmp["model"].append(name)
            tmp["prob"].append(end)
            tmp["data"].append(model[:-1])
            tmp["accuracy"].append(m.calculate_accuracy(test_words))
            tmp["perplexity"].append(m.perplexity(test_words))
            tmp["ES+"].append(m.calculate_accuracy(es1_words))
            tmp["ES-"].append(m.calculate_accuracy(es0_words))
        
        dataset.set_split('test')
        batch_generator = utils.generate_batches(dataset, 
                                   batch_size=args.batch_size, 
                                   device=args.device)
        
        lstm_model = torch.load(args.model_save_file + m_name + ".pt")
        lstm_model.eval()
        
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
        tmp["data"].append(model[:-1])
        tmp["accuracy"].append(acc)
        tmp["perplexity"].append(math.exp(perp))
        tmp["ES-"].append(get_acc(lstm_model, vectorizer,
                                  es0_words, args.device))
        tmp["ES+"].append(get_acc(lstm_model, vectorizer,
                                  es1_words, args.device))
        
        results = pd.concat([results, pd.DataFrame(tmp)], axis=0)
        
sns.catplot(x="prob", y="accuracy", hue="model", row="data", kind='point',
            data=results, palette="Reds")

sns.catplot(x="prob", y="perplexity", hue="model", row="data", kind='point',
            data=results, palette="Reds")

exp_data = results[['model', 'prob', 'data', 'ES-', 'ES+']]
exp_data = pd.melt(exp_data, id_vars=['model', 'prob', 'data'],
                   value_vars=['ES-', 'ES+'])

sns.catplot(x="prob", y="value", hue="variable", row="data", col="model",
            data=exp_data, kind="bar", palette="Reds")

#%% Distribution of last character
# TODO Compare distribution of last character in test words
# Create a list of distributions for each word [len(test_words), len(vocab)]
# Average the KL divergence/other distance metric over all the words in test
# Plot

#%% Separation of langauges in hidden layers
hidden_repr = pd.DataFrame()

eval_repr = pd.DataFrame()

eval_words.columns = ['data', 'label']

for data, model in zip(args.datafiles, args.modelfiles):
    for prob in args.probs:
        end = f"{int(prob*100)}-{int((1-prob)*100)}"
        m_name = f"{model}{end}"
        print(f"{data}: {m_name}")
        
        dataset = utils.TextDataset.load_dataset_and_make_vectorizer(
                args.csv + data,
                p=prob, seed=args.seed)
        vectorizer = dataset.get_vectorizer()
        
        test_df = dataset.test_df
        
        lstm_model = torch.load(args.model_save_file + m_name + ".pt")
        
        tsne = TSNE(n_components=2, n_jobs=-1, random_state=args.seed)
        
        repr_df = get_hidden_representation(lstm_model, vectorizer,
                                            test_df, args.device)
        
        repr_df[['tsne1','tsne2']] = tsne.fit_transform(repr_df.iloc[:, 3:])
        repr_df['model'] = end
        repr_df['dataset'] = data
        
        hidden_repr = pd.concat([hidden_repr, repr_df], ignore_index=True)
        
        eval_df = get_hidden_representation(lstm_model, vectorizer,
                                            eval_words, args.device)
        
        eval_df[['tsne1','tsne2']] = tsne.fit_transform(eval_df.iloc[:, 2:])
        eval_df['model'] = end
        eval_df['dataset'] = data
        
        eval_repr = pd.concat([eval_repr, eval_df], ignore_index=True)
        
g = sns.FacetGrid(hidden_repr, col="model", row="dataset", hue="label")
g.map(sns.scatterplot, "tsne1", "tsne2", alpha=0.7)
g.add_legend()

g = sns.FacetGrid(eval_repr, col="model", row="dataset", hue="label")
g.map(sns.scatterplot, "tsne1", "tsne2", alpha=0.7)
g.add_legend()
