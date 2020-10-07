# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 17:23:28 2020

@author: josea
"""
#%% Imports
import torch
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import json

#%% Helper functions
# def encode(st, stoi):
#     """
#     Output: List of <SOS> + char indices + <EOS>
#     """
#     return [0] + [stoi[c] for c in st] + [1]

# def letter2onehot(letter, n_letters, stoi):
#     '''
#     Returns one-hot encoding of a letter in shape (1, n_letters)
#     '''
#     onehot = torch.zeros(1, n_letters)
#     onehot[0][stoi[letter]] = 1
#     return onehot


# def line2tensor(st, n_letters):
#     '''
#     Returns one-hot Tensor of a string in shape (len(st), 1, n_letters)
#     '''
#     tensor = torch.zeros(len(st), 1, n_letters)
#     for i, l in enumerate(st):
#         tensor[i][0][l] = 1
#     return tensor

# def randomExample(data, n_letters, stoi, device, langs, ref='ESP', p=[0.5, 0.5]):
#     lang = np.random.choice(langs, p=p)
#     word = np.random.choice(data[lang], size=1)[0]
#     word = encode(word, stoi)
#     in_ = line2tensor(word[:-1], n_letters).to(device)
#     out_ = torch.LongTensor(word[1:]).to(device)
#     lang_ = torch.tensor(0.0 if lang == ref else 1.0).view(1, -1).to(device)
#     return in_, out_, lang_

def activ2color(value):
    colors = ['#85c2e1', '#89c4e2', '#95cae5', '#99cce6', '#a1d0e8'
    		'#b2d9ec', '#baddee', '#c2e1f0', '#eff7fb', '#f9e8e8',
    		'#f9e8e8', '#f9d4d4', '#f9bdbd', '#f8a8a8', '#f68f8f',
    		'#f47676', '#f45f5f', '#f34343', '#f33b3b', '#f42e2e']
    value = int((value*100)/5)
    return colors[value]

def generate_batches(dataset, batch_size, shuffle=True,
                     drop_last=True, device="cpu"):
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                             shuffle=shuffle, drop_last=drop_last)
    
    for data_dict in dataloader:
        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name].to(device)
        yield out_data_dict
        
def set_all_seeds(seed, device):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == torch.device('cuda:0'):
        torch.cuda.manual_seed_all(seed)
        
#%% Helper classes
class Vocabulary(object):
    def __init__(self, stoi=None, add_SOS=True, add_EOS=True):
        if stoi is None:
            stoi = {}
        self._stoi = stoi
        self._itos = {i:s for s,i in self._stoi.items()}
        
        self._add_SOS = add_SOS
        self._SOS_token = "<SOS>"
        self._add_EOS = add_EOS
        self._EOS_token = "<EOS>"
        
        self._SOS_idx = -1
        self._EOS_idx = -99
        
        if self._add_SOS:
            self._SOS_idx = self.add_token(self._SOS_token)
        if self._add_EOS:
            self._EOS_idx = self.add_token(self._EOS_token)
            
    def to_dict(self):
        """
        Returns vocabulary dictionary
        """
        return {
            "stoi": self._stoi,
            "itos": self._itos,
            "SOS_token": self._SOS_token,
            "EOS_token": self._EOS_token
            }

    @classmethod
    def from_dict(cls, contents):
        """
        Instantiates vocabulary from dictionary
        """
        return cls(**contents)
    
    def add_token(self, token):
        try:
            idx = self._stoi[token]
        except KeyError:
            idx = len(self._stoi)
            self._stoi[token] = idx
            self._itos[idx] = token
        return idx
    
    def add_many(self, tokens):
        return [self.add_token(token) for token in tokens]
    
    def token2idx(self, token):
        return self._stoi[token]
    
    def idx2token(self, idx):
        if idx not in self._itos:
            raise KeyError(f"Index {idx} not in Vocabulary")
        return self._itos[idx]
    
    def __str__(self):
        return f"<Vocabulary(size={len(self)}>"
    
    def __len__(self):
        return len(self._stoi)
    
class TextDataset(Dataset):
    def __init__(self, df, vectorizer):
        self.df = df
        self._vectorizer = vectorizer
        
        self.train_df = self.df[self.df.split=='train']
        self.train_size = len(self.train_df)
        
        self.val_df = self.df[self.df.split=='val']
        self.val_size = len(self.val_df)
        
        self.test_df = self.df[self.df.split=='test']
        self.test_size = len(self.test_df)
        
        self._lookup_dict = {
            "train": (self.train_df, self.train_size),
            "val": (self.val_df, self.val_size),
            "test": (self.test_df, self.test_size)
            }
        
        self.set_split('train')
        
        # Handles imbalanced labels
        labels = df.label.value_counts().to_dict()
        def sort_key(item):
            return self._vectorizer.label_vocab.token2idx(item[0])
        sorted_cnts = sorted(labels.items(), key=sort_key)
        freqs = [cnt for _, cnt in sorted_cnts]
        self.label_weights = 1.0 / torch.tensor(freqs, dtype=torch.float32)
        
    @classmethod
    def load_dataset_and_make_vectorizer(cls, csv, split="char"):
        df = pd.read_csv(csv)
        train_df = df[df.split=='train']
        return cls(df, Vectorizer.from_df(train_df, split=split))
    
    @classmethod
    def load_dataset_and_load_vectorizer(cls, csv, vectorizer_path):
        df = pd.read_csv(csv)
        with open(vectorizer_path) as f:
            vectorizer = Vectorizer.from_dict(json.load(f))
        return cls(df, vectorizer)
    
    def save_vectorizer(self, vectorizer_path):
        with open(vectorizer_path, 'w') as f:
            json.dump(self._vectorizer.to_dict(), f)
    
    def get_vectorizer(self):
        return self._vectorizer
    
    def set_split(self, split="train"):
        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]
    
    def __len__(self):
        return self._target_size
    
    def __getitem__(self, index):
        row = self._target_df.iloc[index]
        
        vector = self._vectorizer.vectorize(row.text)
        
        label = self._vectorizer.label_vocab.lookup_token(row.label)
        
        return {'X': vector, 'target': label}
    
    def get_num_batches(self, batch_size):
        return len(self) // batch_size

class Vectorizer(object):
    def __init__(self, data_vocab, label_vocab):
        self.data_vocab = data_vocab
        self.label_vocab = label_vocab
        
    def vectorize(self, data, add_SOS=True, add_EOS=True):
        one_hot = torch.zeros(len(self.data_vocab), dtype=torch.float32)
        if add_SOS:
            data = [self.data_vocab._SOS_token] + data
        if add_EOS:
            data = data + [self.data_vocab._EOS_token]
        for token in data:
            one_hot[self.data_vocab.token2idx(token)] = 1
        return one_hot

    @classmethod
    def from_df(cls, df, split="char"):
        data_vocab = Vocabulary()
        label_vocab = Vocabulary()
        
        for i, row in df.iterrows():
            if split == "char":
                data_vocab.add_many([c for c in row.data])
            else:
                data_vocab.add_many([c for c in row.data.split(split)])
            
            label_vocab.add_token(row.label)
        
        return cls(data_vocab, label_vocab)
    
    @classmethod
    def from_dict(cls, contents):
        data_vocab = Vocabulary.from_dict(contents['data_vocab'])
        label_vocab = Vocabulary.from_dict(contents['label_vocab'])
        return cls(data_vocab, label_vocab)
    
    def to_dict(self):
        return {
            'data_vocab': self.data_vocab.to_dict(),
            'label_vocab': self.label_vocab.to_dict()
            }
        