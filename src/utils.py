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
        
def make_train_state(args):
    return {
        'learning_rate': args.learning_rate,
        'epoch_idx': 0,
        'train_loss': [],
        'train_acc': [],
        'train_lang_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_lang_acc': [],
        'test_loss': -1,
        'test_acc': -1,
        'test_lang_acc': -1,
        }

def compute_lang_accuracy(y_pred, y_target):
    preds = torch.sigmoid(y_pred)
    n_correct = torch.eq(preds > 0.5, y_target).sum().item()
    return n_correct / len(preds) * 100

def compute_accuracy(y_pred, y_true, mask_index):
    y_pred, y_true = normalize_sizes(y_pred, y_true)

    _, y_pred_indices = y_pred.max(dim=1)
    
    correct_indices = torch.eq(y_pred_indices, y_true).float()
    valid_indices = torch.ne(y_true, mask_index).float()
    
    n_correct = (correct_indices * valid_indices).sum().item()
    n_valid = valid_indices.sum().item()

    return n_correct / n_valid * 100

def normalize_sizes(y_pred, y_true):
    """Normalize tensor sizes
    
    Args:
        y_pred (torch.Tensor): the output of the model
            If a 3-dimensional tensor, reshapes to a matrix
        y_true (torch.Tensor): the target predictions
            If a matrix, reshapes to be a vector
    """
    if len(y_pred.size()) == 3:
        y_pred = y_pred.contiguous().view(-1, y_pred.size(2))
    if len(y_true.size()) == 2:
        y_true = y_true.contiguous().view(-1)
    return y_pred, y_true

def print_state_dict(train_state):
    print((f"Epoch: {train_state['epoch_idx'] + 1} | "
           f"train_loss: {train_state['train_loss'][-1]:.4f} | "
           f"val_loss: {train_state['val_loss'][-1]:.4f}\n"
           f"train_acc_chars: {train_state['train_acc'][-1]:.2f} | "
           f"train_acc_lang: {train_state['train_lang_acc'][-1]:.2f}\n"
           f"val_acc_chars: {train_state['val_acc'][-1]:.2f} | "
           f"val_acc_lang: {train_state['val_lang_acc'][-1]:.2f}\n"))
    

#%% Helper classes
class Vocabulary(object):
    def __init__(self, stoi=None, SOS="<s>", EOS="</s>", PAD="<p>"):
        """Supports either char or word level vocabulary"""
        if stoi is None:
            stoi = {}
        self._stoi = stoi
        self._itos = {i:s for s,i in self._stoi.items()}
        
        self._SOS_token = SOS
        self._EOS_token = EOS
        self._PAD_token = PAD
        
        if self._SOS_token is not None:
            self.SOS_idx = self.add_token(self._SOS_token)
        if self._EOS_token is not None:
            self.EOS_idx = self.add_token(self._EOS_token)
        if self._PAD_token is not None:
            self.PAD_idx = self.add_token(self._PAD_token)
            
    def to_dict(self):
        """
        Returns vocabulary dictionary
        """
        return {
            "stoi": self._stoi,
            "itos": self._itos,
            "SOS_token": self._SOS_token,
            "EOS_token": self._EOS_token,
            "PAD_token": self._PAD_token
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
        return f"<Vocabulary(size={len(self)})>"
    
    def __len__(self):
        return len(self._stoi)


class Vectorizer(object):
    def __init__(self, data_vocab, label_vocab):
        self.data_vocab = data_vocab
        self.label_vocab = label_vocab
        
    def vectorize(self, data, vector_len=-1):
        indices = [self.data_vocab.SOS_idx]
        indices.extend(self.data_vocab.token2idx(t) for t in data)
        indices.append(self.data_vocab.EOS_idx)
    
        # if add_SOS:
        #     indices = [self.data_vocab._SOS_token] + indices
        # if add_EOS:
        #     indices = indices + [self.data_vocab._EOS_token]
        
        if vector_len < 0:
            vector_len = len(indices)-1
        
        from_vector = torch.empty(vector_len, len(self.data_vocab), dtype=torch.float32)
        from_indices = indices[:-1]
        # Add pre-padding
        from_vector[:-len(from_indices)] = self.onehot([self.data_vocab.PAD_idx])
        from_vector[-len(from_indices):] = self.onehot(from_indices)
        
        to_vector = torch.empty(vector_len, dtype=torch.int64)
        to_indices = indices[1:]
        to_vector[:-len(to_indices)] = self.data_vocab.PAD_idx
        to_vector[-len(to_indices):] = torch.LongTensor(to_indices)
        
        return from_vector, to_vector
    
    def onehot(self, indices):
        onehot = torch.zeros(len(indices), len(self.data_vocab), dtype=torch.float32)
        
        for i, idx in enumerate(indices):
            onehot[i][idx] = 1.
        return onehot
        
    @classmethod
    def from_df(cls, df, split="char"):
        data_vocab = Vocabulary()
        label_vocab = Vocabulary(SOS=None, EOS=None, PAD=None)
        
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
        

class TextDataset(Dataset):
    def __init__(self, df, vectorizer, p=None):
        self.df = df
        self._vectorizer = vectorizer
        
        self.train_df = self.df[self.df.split=='train']
        if p is not None:
            labs = self.train_df.label.unique()
            tmp = pd.DataFrame()
            if isinstance(p, float):
                p = [p, 1-p]
            for frac, l in zip(p, labs):
                dat = self.train_df[self.train_df.label == l]
                tmp = pd.concat([tmp, dat.sample(frac=frac)])
            self.train_df = tmp
        self.train_size = len(self.train_df)
        
        self.val_df = self.df[self.df.split=='val']
        self.val_size = len(self.val_df)
        
        self.test_df = self.df[self.df.split=='test']
        self.test_size = len(self.test_df)
        
        self._max_seq_len = max(map(len, self.df.data)) + 1 # Adding either SOS/EOS
        
        self._lookup_dict = {
            "train": (self.train_df, self.train_size),
            "val": (self.val_df, self.val_size),
            "test": (self.test_df, self.test_size)
            }
        
        self.set_split('train')
        
        # Handles imbalanced labels
        labels = self.train_df.label.value_counts().to_dict()
        def sort_key(item):
            return self._vectorizer.label_vocab.token2idx(item[0])
        sorted_cnts = sorted(labels.items(), key=sort_key)
        freqs = [cnt for _, cnt in sorted_cnts]
        self.label_weights = 1.0 / torch.tensor(freqs, dtype=torch.float32)
        
    @classmethod
    def load_dataset_and_make_vectorizer(cls, csv, split="char", p=None):
        df = pd.read_csv(csv)
        train_df = df[df.split=='train']
        return cls(df, Vectorizer.from_df(train_df, split=split), p)
    
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
        
        from_vector, to_vector = self._vectorizer.vectorize(row.data,
                                                            self._max_seq_len)
        
        label = self._vectorizer.label_vocab.token2idx(row.label)
        
        return {'X': from_vector, 'Y': to_vector, 'label': label}
    
    def get_num_batches(self, batch_size):
        return len(self) // batch_size

