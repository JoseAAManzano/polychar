# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 10:25:09 2020

TODO: Documentation

@author: josea
"""
import torch
from torch.utils.data import Dataset
import pandas as pd
import Vectorizer
import json

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
    def load_dataset_and_make_vectorizer(cls, csv):
        df = pd.read_csv(csv)
        train_df = df[df.split=='train']
        return cls(df, Vectorizer.from_df(train_df))
    
    @classmethod
    def load_dataset_and_load_vectorizer(cls, csv, vectorizer_path):
        df = pd.read_csv(csv)
        with open(vectorizer_path) as f:
            vectorizer = Vectorizer.from_dict(json.load(f))
        return cls(df, vectorizer)
    
    def save_vectorizer(self, vectorizer_path):
        with open(vectorizer_path) as f:
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

            