# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 11:24:40 2020

TODO: Documentation

@author: josea
"""
import torch
import Vocabulary

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
                data_vocab.add_many([c for c in row.text])
            else:
                data_vocab.add_many([c for c in row.text.split(split)])
            
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
        