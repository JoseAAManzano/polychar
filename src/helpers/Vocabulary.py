# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 11:10:54 2020

TODO: Documentation

@author: josea
"""
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
    