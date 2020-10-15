# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 11:39:59 2020

@author: josea
"""

import string
import math
from collections import Counter
from itertools import product
import pandas as pd

class CharNGram(object):
    """A character n-gram model trained on a list of words.
    
    Concepts from Jurafsky, D., & Martin, J.H. (2019). Speech and Language
    Processing. Stanford Press. https://web.stanford.edu/~jurafsky/slp3/
    
    TODO ADD DOC
    """
    def __init__(self, data, n, laplace=1, SOS_token='<s>', EOS_token='</s>'):
        """
        Data should be iterable of words
        """
        assert (n > 0), n
        self.n = n
        self.laplace = laplace
        self.SOS_token = SOS_token
        self.EOS_token = EOS_token
        self.data = data
        self.vocab = list(string.ascii_lowercase) + [SOS_token, EOS_token]
        self.processed_data = self._preprocess(self.data, n)
        self.ngrams = self._split_and_count(self.processed_data, self.n)
        self.model = self._smooth()
        
    def _preprocess(self, data, n):
        new_data = []
        for word in data:
            new_data.append(self.process_word(word, n))
        return new_data
    
    def process_word(self, word, n):
        pad = (n-1) if n > 1 else 1
        return [self.SOS_token] * pad +\
                    list(word.lower()) +\
                    [self.EOS_token] * pad
    
    def _split_word(self, word, n):
        for i in range(len(word) - n + 1):
            yield tuple(word[i:i+n])
    
    def _split_and_count(self, data, n):
        cntr = self._initialize_counts(n)
        for word in data:
            for ngram in self._split_word(word, n):
                cntr[ngram] += 1
        return cntr
    
    def _initialize_counts(self, n):
        def is_plausible(permutation):
            if self.SOS_token not in permutation or self.EOS_token not in permutation:
                return True
            n = len(permutation)
            
            if self.EOS_token in permutation[:-1]: return False
            
            flg = False
            cnt = 0
            for i in range(n-1, -1, -1):
                if self.SOS_token == permutation[i]:
                    flg = True
                    cnt += 1
                else:
                    if flg: return False
            if cnt == n: return False
            
            flg = False
            cnt = 0
            for i in range(n):
                if self.EOS_token == permutation[i]:
                    flg = True
                    cnt += 1
                else:
                    if flg: return False
            return True
            if cnt == n : return False
        
        cntr = Counter()
        for perm in product(self.vocab, repeat=n):
            if is_plausible(perm):
                cntr[tuple(perm)] = 0
        return cntr
    
    def _smooth(self):
        if self.n == 1:
            s = sum(self.ngrams.values())
            return Counter({key: val/s for key, val in self.ngrams.items()})
        else:
            vocab_size = len(self.vocab)-1
            
            ret = self.ngrams.copy()
            
            m = self.n - 1
            m_grams = self._split_and_count(self.processed_data, m)
            
            for ngram, value in self.ngrams.items():
                m_gram = ngram[:-1]
                m_count = m_grams[m_gram]
                ret[ngram] = (value + self.laplace) /\
                            (m_count + self.laplace * vocab_size)
            
            return ret
    
    def to_txt(self, filepath):
        with open(filepath, 'w') as file:
            for ngram, value in self.model.items():
                file.write(f"{' '.join(ngram)}\t{value}\n")
    
    def from_txt(self, filepath):
        with open(filepath, 'r') as file:
            data = file.readlines()
        model = Counter()
        for ngram, value in data.split('\t'):
            model[tuple(ngram.split(' '))] = value
        return model
    
    def to_df(self):
        """Do not use with >= 4-grams"""
        idxs, cols = set(), set()
        for k in self.model.keys():
            idxs.add(' '.join(k[:-1]))
            cols.add(k[-1])
        df = pd.DataFrame(data=0.0,
                          index=sorted(list(idxs)),
                          columns=sorted(list(cols)))
        for ngram, value in self.model.items():
            cntx = ' '.join(ngram[:-1])
            trgt = ngram[-1]
            df.loc[cntx, trgt] = value
        return df.fillna(0.0).drop(self.SOS_token, axis=1)
    
    def to_csv(self):
        pass
    
    def get_single_probability(self, word, log=False):
        if isinstance(word, str):
            word = self.process_word(word, self.n)
        prob = 0.0 if log else 1.0
        for ngram in self._split_word(word, self.n):
            if ngram not in self.model.keys():
                print(ngram)
            p = self.model[ngram]
            if log:
                prob += math.log(p)
            else:
                prob *= p
        return prob
    
    def get_distribution_from_context(self, context):
        pad = self.n - 1 if self.n > 1 else 1
        context = [self.SOS_token] * pad + list(context)
        
    
    def compare_likelihood(self, data1, data2):
        probs1 = sum([self.get_single_probability(w) for w in data1])
        probs2 = sum([self.get_single_probability(w) for w in data2])
        likelihood = probs1 / probs2
        if likelihood > 1:
            ret = f"Data1 is {likelihood:.2f} times more likely than Data2 under this model"
        else:
            ret = f"Data2 is {1/likelihood:.2f} times more likely than Data1 under this model"
        return ret
    
    def perplexity(self, test):
        test_tokens = self._preprocess(test, self.n)
        N = len(test_tokens)
        
        probs = 0.0
        for word in test_tokens:
            probs += self.get_single_probability(word, log=True) / len(word)
        
        return math.exp((-1/N) * probs)
    
    def __len__(self):
        return len(self.ngrams)
    
    def __str__(self):
        return f"<{self.n}-gram model(size={len(self)})>"
    
    
idxs = int(len(eng_words) * 0.8)

eng_train = eng_words[:idxs]
eng_test = eng_words[idxs:]

model = CharNGram(eng_train, 2)

df = model.to_df()
