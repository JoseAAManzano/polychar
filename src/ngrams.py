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
    """A character n-gram table trained on a list of words.
    
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
        self.vocab = list(string.ascii_lowercase) + [EOS_token]
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
        cntr = Counter()
        for perm in product(string.ascii_lowercase, repeat=n):
            cntr[perm] = 0
        return cntr
    
    def _smooth(self):
        if self.n == 1:
            s = sum(self.ngrams.values())
            return Counter({key: val/s for key, val in self.ngrams.items()})
        else:
            vocab_size = len(self.vocab)
            
            ret = self.ngrams.copy()
            
            m = self.n - 1
            m_grams = self._split_and_count(self.processed_data, m)
            
            for ngram, value in self.ngrams.items():
                m_gram = ngram[:-1]
                m_count = m_grams[m_gram]
                ret[ngram] = (value + self.laplace) /\
                            (m_count + self.laplace * vocab_size)
            
            return ret
    
    def to_csv(self, filepath):
        if self.n == 1:
            df = pd.DataFrame({k:[v] for k,v in self.model.items()})
        else:
            idxs = sorted(set([' '.join(k[:-1]) for k in self.model.keys()]))
            cols = sorted(set([k[-1] for k in self.model.keys()]))
            df = pd.DataFrame(data=0.0, index=idxs, columns=cols)
            for ngram, value in self.model.items():
                context = ' '.join(ngram[:-1])
                target = ngram[-1]
                df.loc[context, target] = value
            df = df.fillna(0.0)
        df.to_csv(filepath, encoding='utf-8')
    
    def get_single_probability(self, word, log=False):
        if isinstance(word, str):
            word = self.process_word(word, self.n)
        prob = 0.0 if log else 1.0
        for ngram in self._split_word(word, self.n):
            p = self.model.loc[' '.join(ngram[:-1]), ngram[-1]]
            if log:
                prob += p
            else:
                prob *= math.exp(p)
        return prob / len(word)
    
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
            probs += self.get_single_probability(word, log=True)
        
        return math.exp((-1/N) * probs)
    
    def __len__(self):
        return len(self.ngrams)
    
    def __str__(self):
        return f"<{self.n}-gram model(size={len(self)})>"
 
idx1 = int(len(eng_words) * 0.80)

eng_train = eng_words[:idx1]
eng_test = eng_words[idx1:]

esp_train = esp_words[:idx1]
esp_test = esp_words[idx1:]

eus_train = eus_words[:idx1]
eus_test = eus_words[idx1:]

eng = CharNGram(eng_train, 3)
df = eng.to_csv('eng.csv')

# print(eng.get_single_probability('alien'))
