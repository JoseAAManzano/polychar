# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 11:39:59 2020

@author: josea
"""

import string
import math
from collections import Counter
from itertools import permutations

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
        
        self.vocab = list(string.ascii_lowercase) + [SOS_token, EOS_token]
        self.data = self._preprocess(data, n)
        self.ngrams = self._split_and_count(self.data, self.n)

        self.model = self._create_model()
        
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
        for perm in permutations(string.ascii_lowercase, n):
            cntr[perm] = 0
        return cntr
    
    def _create_model(self):
        if self.n == 1:
            s = sum(self.ngrams.values())
            return Counter({gram:count/s for gram, count in self.ngrams.items()})
        else:
            vocab_size = len(self.vocab)
            
            ret = self.ngrams
            
            m_grams = self._split_and_count(self.data, self.n-1)
            
            for n_gram, n_count in self.ngrams.items():
                m_gram = n_gram[:-1]
                m_count = m_grams[m_gram]
                ret[n_gram] = math.log((n_count + self.laplace) /\
                    (m_count + self.laplace * vocab_size))
            
            return ret
        
    
    def get_single_probability(self, word, log=False):
        if isinstance(word, str):
            word = self.process_word(word, self.n)
        prob = 0.0 if log else 1.0
        for ngram in self._split_word(word, self.n):
            p = self.model[ngram]
            if log:
                prob += p
            else:
                prob *= math.exp(p)
        return prob
    
    def get_distribution_from_context(self, context: tuple):
        pass
        
    def perplexity(self, test):
        test_tokens = self._preprocess(test, self.n)
        # test_ngrams = nltk.ngrams(test_tokens, self.n)
        N = len(test_tokens)
        
        probs = 0.0
        for word in test_tokens:
            probs += self.get_single_probability(word, log=True) / len(word)
        
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

eng = CharNGram(eng_train, 3)
esp = CharNGram(esp_train, 2)

print(eng.get_perplexity(eng_test))
print(eng.get_perplexity(esp_test))

print(esp.get_perplexity(eng_test))
print(esp.get_perplexity(esp_test))