# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 15:31:40 2020

@author: josea
"""
import os
import re
import random
import unicodedata
import pandas as pd

from itertools import product

random.seed(404)

path = os.getcwd()
os.chdir(path)

file_path = "../../data/"
target_path = "../../processed_data/"

esp = pd.read_csv(os.path.join(file_path, 'ESP.csv'), encoding='utf-8')
eng = pd.read_csv(os.path.join(file_path, 'ENG.csv'), sep=',',
                  encoding='utf-8')

eus = pd.read_csv(os.path.join(file_path, 'EUS.txt'), sep='\t', header=None)
eus.columns = ['spelling', 'freq']
eus['len'] = eus['spelling'].apply(len)

# %% Normalizing eus data
eus = eus[(eus.freq > eus.freq.quantile(q=0.5))]

esp = esp[(esp.zipf > esp.zipf.quantile(q=0.5))]
eng = eng[(eng.ZipfUS > eng.ZipfUS.quantile(q=0.5))]

esp = esp[(esp.len >= 3) & (esp.len <= 10)]
eng = eng[(eng.Length >= 3) & (eng.Length <= 10)]
eus = eus[(eus.len >= 3) & (eus.len <= 10)]


def preprocess(st):
    st = ''.join(c for c in unicodedata.normalize('NFD', st)
                 if unicodedata.category(c) != 'Mn')
    st = re.sub(r"[^a-zA-Z]", r"", st)
    return st.lower()


esp_words = list(set([preprocess(st) for st in esp.spelling]))
eng_words = list(set([preprocess(st) for st in eng.spelling]))
eus_words = list(set([preprocess(st) for st in eus.spelling]))


def editDistance(word1, word2):
    '''
    Return minimum number of edits required to transform word1 into word2
    Edits include: deletion, insertion, replacement

    Uses memoization to speed up the process
    '''
    n1, n2 = len(word1), len(word2)
    memo = [[0]*(n2) for _ in range(n1)]

    def minDist(i, j):
        if i < 0:
            return j+1
        if j < 0:
            return i+1
        if memo[i][j]:
            return memo[i][j]
        if word1[i] == word2[j]:
            memo[i][j] = minDist(i-1, j-1)
            return memo[i][j]
        memo[i][j] = 1 + min(minDist(i, j-1),
                             minDist(i-1, j),
                             minDist(i-1, j-1))
        return memo[i][j]

    return minDist(n1-1, n2-1)


def get_num_cognates(vocab1, vocab2):
    cognates = 0
    for w1, w2 in product(vocab1, vocab2):
        if editDistance(w1, w2) == 1:
            cognates += 1
    return cognates

# print(get_num_cognates(esp_words, eng_words))
# print(get_num_cognates(esp_words, eus_words))
# print(get_num_cognates(eus_words, eng_words))


random.shuffle(esp_words)
random.shuffle(eng_words)
random.shuffle(eus_words)

esp_words = esp_words[:min(len(esp_words), len(eng_words), len(eus_words))]
eng_words = eng_words[:min(len(esp_words), len(eng_words), len(eus_words))]
eus_words = eus_words[:min(len(esp_words), len(eng_words), len(eus_words))]

idx1 = int(len(esp_words)*0.8)
idx2 = int(len(esp_words)*0.1)
idx3 = int(len(esp_words)*0.1)

idx1 += len(esp_words) - idx1 - idx2 - idx3

assert idx1 + idx2 + idx3 == len(esp_words)

# %% First dataset
data = pd.DataFrame(columns=['data', 'label', 'split'])
data['data'] = esp_words + eng_words
data['label'] = ['ESP'] * len(esp_words) + ['ENG'] * len(eng_words)
splits = ['train']*idx1 + ['val']*idx2 + ['test'] * \
    idx3 + ['train']*idx1 + ['val']*idx2 + ['test']*idx3
data['split'] = splits

data.to_csv(os.path.join(target_path, 'ESP-ENG.csv'),
            index=False, encoding='utf-8')

# %% Second dataset
data = pd.DataFrame(columns=['data', 'label', 'split'])
data['data'] = esp_words + eus_words
data['label'] = ['ESP'] * len(esp_words) + ['EUS'] * len(eus_words)
splits = ['train']*idx1 + ['val']*idx2 + ['test'] * \
    idx3 + ['train']*idx1 + ['val']*idx2 + ['test']*idx3
data['split'] = splits

data.to_csv(os.path.join(target_path, 'ESP-EUS.csv'),
            index=False, encoding='utf-8')
