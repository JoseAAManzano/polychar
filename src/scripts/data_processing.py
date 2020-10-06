# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 15:31:40 2020

@author: josea
"""
import os, unicodedata, re
import pandas as pd
import random

random.seed(4004)

path = os.getcwd()
os.chdir(path)

file_path = "../../data/"
target_path = "../../processed_data/"

esp = pd.read_csv(os.path.join(file_path, 'ESP.csv'), encoding='utf-8')
esp = esp[esp.percent_total > 90]
eus = pd.read_csv(os.path.join(file_path, 'EUS.txt'), sep='\t',
                    encoding='utf-8', header=None)
cols = ['word', 'freq']
eus.columns = cols

esp = esp[['spelling', 'freq', 'zipf']]
esp.columns = cols + ['zipf']

eus['len'] = eus['word'].apply(lambda x: len(x))
esp['len'] = esp['word'].apply(lambda x: len(x))

eus = eus[(eus.freq > 0.1) & (eus.freq < 1000)]
esp = esp[(esp.freq > 0.1) & (esp.freq < 1000)]

def remove_outliers(data):
    data['zfreq'] = (data.freq - data.freq.mean())/data.freq.std()
    print(min(data.zfreq), max(data.zfreq), data.freq.mean(), data.freq.std())
    return data[(data.zfreq > -3) & (data.zfreq < 3)]

for _ in range(1):
    eus = remove_outliers(eus)
    esp = remove_outliers(esp)

eus = eus[(eus['len'] >= 3) & (eus['len'] <= 10)]
esp = esp[(esp['len'] >= 3) & (esp['len'] <= 10)]

def preprocess(st):
    st = ''.join(c for c in unicodedata.normalize('NFD', st)
                   if unicodedata.category(c) != 'Mn')
    st = re.sub(r"[^a-zA-Z]", r"", st)
    return st.lower()

esp_words = [preprocess(st) for st in esp.word.values]
eus_words = [preprocess(st) for st in eus.word.values]

random.shuffle(esp_words)
random.shuffle(eus_words)

esp_words = esp_words[:min(len(esp_words), len(eus_words))]
eus_words = eus_words[:min(len(esp_words), len(eus_words))]

idx1 = int(len(esp_words)*0.7)
idx2 = int(len(esp_words)*0.15)
idx3 = int(len(esp_words)*0.15)

idx1 += len(esp_words) - idx1 - idx2 - idx3

assert idx1 + idx2 + idx3 == len(esp_words)

data = pd.DataFrame(columns=['data', 'label', 'split'])
data['data'] = esp_words + eus_words
data['label'] = ['ESP'] * len(esp_words) + ['EUS'] * len(eus_words)
splits = ['train']*idx1 + ['val']*idx2 + ['test']*idx3 + ['train']*idx1 + ['val']*idx2 + ['test']*idx3
data['split'] = splits

data.to_csv(os.path.join(target_path, 'data.csv'))
