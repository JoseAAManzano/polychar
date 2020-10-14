# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 15:31:40 2020

@author: josea
"""
import os, unicodedata, re
import pandas as pd
import random
import numpy as np

random.seed(404)

path = os.getcwd()
os.chdir(path)

file_path = "../../data/"
target_path = "../../processed_data/"

esp = pd.read_csv(os.path.join(file_path, 'ESP.csv'), encoding='utf-8')
esp = esp[esp.percent_total > 90]
eng = pd.read_csv(os.path.join(file_path, 'ENG.csv'), sep=',',
                    encoding='utf-8')
eng = eng[eng.accuracy > 0.9]

eus = pd.read_csv(os.path.join(file_path, 'EUS.txt'), sep='\t', header=None)
eus.columns = ['spelling', 'freq']
eus['len'] = eus['spelling'].apply(len)

#%% Normalizing eus data
eus['logfreq'] = np.log(eus.freq)

eus['zfreq'] = (eus.logfreq - eus.logfreq.mean())/eus.logfreq.std()
eus = eus[(eus.zfreq > 0.5)]

esp = esp[(esp.zipf >= 0.5)]
eng = eng[(eng.ZipfUS >= 0.5)]

esp = esp[(esp.len >= 3) & (esp.len <= 10)]
eng = eng[(eng.Length >= 3) & (eng.Length <=10)]
eus = eus[(eus.len >= 3) & (eus.len <=10)]

def preprocess(st):
    st = ''.join(c for c in unicodedata.normalize('NFD', st)
                    if unicodedata.category(c) != 'Mn')
    st = re.sub(r"[^a-zA-Z]", r"", st)
    return st.lower()

esp_words = list(set([preprocess(st) for st in esp.spelling]))
eng_words = list(set([preprocess(st) for st in eng.spelling]))
eus_words = list(set([preprocess(st) for st in eus.spelling]))

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

#%% First dataset
data = pd.DataFrame(columns=['data', 'label', 'split'])
data['data'] = esp_words + eng_words
data['label'] = ['ESP'] * len(esp_words) + ['ENG'] * len(eng_words)
splits = ['train']*idx1 + ['val']*idx2 + ['test']*idx3 + ['train']*idx1 + ['val']*idx2 + ['test']*idx3
data['split'] = splits

data.to_csv(os.path.join(target_path, 'ESP-ENG.csv'), index=False, encoding='utf-8')

#%% Second dataset
data = pd.DataFrame(columns=['data', 'label', 'split'])
data['data'] = esp_words + eus_words
data['label'] = ['ESP'] * len(esp_words) + ['EUS'] * len(eus_words)
splits = ['train']*idx1 + ['val']*idx2 + ['test']*idx3 + ['train']*idx1 + ['val']*idx2 + ['test']*idx3
data['split'] = splits

data.to_csv(os.path.join(target_path, 'ESP-EUS.csv'), index=False, encoding='utf-8')
