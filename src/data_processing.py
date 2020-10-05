# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 15:31:40 2020

@author: josea
"""
import os, unicodedata, re
import pandas as pd

path = os.getcwd()
os.chdir(path)

file_path = "../data/"
target_path = "../processed_data/"

esp = pd.read_csv(os.path.join(file_path, 'ESP.csv'), encoding='utf-8')
eus = pd.read_csv(os.path.join(file_path, 'EUS.txt'), sep='\t',
                    encoding='utf-8', header=None)
cols = ['word', 'freq']
eus.columns = cols

esp = esp[['spelling', 'freq', 'zipf']]
esp.columns = cols + ['zipf']

eus['len'] = eus['word'].apply(lambda x: len(x))
esp['len'] = esp['word'].apply(lambda x: len(x))

eus = eus[(eus['len'] >= 4) & (eus['len'] <= 8)]
esp = esp[(esp['len'] >= 4) & (esp['len'] <= 8)]

def preprocess(st):
    st = ''.join(c for c in unicodedata.normalize('NFD', st)
                   if unicodedata.category(c) != 'Mn')
    st = re.sub(r"[^a-zA-Z]", r"", st)
    return st.lower()

esp_words = [preprocess(st) for st in esp.word.values]
eus_words = [preprocess(st) for st in eus.word.values]

with open(os.path.join(target_path, 'ESP.txt'), 'w') as f:
    for l in esp_words:
        f.write(l+'\n')
        
with open(os.path.join(target_path, 'EUS.txt'), 'w') as f:
    for l in eus_words:
        f.write(l+'\n')