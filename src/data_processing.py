# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 15:31:40 2020

@author: josea
"""
import os, unicodedata, re
import pandas as pd
import numpy as np

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

eus['zipf'] = np.log10((eus.freq*10e5 + 1)/(201336 + 159))
print(max(esp.zipf), min(esp.zipf))
print(max(eus.zipf), min(eus.zipf))

res_eus = eus[(eus.zipf >= 1) & (eus.zipf <= 3)]
res_esp = esp[(esp.zipf >= 3) & (esp.zipf <= 5)]

def preprocess(st):
    st = ''.join(c for c in unicodedata.normalize('NFD', st)
                   if unicodedata.category(c) != 'Mn')
    st = re.sub(r"[^a-zA-Z]", r"", st)
    return st.lower()

esp_words = [preprocess(st) for st in res_esp.word.values]
eus_words = [preprocess(st) for st in res_eus.word.values]

with open(os.path.join(target_path, 'ESP.txt'), 'w') as f:
    for l in esp_words:
        f.write(l+'\n')
        
with open(os.path.join(target_path, 'EUS.txt'), 'w') as f:
    for l in eus_words:
        f.write(l+'\n')