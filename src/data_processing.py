# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 15:31:40 2020

@author: josea
"""
import os, unicodedata, re
import pandas as pd
from random import shuffle

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

shuffle(esp_words)
shuffle(eus_words)

esp_words = esp_words[:min(len(esp_words), len(eus_words))]
eus_words = eus_words[:min(len(esp_words), len(eus_words))]

idx1 = int(len(esp_words)*0.7)
idx2 = int(len(esp_words)*0.85)

train_esp = esp_words[:idx1]
val_esp = esp_words[idx1:idx2]
test_esp = esp_words[idx2:]

train_eus = eus_words[:idx1]
val_eus = eus_words[idx1:idx2]
test_eus = eus_words[idx2:]

def save(file_name, data):
    with open(os.path.join(target_path, f"{file_name}.txt"), 'w') as f:
        for l in data:
            f.write(l + '\n')

save('train_esp', train_esp)
save('val_esp', val_esp)
save('test_esp', test_esp)
save('ESP', esp_words)

save('train_eus', train_eus)
save('val_eus', val_eus)
save('test_eus', test_eus)
save('EUS', eus_words)
