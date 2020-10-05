# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 10:54:40 2020

@author: josea
"""
#%% Imports
import torch
import torch.nn as nn
import numpy as np
from polychar import PolyChar

from io import open
import os, random, string
from glob import glob
from datetime import datetime
import matplotlib.pyplot as plt
import helpers

#%% Set-up data
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
path = os.getcwd()
os.chdir(path)

#%% Read the data
train_path = "../processed_data/train/"
val_path = "../processed_data/val/"
test_path = "../processed_data/test"

files = glob(train_path + '*.txt')

train_data = {}
langs = []
for f in files:
    lang = os.path.basename(f).split('.')[0]
    langs.append(lang)
    train_data[lang] = open(f, encoding='utf-8').read().strip().split('\n')
    
all_letters = string.ascii_lowercase

stoi = {'<SOS>':0, '<EOS>':1}
for l in all_letters:
    stoi[l] = len(stoi)
    
itos = {i:c for c,i in stoi.items()}

n_letters = len(stoi)

#%% Define model
n_hidden = 128
n_layers = 2
model = PolyChar(n_letters,
                 n_hidden,
                 n_layers,
                 n_letters).to(device)

lr = 0.001
criterion = nn.CrossEntropyLoss()
criterion_lang = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#%% Training loop
def train(input_tensor, output_tensor, cat_tensor):
    output_tensor = output_tensor.view(-1, 1)
    hidden = model.initHidden(device)
    
    optimizer.zero_grad()
    
    loss = 0
    
    for i in range(input_tensor.size(0)):
        out, hidden, lang = model(input_tensor[i], hidden)
        loss += criterion(out, output_tensor[i])
        loss += criterion_lang(lang.view(1,-1), cat_tensor.float())
    
    loss.backward()
    optimizer.step()
    
    return out, loss.item()/input_tensor.size(0)

def evaluate(input_tensor, output_tensor, cat_tensor):
    with torch.no_grad():
        output_tensor = output_tensor.view(-1, 1)
        hidden = model.initHidden(device)
        
        acc = 0
        loss = 0
        
        for i in range(input_tensor.size(0)):
            out, hidden = model(input_tensor[i], cat_tensor, hidden)
            loss += criterion(out, output_tensor[i])
            acc += (out.topk(1)[1][0] == output_tensor[i]).item()
            
        return loss.item()/input_tensor.size(0), acc/output_tensor.size(0)

n_epochs = 10000
log_every = 1000
all_losses = []
total_loss = 0

t0 = datetime.now()

for it in range(1, n_epochs + 1):
    example = helpers.randomExample(train_data, n_letters, stoi, device, langs)
    output, loss = train(*example) # Unpack to function *args
    total_loss += loss
    
    if it % log_every == 0:
        dt = datetime.now() - t0
        print(f"Time elapsed: {dt.total_seconds():.2f}s")
        print(f"Iteration: {it} | Current avg. loss: {total_loss/log_every:.4f}")
        t0 = datetime.now()
        # for _ in range(5):
        #     l, a = evaluate(*randomExample())
        #     print(f"val_loss: {l:.4f}, val_acc: {a:.2f}")
        all_losses.append(total_loss / log_every)
        total_loss = 0
        # torch.save(model, os.path.join(os.getcwd(),
        #                                f'models/checkpoints/bil-50_50_{it}.pt'))
        

#%% Plot loss
plt.plot(all_losses)
plt.show()

#%% Save model weights
#torch.save(model.state_dict(), '/models/bil-50_50.pt')
torch.save(model, os.path.join(os.getcwd(), 'models/bil-50_50.pt'))

#%% Evaluate the model on unseen words
def eval_model(input_tensor, output_tensor, cat_tensor):
    with torch.no_grad():
        output_tensor = output_tensor.view(-1, 1)
        hidden = model.initHidden(device)
        
        loss = 0
    
    for i in range(input_tensor.size(0)):
        out, hidden = model(input_tensor[i], cat_tensor, hidden)
        loss += criterion(out, output_tensor[i])
    
    return loss.item()/input_tensor.size(0)

eval_loss = 0
for _ in range(100):
    example = helpers.randomExample(train_data, n_letters, stoi, device, langs)
    eval_loss += eval_model(*example)

print(eval_loss / 100)


#%% Examine weights
fc1_weights = model.fc.weight
_, indices = torch.sort(fc1_weights, dim=0, descending=True)

# Top 20 characters
for i in range(20):
    for j in range(10):
        print(itos[indices[i][j].cpu().detach().item()])
    