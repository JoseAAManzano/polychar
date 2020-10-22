# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 10:54:40 2020

@author: josea
"""
#%% Imports
import utils
import torch
import torch.nn as nn

from polychar import PolyChar
from argparse import Namespace 



#%% Set-up paramenters
args = Namespace(
    csv='../processed_data/',
    vectorizer_file="../processed_data/vectorizer.json",
    model_checkpoint_file='/models/checkpoints/',
    model_save_file='models/',
    hidden_dim=128,
    n_lstm_layers=1,
    n_epochs=25,
    learning_rate=0.001,
    batch_size=64,
    datafiles = ['ESP-ENG.csv', 'ESP-EUS.csv'],
    modelfiles = ['ESEN_', 'ESEU_'],
    probs = [0.01, 0.3, 0.5, 0.7, 0.99],
    plotting=False,
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    seed=404
    )

utils.set_all_seeds(args.seed, args.device)

for data, category in zip(args.datafiles, args.modelfiles):
    for prob in args.probs:
        end = f"{int(prob*100)}-{int((1-prob)*100)}"
        m_name = f"{category}{end}"
        print(f"{data}: {m_name}\n")
        #%% Set-up dataset, vectorizer, and model
        dataset = utils.TextDataset.load_dataset_and_make_vectorizer(
            args.csv + data,
            p=prob, seed=args.seed)
        # dataset.save_vectorizer(args.vectorizer_file)
        vectorizer = dataset.get_vectorizer()
        
        model = PolyChar(
            n_in=len(vectorizer.data_vocab),
            n_hidden=args.hidden_dim,
            n_layers=args.n_lstm_layers,
            n_out=len(vectorizer.data_vocab)
            ).to(args.device)
        
        loss_func1 = nn.CrossEntropyLoss(ignore_index=vectorizer.data_vocab.PAD_idx)
        # loss_func2 = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        
        
        #%% Training loop
        train_state = utils.make_train_state(args)
        
        for it in range(args.n_epochs):
            train_state['epoch_idx'] = it
            
            dataset.set_split('train')
            batch_generator = utils.generate_batches(dataset,
                                                     batch_size=args.batch_size,
                                                     device=args.device)
            
            running_loss = 0.
            running_acc = 0.
            # running_lang_acc = 0.
            
            model.train()
            for batch_id, batch_dict in enumerate(batch_generator):
                optimizer.zero_grad()
                
                hidden = model.initHidden(args.batch_size, args.device)
                
                out, hidden = model(batch_dict['X'], hidden)
                
                loss = loss_func1(*utils.normalize_sizes(out, batch_dict['Y']))
                # loss += loss_func2(lang,
                #                    batch_dict['label'].float().unsqueeze(1))
                
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                
                # Update train_state arguments
                running_loss += (loss.item() - running_loss) / (batch_id + 1)
                acc_chars = utils.compute_accuracy(
                                out, batch_dict['Y'], vectorizer.data_vocab.PAD_idx
                                )
                # acc_lang = utils.compute_lang_accuracy(
                #                 lang, 
                #                 batch_dict['label'].float().unsqueeze(1)
                #                 )
                running_acc += (acc_chars - running_acc) / (batch_id + 1)
                # running_lang_acc += (acc_lang - running_lang_acc) / (batch_id + 1)
                
            train_state['train_loss'].append(running_loss)
            train_state['train_acc'].append(running_acc)
            # train_state['train_lang_acc'].append(running_lang_acc)
            
            # EVAL
            dataset.set_split('val')
            batch_generator = utils.generate_batches(dataset,
                                                     batch_size=args.batch_size,
                                                     device=args.device)
            running_loss = 0.
            running_acc = 0.
            # running_lang_acc = 0.
            
            model.eval()
            for batch_id, batch_dict in enumerate(batch_generator):
                hidden = model.initHidden(args.batch_size, args.device)
                
                out, hidden = model(batch_dict['X'], hidden)
                
                loss = loss_func1(*utils.normalize_sizes(out, batch_dict['Y']))
                # loss += loss_func2(lang,
                #                    batch_dict['label'].float().unsqueeze(1))
                
                running_loss += (loss.item() - running_loss) / (batch_id + 1)
                acc_chars = utils.compute_accuracy(
                                out, batch_dict['Y'], vectorizer.data_vocab.PAD_idx
                                )
                # acc_lang = utils.compute_lang_accuracy(
                #                 lang, 
                #                 batch_dict['label'].float().unsqueeze(1)
                #                 )
                running_acc += (acc_chars - running_acc) / (batch_id + 1)
                # running_lang_acc += (acc_lang - running_lang_acc) / (batch_id + 1)
            
            train_state['val_loss'].append(running_loss)
            train_state['val_acc'].append(running_acc)
                # train_state['val_lang_acc'].append(running_lang_acc)
            utils.print_state_dict(train_state)
        
        if args.plotting:
            import matplotlib.pyplot as plt
            
            plt.plot(train_state['train_loss'], label='train_loss')
            plt.plot(train_state['val_loss'], label='val_loss')
            plt.legend()
            plt.show()
            
            plt.plot(train_state['train_acc'], label='train_acc')
            plt.plot(train_state['val_acc'], label='val_acc')
            plt.legend()
            plt.show()
        
        # plt.plot(train_state['train_lang_acc'], label='train_lang_acc')
        # plt.plot(train_state['val_lang_acc'], label='val_lang_acc')
        # plt.legend()
        # plt.show()
        
        torch.save(model, args.model_save_file + m_name + ".pt")
    