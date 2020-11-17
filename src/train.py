# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 10:54:40 2020

@author: josea
"""
# %% Imports
import os
import utils
import torch
import torch.nn as nn
import pandas as pd

from polychar import PolyChar
from argparse import Namespace

# %% Set-up paramenters
args = Namespace(
    csv='../processed_data/',
    vectorizer_file="../processed_data/vectorizer.json",
    model_checkpoint_file='/models/checkpoints/',
    model_save_file='models/',
    hidden_dims=128,
    n_lstm_layers=1,
    drop_p=0.1,
    n_epochs=100,
    early_stopping_patience=5,
    learning_rate=0.001,
    batch_size=64,
    datafiles=['ESP-ENG.csv', 'ESP-EUS.csv'],
    modelfiles=['ESEN_', 'ESEU_'],
    probs=[1, 20, 40, 50, 60, 80, 99],
    n_runs=5,
    plotting=False,
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    seed=404
)

utils.set_all_seeds(args.seed, args.device)

print(args.device)

for data, category in zip(args.datafiles, args.modelfiles):
    for prob in args.probs:
        end = f"{prob:02}-{100-prob:02}"

        df = pd.read_csv(args.csv + data)
        vectorizer = utils.Vectorizer.from_df(df)

        # %% Set-up dataset, vectorizer, and model
        dataset = utils.TextDataset.make_text_dataset(df, vectorizer,
                                                      p=prob/100, seed=args.seed)

        for run in range(args.n_runs):
            m_name = f"{category}{end}"

            print(f"\n{data}: {m_name}_{run}\n")

            model = PolyChar(
                n_in=len(vectorizer.data_vocab),
                n_hidden=args.hidden_dims,
                n_layers=args.n_lstm_layers,
                n_out=len(vectorizer.data_vocab),
                drop_p=args.drop_p
            ).to(args.device)

            loss_func1 = nn.CrossEntropyLoss(
                ignore_index=vectorizer.data_vocab.PAD_idx)

            optimizer = torch.optim.Adam(
                model.parameters(), lr=args.learning_rate)

            # %% Training loop
            train_state = utils.make_train_state()

            for it in range(args.n_epochs):
                if (it+1) % 10 == 0:
                    print(it+1)
                train_state['epoch_idx'] = it

                dataset.set_split('train')
                batch_generator = utils.generate_batches(dataset,
                                                         batch_size=args.batch_size,
                                                         device=args.device)

                running_loss = 0.
                running_acc = 0.

                model.train()
                for batch_id, batch_dict in enumerate(batch_generator):
                    optimizer.zero_grad()

                    hidden = model.initHidden(args.batch_size, args.device)

                    out, hidden = model(batch_dict['X'], hidden)

                    loss = loss_func1(
                        *utils.normalize_sizes(out, batch_dict['Y']))

                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5)
                    optimizer.step()

                    # Update train_state arguments
                    running_loss += (loss.item() -
                                     running_loss) / (batch_id + 1)
                    acc_chars = utils.compute_accuracy(
                        out, batch_dict['Y'], vectorizer.data_vocab.PAD_idx
                    )

                    running_acc += (acc_chars - running_acc) / (batch_id + 1)

                train_state['train_loss'].append(running_loss)
                train_state['train_acc'].append(running_acc)

                # EVAL
                dataset.set_split('val')
                batch_generator = utils.generate_batches(dataset,
                                                         batch_size=args.batch_size,
                                                         device=args.device)
                running_loss = 0.
                running_acc = 0.

                model.eval()
                for batch_id, batch_dict in enumerate(batch_generator):
                    hidden = model.initHidden(args.batch_size, args.device)

                    out, hidden = model(batch_dict['X'], hidden)

                    loss = loss_func1(
                        *utils.normalize_sizes(out, batch_dict['Y']))

                    running_loss += (loss.item() -
                                     running_loss) / (batch_id + 1)
                    acc_chars = utils.compute_accuracy(
                        out, batch_dict['Y'], vectorizer.data_vocab.PAD_idx
                    )
                    running_acc += (acc_chars - running_acc) / (batch_id + 1)

                train_state['val_loss'].append(running_loss)
                train_state['val_acc'].append(running_acc)

                # Early-stopping and adaptive learning rate
                if train_state['epoch_idx'] == 0:
                    try:
                        os.stat(args.model_save_file + f"{m_name}")
                    except:
                        os.mkdir(args.model_save_file + f"{m_name}")
                    torch.save(model, args.model_save_file +
                               f"{m_name}/{m_name}_{run}" + ".pt")
                elif train_state['epoch_idx'] >= 1:
                    loss_tm1, loss_t = train_state['val_loss'][-2:]

                    if loss_t >= loss_tm1:
                        train_state['early_stopping_step'] += 1
                    else:
                        if loss_t < train_state['early_stopping_best_val']:
                            torch.save(model, args.model_save_file +
                                       f"{m_name}/{m_name}_{run}" + ".pt")
                            train_state['early_stopping_best_val'] = loss_t
                        train_state['early_stopping_step'] = 0

                if train_state['early_stopping_step'] >= args.early_stopping_patience:
                    print(f"Early stopping reached at epoch {it+1}")
                    break

            running_loss = 0.
            running_acc = 0.

            dataset.set_split('test')
            batch_generator = utils.generate_batches(dataset,
                                                     batch_size=args.batch_size,
                                                     device=args.device)

            model.eval()
            for batch_id, batch_dict in enumerate(batch_generator):
                hidden = model.initHidden(args.batch_size, args.device)

                out, hidden = model(batch_dict['X'], hidden)

                loss = loss_func1(*utils.normalize_sizes(out, batch_dict['Y']))

                running_loss += (loss.item() - running_loss) / (batch_id + 1)
                acc_chars = utils.compute_accuracy(
                    out, batch_dict['Y'], vectorizer.data_vocab.PAD_idx
                )
                running_acc += (acc_chars - running_acc) / (batch_id + 1)

            train_state['test_loss'] = running_loss
            train_state['test_acc'] = running_acc

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
