# -*- coding: utf-8 -*-
"""
Classes and functions to handle input data for PyTorch models

Classes heavily inspired from Rao, D., & McMahan, B. (2019). Natural Language
Processing with PyTorch. O'Reilly. https://github.com/joosthub/PyTorchNLPBook

Created on Thu Oct  1 17:23:28 2020

@author: Jose Armando Aguasvivas
"""
#%% Imports
import math
import json
import torch
import string
import numpy as np
import pandas as pd

from itertools import product
from collections import Counter
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

#%% Helper functions
def generate_batches(dataset, batch_size, shuffle=True,
                     drop_last=True, device="cpu"):
    """
    Generator function wrapping PyTorch's DataLoader
    
    Ensures torch.Tensors are sent to appropriate device
    
    Args:
        dataset (Dataset): instance of Dataset class
        batch_size (int)
        shuffle (bool): whether to shuffle the data
            Default True
        drop_last (bool): drops reamining data if it doesn't fit in batch
            Default True
        device (torch.device): device to send tensors (for GPU computing)
    """
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                             shuffle=shuffle, drop_last=drop_last)
    
    for data_dict in dataloader:
        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = tensor.to(device)
        yield out_data_dict
        
def set_all_seeds(seed, device):
    """Simultaneously set all seeds from numpy and PyTorch"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == torch.device('cuda:0'):
        torch.cuda.manual_seed_all(seed)
        
def make_train_state(args):
    return {
        'learning_rate': args.learning_rate,
        'epoch_idx': 0,
        'train_loss': [],
        'train_acc': [],
        'train_lang_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_lang_acc': [],
        'test_loss': -1,
        'test_acc': -1,
        'test_lang_acc': -1,
        }

def compute_lang_accuracy(y_pred, y_target):
    preds = torch.sigmoid(y_pred)
    n_correct = torch.eq(preds > 0.5, y_target).sum().item()
    return n_correct / len(preds) * 100

def normalize_sizes(y_pred, y_true):
    """Normalize tensor sizes
    
    Args:
        y_pred (torch.Tensor): the output of the model
            If a 3-dimensional tensor, reshapes to a matrix
        y_true (torch.Tensor): the target predictions
            If a matrix, reshapes to be a vector
    """
    if len(y_pred.size()) == 3:
        y_pred = y_pred.contiguous().view(-1, y_pred.size(2))
    if len(y_true.size()) == 2:
        y_true = y_true.contiguous().view(-1)
    return y_pred, y_true

def compute_accuracy(y_pred, y_true, mask_index=None):
    y_pred, y_true = normalize_sizes(y_pred, y_true)

    _, y_pred_indices = y_pred.max(dim=1)
    
    correct_indices = torch.eq(y_pred_indices, y_true).float()
    valid_indices = torch.ne(y_true, mask_index).float()
    
    n_correct = (correct_indices * valid_indices).sum().item()
    n_valid = valid_indices.sum().item()

    return n_correct / n_valid * 100

def print_state_dict(train_state):
    print((f"Epoch: {train_state['epoch_idx'] + 1} | "
           f"train_loss: {train_state['train_loss'][-1]:.4f} | "
           f"val_loss: {train_state['val_loss'][-1]:.4f}\n"
           f"train_acc_chars: {train_state['train_acc'][-1]:.2f} | "
           # f"train_acc_lang: {train_state['train_lang_acc'][-1]:.2f}\n"
           f"val_acc_chars: {train_state['val_acc'][-1]:.2f} | "
           # f"val_acc_lang: {train_state['val_lang_acc'][-1]:.2f}\n"
           ))
    
def sample_from_model(model, vectorizer, num_samples=1, sample_size=10,
                      temp=1.0, device='cpu'):
    begin_seq = [vectorizer.data_vocab.SOS_idx for _ in range(num_samples)]
    begin_seq = vectorizer.onehot(begin_seq).unsqueeze(1).to(device)
    indices = [begin_seq]
    
    h_t = model.initHidden(batch_size=num_samples, device=device)
    
    for time_step in range(sample_size):
        x_t = indices[time_step]
        out, hidden = model(x_t, h_t)
        prob = out.view(-1).div(temp).exp()
        selected = vectorizer.onehot(torch.multinomial(prob,
                                                       num_samples=num_samples
                                                       ))
        selected = selected.unsqueeze(1).to(device)
        indices.append(selected)
    indices = torch.stack(indices).squeeze(1).permute(2, 0, 1)
    print(indices.shape)
    return indices
        


#%% Helper classes
    
#%% Vocabulary class
class Vocabulary(object):
    """
    Class to handle vocabulary extracted from list of words or sentences.
    
    TODO: Extend to handle phonemes as well
    """
    def __init__(self, stoi=None, SOS="<s>", EOS="</s>", PAD="<p>"):
        """
        Args:
            stoi (dict or None): mapping from tokens to indices
                If None, creates an empty dict
                Default None
            SOS (str or None): Start-of-Sequence token
                Default "<s>"
            EOS (str or None): End-of-Sequence token
                Default "</s>"
            PAD (str or None): Padding token used for handling mini-batches
                Default "<p>"
        """
        if stoi is None:
            stoi = {}
        self._stoi = stoi
        self._itos = {i:s for s,i in self._stoi.items()}
        
        self._SOS_token = SOS
        self._EOS_token = EOS
        self._PAD_token = PAD
        
        if self._SOS_token is not None:
            self.SOS_idx = self.add_token(self._SOS_token)
        if self._EOS_token is not None:
            self.EOS_idx = self.add_token(self._EOS_token)
        if self._PAD_token is not None:
            self.PAD_idx = self.add_token(self._PAD_token)
            
    def to_dict(self):
        """Returns full vocabulary dictionary"""
        return {
            "stoi": self._stoi,
            "itos": self._itos,
            "SOS_token": self._SOS_token,
            "EOS_token": self._EOS_token,
            "PAD_token": self._PAD_token
            }

    @classmethod
    def from_dict(cls, contents):
        """Instantiates vocabulary from dictionary"""
        return cls(**contents)
    
    def add_token(self, token):
        """Update mapping dicts based on token
        
        Args:
            token (str): token to be added
        Returns:
            idx (int): index corresponding to the token
        """
        try:
            idx = self._stoi[token]
        except KeyError:
            idx = len(self._stoi)
            self._stoi[token] = idx
            self._itos[idx] = token
        return idx
    
    def add_many(self, tokens):
        """Adds multiple tokens, one at a time"""
        return [self.add_token(token) for token in tokens]
    
    def token2idx(self, token):
        """Returns index of token"""
        return self._stoi[token]
    
    def idx2token(self, idx):
        """Returns token based on index"""
        if idx not in self._itos:
            raise KeyError(f"Index {idx} not in Vocabulary")
        return self._itos[idx]
    
    def __str__(self):
        return f"<Vocabulary(size={len(self)})>"
    
    def __len__(self):
        return len(self._stoi)

#%% Vectorizer Class
class Vectorizer(object):
    """
    The Vectorizer creates one-hot vectors from sequence of characters/words
    stored in the Vocabulary
    
    TODO: split into word/phoneme vectorizers
    """
    def __init__(self, data_vocab, label_vocab):
        """
        Args:
            data_vocab (Vocabulary): maps char/words to indices
            label_vocab (Vocabulary): maps labels to indices
        """
        self.data_vocab = data_vocab
        self.label_vocab = label_vocab
        
    def vectorize(self, data, vector_len=-1):
        """Vectorize data into observations and targets
        
        Outputs are the vectorized data split into:
            data[:-1] and data[1:]
        At each timestep, the first tensor is the observations, the second
        vector is the target predictions (indices of words and characters)
        
        Args:
            data (str or List[str]): data to be vectorized
                Works for both char level and word level vectorizations
            vector_len (int): Maximum vector length for mini-batch
                Defaults to len(data) - 1
        Returns:
            from_vector (torch.Tensor): observation tensor of
                shape [vector_len, len(data_vocab)]
            to_vector (torch.Tensor): target prediction tensor of
                shape [vector_len, 1]
        """
        indices = [self.data_vocab.SOS_idx]
        indices.extend(self.data_vocab.token2idx(t) for t in data)
        indices.append(self.data_vocab.EOS_idx)
    
        # if add_SOS:
        #     indices = [self.data_vocab._SOS_token] + indices
        # if add_EOS:
        #     indices = indices + [self.data_vocab._EOS_token]
        
        if vector_len < 0:
            vector_len = len(indices)-1
        
        from_vector = torch.empty(vector_len, len(self.data_vocab),
                                  dtype=torch.float32)
        from_indices = indices[:-1]
        # Add pre-padding
        from_vector[:-len(from_indices)] = self.onehot([
                                        self.data_vocab.PAD_idx
                                        ])
        from_vector[-len(from_indices):] = self.onehot(from_indices)
        
        to_vector = torch.empty(vector_len, dtype=torch.int64)
        to_indices = indices[1:]
        to_vector[:-len(to_indices)] = self.data_vocab.PAD_idx
        to_vector[-len(to_indices):] = torch.LongTensor(to_indices)
        
        return from_vector, to_vector
    
    def vectorize_single_char(self, word):
        """Encodes a word character by character
        
        Args:
            word (str): word to encode
        Yields:
            i (int): character position
            from_vector (torch.Tensor): observation tensor of
                shape [1, len(data_vocab)]
            to_vector (torch.Tensor): target prediction tensor of
                shape [1, 1]
        """
        indices = [self.data_vocab.SOS_idx]
        indices.extend(self.data_vocab.token2idx(c) for c in word)
        indices.append(self.data_vocab.EOS_idx)
        
        for i, (idx1, idx2) in enumerate(zip(indices[:-1], indices[1:])):
            from_vector = self.onehot([idx1])
            to_vector = torch.LongTensor([idx2])
            yield i, (from_vector, to_vector)
    
    def onehot(self, indices):
        """Encodes a list of indices into a one-hot tensor
        
        Args:
            indices (List[int]): list of indices to encode
        Returns:
            onehot (torch.Tensor): one-hot tensor from indices of
                shape [len(indices), len(data_vocab)]
        """
        onehot = torch.zeros(len(indices), len(self.data_vocab),
                             dtype=torch.float32)
        
        for i, idx in enumerate(indices):
            onehot[i][idx] = 1.
        return onehot
        
    @classmethod
    def from_df(cls, df, split="char"):
        """Instantiate the vectorizer from a dataframe
        
        Args:
            df (pandas.DataFrame): the dataset
            splits (str): split data into chars or words
                Default "chars"
        Returns:
            an instance of Vectorizer
            
        TODO: split into different window sizes using int splits
        """
        data_vocab = Vocabulary()
        label_vocab = Vocabulary(SOS=None, EOS=None, PAD=None)
        
        for i, row in df.iterrows():
            if split == "char":
                data_vocab.add_many([c for c in row.data])
            else:
                data_vocab.add_many([c for c in row.data.split(split)])
            
            label_vocab.add_token(row.label)
        
        return cls(data_vocab, label_vocab)
    
    @classmethod
    def from_dict(cls, contents):
        """Instantiate the vectorizer from a dictionary
        
        Args:
            contents (pandas.DataFrame): the dataset
        Returns:
            an instance of Vectorizer
        """
        data_vocab = Vocabulary.from_dict(contents['data_vocab'])
        label_vocab = Vocabulary.from_dict(contents['label_vocab'])
        return cls(data_vocab, label_vocab)
    
    def to_dict(self):
        """Returns a dictionary of the vocabularies"""
        return {
            'data_vocab': self.data_vocab.to_dict(),
            'label_vocab': self.label_vocab.to_dict()
            }
        
#%% TextDataset class
class TextDataset(Dataset):
    """Combines Vocabulary and Vectorizer classes into one easy interface"""
    def __init__(self, df, vectorizer=None, p=None, split="char", seed=None):
        """
        Args:
            df (pandas.DataFrame): the dataset
            vectorizer (Vectorizer): Vectorizer instantiated from the dataset
            p (List[float] or None): proportion of each train label 
                to use (e.g. 50/50). If None, selects full train data    
                Default None
        """
        self.df = df
        
        self.train_df = self.df[self.df.split=='train']
        if p is not None:
            labs = self.train_df.label.unique()
            tmp = pd.DataFrame()
            if isinstance(p, float):
                p = [p, 1-p]
            for frac, l in zip(p, labs):
                dat = self.train_df[self.train_df.label == l]
                tmp = pd.concat([tmp, dat.sample(frac=frac,
                                                 random_state=seed)])
            self.train_df = tmp
        self.train_size = len(self.train_df)
        
        if not vectorizer:
            self._vectorizer = Vectorizer.from_df(self.train_df, split=split)
        else:
            self._vectorizer = vectorizer
        
        self.val_df = self.df[self.df.split=='val']
        self.val_size = len(self.val_df)
        
        self.test_df = self.df[self.df.split=='test']
        self.test_size = len(self.test_df)
        
        self._max_seq_len = max(map(len, self.df.data)) + 1 # SOS/EOS
        
        self._lookup_dict = {
            "train": (self.train_df, self.train_size),
            "val": (self.val_df, self.val_size),
            "test": (self.test_df, self.test_size)
            }
        
        self.set_split('train')
        
        # Handles imbalanced labels
        labels = self.train_df.label.value_counts().to_dict()
        def sort_key(item):
            return self._vectorizer.label_vocab.token2idx(item[0])
        sorted_cnts = sorted(labels.items(), key=sort_key)
        freqs = [cnt for _, cnt in sorted_cnts]
        self.label_weights = 1.0 / torch.tensor(freqs, dtype=torch.float32)
        
    @classmethod
    def load_dataset_and_make_vectorizer(cls, csv, split="char", p=None,
                                         seed=None):
        """Loads a pandas DataFrame and makes Vectorizer from scratch
        
        DataFrame should have following named columns:
            [data, labels, split] where
            data are the text (documents) to vectorize
            labels are the target labels for the text (for classification)
            split indicates train, val, and test splits of the data
        
        Args:
            csv (str): path to the dataset
            split (str): split text into chars or words
        Returns:
            Instance of TextDataset
        """
        df = pd.read_csv(csv)
        return cls(df, p=p,split=split, seed=seed)
    
    def save_vectorizer(self, vectorizer_path):
        """Saves vectorizer in json format
        
        Args:
            vectorizer_path (str): path to save vectorizer
        """
        with open(vectorizer_path, 'w') as f:
            json.dump(self._vectorizer.to_dict(), f)
    
    def get_vectorizer(self):
        """Returns vectorizer for the dataset"""
        return self._vectorizer
    
    def set_split(self, split="train"):
        """Changes the split of TextDataset
        
        Options depend on splits used when creating TextDataset
        Ideally "train", "val", "test"
        """
        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]
    
    def __len__(self):
        return self._target_size
    
    def __getitem__(self, index):
        """Primary interface between TextDataset and PyTorch's DataLoader
        
        Used for generating batches of data (see utils.generate_batches)
        
        Args:
            index (int): Index of the data point
        Returns:
            Dictionary holding the data point with keys
                [X, Y, label]
        """
        row = self._target_df.iloc[index]
        
        from_vector, to_vector = self._vectorizer.vectorize(row.data,
                                                            self._max_seq_len)
        
        label = self._vectorizer.label_vocab.token2idx(row.label)
        
        return {'X': from_vector, 'Y': to_vector, 'label': label}
    
    def get_num_batches(self, batch_size):
        """Returns number of batches in the dataset given batch_size
        
        Args:
            batch_size (int)
        Returns:
            Number of batches in dataset
        """
        return len(self) // batch_size

#%% CharNGram Class
class CharNGram(object):
    """A character n-gram model trained on a list of words.
    
    Concepts from Jurafsky, D., & Martin, J.H. (2019). Speech and Language
    Processing. Stanford Press. https://web.stanford.edu/~jurafsky/slp3/
    
    This class is not optimized for large ngram models, use with caution
    for models of order 5 and above.
    """
    def __init__(self, data, n, laplace=1, SOS_token='<s>', EOS_token='</s>'):
        """Data should be iterable of words
        
        Args:
            data (List[str]): dataset from which to create the ngram model
            n (int): order of the model. Should be larger than 0
            laplace (int): additive smoothing factor for unseen combinations
                Default 1
            SOS_token (str): Start-of-Sequence token
            EOS_token (str): End-of-Sequence token
        """
        assert (n > 0), n
        self.n = n
        self.laplace = laplace
        self.SOS_token = SOS_token
        self.EOS_token = EOS_token
        self.data = data
        self.vocab = list(string.ascii_lowercase) + [SOS_token, EOS_token]
        self.processed_data = self._preprocess(self.data, n)
        self.ngrams = self._split_and_count(self.processed_data, self.n)
        self.model = self._smooth()
        
    def _preprocess(self, data, n):
        """Private method to preprocess a dataset of documents
        
        Args:
            data (List[str]): documents to be processed
            n (int): order of ngram model for processing
        Returns:
            new_data (List[str]): preprocessed data
        """
        new_data = []
        for word in data:
            new_data.append(self.process_word(word, n))
        return new_data
    
    def process_word(self, word, n):
        """Adds SOS and EOS tokens with padding
        
        Adds padding of SOS_tokens and EOS_tokens to each document
            padding size = n-1 for n > 1
        
        Args:
            word (str): word to be padded
            n (int): order of ngram model
        Returns:
            padded word (List[str])
        """
        pad = max(1, n-1)
        return [self.SOS_token] * pad +\
                    list(word.lower()) +\
                    [self.EOS_token]
    
    def _split_word(self, word, n):
        """Private generator to handle moving window over word of size n"""
        for i in range(len(word) - n + 1):
            yield tuple(word[i:i+n])
    
    def _split_and_count(self, data, n):
        """Private method to create ngram counts
        
        Args:
            data (List[str]): preprocessed data
            n (int): order of ngram model
        Returns:
            cntr (Counter): count of each ngram in data
        """
        cntr = self._initialize_counts(n)
        for word in data:
            for ngram in self._split_word(word, n):
                cntr[ngram] += 1
        return cntr
    
    def _initialize_counts(self, n):
        """Private method to initialize the ngram counter
        
        Accounts for unseen tokens by taking the product of the vocabulary
        
        Args:
            n (int): order of ngram model
        Returns:
            cntr (Counter): initialized counter of 0s for each plausible ngram
        """
        def is_plausible(permutation):
            if self.SOS_token not in permutation and \
                self.EOS_token not in permutation: return True
            n = len(permutation)
            
            if self.EOS_token in permutation[0]: return False
            if self.SOS_token in permutation[-1]: return False
            flg = False
            cnt = 0
            for i in range(n-1, -1, -1):
                if self.SOS_token == permutation[i]:
                    flg = True
                    cnt += 1
                else:
                    if flg: return False
            if cnt == n: return False
            
            flg = False
            cnt = 0
            for i in range(n):
                if self.EOS_token == permutation[i]:
                    flg = True
                    cnt += 1
                else:
                    if flg: return False
            return True
            if cnt == n : return False
        
        cntr = Counter()
        for perm in product(self.vocab, repeat=n):
            if is_plausible(perm):
                cntr[tuple(perm)] = 0
        return cntr
    
    def _smooth(self):
        """Private method to convert counts to probabilities using
        additive Laplace smoothing
        
        Returns:
            cntr (Counter): normalized probabilities of each ngram in data
        """
        if self.n == 1:
            s = sum(self.ngrams.values())
            return Counter({key: val/s for key, val in self.ngrams.items()})
        else:
            vocab_size = len(self.vocab)-1
            
            ret = self.ngrams.copy()
            
            m = self.n - 1
            m_grams = self._split_and_count(self.processed_data, m)
            
            for ngram, value in self.ngrams.items():
                m_gram = ngram[:-1]
                m_count = m_grams[m_gram]
                ret[ngram] = (value + self.laplace) /\
                            (m_count + self.laplace * vocab_size)
            
            return ret
    
    def to_txt(self, filepath):
        """Saves model to disk as a tab separated txt file"""
        with open(filepath, 'w') as file:
            for ngram, value in self.model.items():
                file.write(f"{' '.join(ngram)}\t{value}\n")
    
    def from_txt(self, filepath):
        """Reads model from a tab separated txt file"""
        with open(filepath, 'r') as file:
            data = file.readlines()
        self.model = Counter()
        for ngram, value in data.split('\t'):
            self.model[tuple(ngram.split(' '))] = value
        self.n = len(self.model.keys()[0])
    
    def to_df(self):
        """Creates a DataFrame from Counter of ngrams
        
        Warning: Do not use with ngrams of order >= 5
        
        Returns:
            df (pandas.DataFrame): dataframe of normalized probabilities
                shape [n_plausible_ngrams, len(vocab)]
        """
        idxs, cols = set(), set()
        for k in self.model.keys():
            idxs.add(' '.join(k[:-1]))
            cols.add(k[-1])
        df = pd.DataFrame(data=0.0,
                          index=sorted(list(idxs)),
                          columns=sorted(list(cols)))
        for ngram, value in self.model.items():
            cntx = ' '.join(ngram[:-1])
            trgt = ngram[-1]
            df.loc[cntx, trgt] = value
        return df.fillna(0.0)
    
    def get_single_probability(self, word, log=False):
        """Calculates the probability (likelihood) of a word given the ngram
        model
        
        Args:
            word (str or List[str]): target word
            log (bool): whether to get loglikelihood instead of probability
        Returns:
            prob (float): probability of the word given the ngram model
        """
        if isinstance(word, str):
            word = self.process_word(word, self.n)
        n = len(word)
        prob = 0.0 if log else 1.0
        for ngram in self._split_word(word, self.n):
            if ngram not in self.model:
                print(ngram)
            p = self.model[ngram]
            if log:
                prob += math.log(p)
            else:
                prob *= p
        return prob / n
    
    def perplexity(self, data):
        """Calculates the perplexity of an entire dataset given the model
        
        Perplexity is the inverse probability of the dataset, normalized
        by the number of words
        
        To avoid numeric overflow due to multiplication of probabilities,
        the probabilties are log-transformed and the final score is then
        exponentiated. Thus:
            
            Perplexity = exp(-(sum(probs)/N)) ~ exp(NLL/N)
        
        where N is the number of words and probs is the vector of probabilities
        for each word in the dataset.
        
        Lower perplexity is equivalent to higher probability of the data given
        the ngram model.
        
        Args:
            data (\List[str]): datset of words
        Returns:
            perplexity (float): perplexity of the dataset given the ngram model
        """
        test_tokens = self._preprocess(data, self.n)
        N = len(test_tokens)
        
        probs = 0.0
        for word in test_tokens:
            probs -= self.get_single_probability(word, log=True)
        
        return math.exp(probs/N)
    
    def get_distribution_from_context(self, context):
        """Get the multinomial distribution for the next character given a
        context
        
        Args:
            context (str or List[str]): context of variable length
        Returns:
            dist (dict): probability distribution of the next letter
        """
        m = len(context)
        if m < self.n-1:
            context = [self.SOS_token] * (self.n-m-1) + list(context)
        elif m > self.n-1:
            context = list(context[-self.n+1:])
        context = list(context)
        dist = {v:0 for v in self.vocab}
        for v in self.vocab:
            dist[v] = self.model[tuple(context + [v])]
        del dist[self.SOS_token]
        return dist
    
    def calculate_accuracy(self, wordlist, topk=1):
        N = len(wordlist)
        total_acc = 0.0
        for word in wordlist:
            acc = 0.0
            padded_word = self.process_word(word, self.n)
            for i, ngram in enumerate(self._split_word(padded_word, self.n)):
                if i+self.n >= len(padded_word):
                    break
                dist = self.get_distribution_from_context(ngram)
                topl = [k for k, _ in sorted(dist.items(),
                                              key=lambda x:x[1],
                                              reverse=True)]
                acc += 1 if padded_word[i+self.n] in topl[:topk] else 0
            total_acc += (acc / (len(word)+1))
        return total_acc * 100 / N
    
    def _next_candidate(self, prev, without=[]):
        """Private method to select next candidate from previous context
        
        Candidates are selected at random from a multinomial distribution
        weighted by the probability of next token given context.
        
        Args:
            prev (Tuple[str]): previous context
        Returns:
            letter (str): selected next candidate
            prob (float): probability of next candidate given context
        """
        letters = self.get_distribution_from_context(prev)
        letters = {l:prob for l, prob in letters.items() if l not in without}
        letters, probs = list(letters.keys()), list(letters.values())
        topi = torch.multinomial(torch.FloatTensor(probs), 1)[0].item()
        return letters[topi], probs[topi]
    
    def generate_words(self, num, min_len=3, max_len=10, without=[]):
        """Generates a number of words by sampling from the ngram model
        
        Generator method.
        
        Args:
            num (int): number of words to generate
            min_len (int): minimum length of the words
            max_len (int): maximum length of the words
            without (List[str]): list of tokens to ignore during selection
        Yields:
            word (str): generated word
        """
        for i in range(num):
            word, prob = [self.SOS_token] * max(1, self.n-1), 1
            while word[-1] != self.EOS_token:
                prev = () if self.n == 1 else tuple(word[-self.n+1:])
                blacklist = [self.EOS_token] if len(word) < min_len else []
                next_token, next_prob = self._next_candidate(prev,
                                                             without=blacklist
                                                             )
                word.append(next_token)
                prob *= next_prob
                if len(word) >= max_len:
                    word.append(self.EOS_token)
            word = [w for w in word if w not in [self.SOS_token,
                                                 self.EOS_token]]
            yield ''.join(word), -1/math.log(prob)
    
    def __len__(self):
        return len(self.ngrams)
    
    def __str__(self):
        return f"<{self.n}-gram model(size={len(self)})>"

#%% Trie
class TrieNode(object):
    """Node for the Trie class"""
    def __init__(self, vocab_len=27):
        """
        Args:
            vocab_len (int): length of the vocabulary
        """
        self.finished = False
        self.children = [None] * vocab_len
        self.prob = 0
        self.prefix = ''
        self.cnt = 0
        
class Trie(object):
    """Trie (pronounced "try") or prefix tree is a tree data structure,
    which is used for retrieval of a key in a dataset of strings.
    """
    def __init__(self, vocab_len=27):
        """
        Args:
            vocab_len (int): length of the vocabulary
        """
        self.vocab_len = vocab_len
        self.root = TrieNode(vocab_len=vocab_len)
    
    def _ord(self, c):
        """Private method to get index from character"""
        if c == '</s>': 
            ret = self.vocab_len - 1
        else:
            ret = ord(c) - ord('a')
        
        if not 0 <= ret <= self.vocab_len:
            raise KeyError(f"Character index {ret} not in vocabulary")
        else:
            return ret

    def insert(self, word):
        """Inserts a word into the Trie
        
        Args:
            word (str or List[str]): word to be added to Trie
        """
        curr = self.root
        
        for c in word:
            i = self._ord(c)
            if not curr.children[i]:
                curr.children[i] = TrieNode(vocab_len=self.vocab_len)
            context = curr.prefix
            curr = curr.children[i]
            curr.prefix = context + c
            curr.cnt += 1
        
        curr.finished = True
    
    def insert_many(self, wordlist):
        """Inserts several words into the Trie
        
        Args:
            wordlist (List[List[str]]): list of words to be added to Trie
        """
        for word in wordlist:
            self.insert(word)
    
    def search(self, word):
        """Returns True if word is in the Trie"""
        curr = self.root
        for c in word:
            i = self._ord(c)
            if not curr.children[i]:
                return False
            curr = curr.children[i]
        return curr.finished
            
    def starts_with(self, prefix):
        """Returns True if prefix is in Trie"""
        curr = self.root
        for c in prefix:
            i = self._ord(c)
            if not curr.children[i]:
                return False
            curr = curr.children[i]
        return True
    
    def get_probabilities(self):
        """Calculates the probability of different prefixes"""
        curr = self.root
        total = curr.cnt
        
        for i in range(self.vocab_len):
            if curr.children[i]:
                curr.children[i].prob /= total
                self.calculate_probabilities(curr.children[i])
    
    def calculate_empirical_distribution(self):
        """Calculates empirical distribution for the entire Trie"""
        q = []
        q.append(self.root)
        while q:
            p = []
            curr = q.pop()
            cnt = 0
            for i in range(self.vocab_len):
                if curr.children[i]:
                    q.append(curr.children[i])
                    p.append(curr.children[i].prob)
                else:
                    cnt += 1
                    p.append(0)