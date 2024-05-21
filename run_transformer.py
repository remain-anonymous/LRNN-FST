

import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from transformers import BertTokenizer, BertModel, AutoModel, AutoTokenizer
from torch.utils.data.dataset import random_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import cmath
import time
import matplotlib.pyplot as plt
import torch
import torchvision
import pickle 
import numpy as np
from tqdm import tqdm

from model import *



def run_trial(SEED, hid_feature_size=-1, hid_seq_size=-1, transformer_nhead=-1, hid_rnn_size=-1, blocks=-1, dropout=-1,  use_fst=-1, use_transformer=-1):

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    start_time = time.time()
    #max_duration = 27000
    
    def clean_text(text):
        # Remove special characters, numbers, punctuations
        text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
        text = text.lower()
        text = text.strip()
        return text
    
    class MyDataset(Dataset):
        def __init__(self, features, labels=None):
            self.features = features
            self.labels = labels
            self.bert_model = AutoModel.from_pretrained('transformers/bert-base-uncased')
    
        def __len__(self):
            return len(self.features)
    
        def __getitem__(self, idx):
            with torch.no_grad():  # We don't need gradients for embedding extraction
                input_ids = self.features[idx].unsqueeze(0)  # Add batch dimension
                outputs = self.bert_model(input_ids)
                embeddings = outputs.last_hidden_state.squeeze(0)  # Remove batch dimension
            if self.labels is not None:
                return embeddings, self.labels[idx]
            return embeddings
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained('transformers/bert-base-uncased')
    
    def tokenize_and_pad(text_list, max_length=512):
        # Tokenize each text in the list
        tokenized = [tokenizer.encode(text, add_special_tokens=True, max_length=max_length, truncation=True) for text in text_list]
        
        # Pad each tokenized text to the max_length
        padded = np.array([i + [0] * (max_length - len(i)) for i in tokenized])
        
        return torch.tensor(padded)
    
    
    # Load your datasets
    
    train = pd.read_csv("daigt-v2-train-dataset/train_v2_drcat_02.csv", sep=',')
    train = train.rename(columns={'label': 'generated'})
    train = train.drop_duplicates(subset=['text'])
    train.reset_index(drop=True, inplace=True)
    
    train, val = train_test_split(train, test_size=0.2, stratify=train['generated'], random_state=SEED) 
    
    # Clean the text
    train['cleaned_text'] = train['text'].apply(clean_text)
    val['cleaned_text'] = val['text'].apply(clean_text)
    #test['cleaned_text'] = test['text'].apply(clean_text)
    
    # Tokenize and Pad
    X_train_padded = tokenize_and_pad(train['cleaned_text'].tolist())
    X_val_padded = tokenize_and_pad(val['cleaned_text'].tolist())
    #X_test_padded = tokenize_and_pad(test['cleaned_text'].tolist())
    
    y_train_tensor = torch.tensor(train['generated'].values, dtype=torch.float32)
    y_val_tensor = torch.tensor(val['generated'].values, dtype=torch.float32)
    
     
    # Create instances of the dataset
    train_dataset = MyDataset(X_train_padded, labels=y_train_tensor)
    val_dataset = MyDataset(X_val_padded, labels=y_val_tensor)
    #test_dataset = MyDataset(X_test_padded)
    
    
    # Define DataLoaders for the training and validation sets
    batch_size = 64  # You can modify this based on your requirement
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    #test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    for X, y in train_loader:
        print("Tensor size:", X.size())
    
        # Let's decode the first example of the batch
        example_idx = 0
        token_ids_example = X_train_padded[example_idx].tolist()  # Convert tensor to list of token IDs
        decoded_text = tokenizer.decode(token_ids_example, skip_special_tokens=True)
    
        # Print the decoded text and its label
        print("\nExample Essay:\n", decoded_text)
        print("\nCorresponding Label:", y_train_tensor[example_idx].item())
        break  # Only process the first batch for this example
    
    DTYPE = torch.complex64
    DEVICE=2 #'cpu'
    
    
    model = ParallelRNNBlock(input_dim=768, out_dim=1, seq_size=512, hid_feature_size=hid_feature_size, hid_seq_size=hid_seq_size, 
                             transformer_nhead=transformer_nhead, hid_rnn_size=hid_rnn_size, 
                             blocks=blocks, dropout=dropout,  FST=use_fst, transformer=use_transformer).to(DEVICE)
    print(sum(p.numel() for p in model.parameters()))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)
    
    model.to(DEVICE)
    
    nmpara = 0
    for n,p in model.named_parameters():
        if p.requires_grad==True:
            nmpara += np.prod(p.size())

    print(f'Nparams: {nmpara}' )
        
    results = {'epoch' : [],
                'train_acc' : [],
                'val_acc' : [],
                'mean_train_acc' : [],
                'mean_val_acc' : [],
                'mean_train_acc_last' : [],
                'mean_val_acc_last' : [],
                'nparams' : nmpara, 
                'hid_feature_size': hid_feature_size,
                'hid_seq_size': hid_seq_size,
                'transformer_nhead': transformer_nhead,
                'use_transformer': use_transformer,
                'use_fst': use_fst,
                'blocks' : blocks,
                'dropout' : dropout,
                'lr' : 1e-4,
                'betas' : (0.9, 0.999),
                'eps' : 1e-08,
                      }
    
    nmaxit = 1
    max_val = 0.5
    
    
    for epoch in tqdm(range(50)):
        model.train()
        #print('Training')
        results['epoch'].append(epoch)
    
        _acc = []
        
        total_iterations = nmaxit #len(train_loader)
        #progress_bar = tqdm(enumerate(train_loader), total=total_iterations, desc="Training")
    
        for n, (inputs, target) in enumerate(train_loader):
            optimizer.zero_grad() 
    
            inputs = inputs.to(DEVICE)
            target = target.to(DEVICE)
    
            out, h = model(inputs)
    
            loss = F.binary_cross_entropy(torch.sigmoid(out[:,-1,0]), target)
    
            loss.backward()
            optimizer.step()
        
            current_acc = roc_score(torch.sigmoid(out[:, -1, 0]), target)
            _acc.append(current_acc)
            
            #progress_bar.set_postfix(accuracy=f'{current_acc:.2f}', refresh=True)
            if n == nmaxit:
                break
        
    
        results['train_acc'].append(_acc)
        results['mean_train_acc'].append(np.mean(_acc))
    
        if True:
            #print('Validating')
            with torch.no_grad():
                model.eval()
    
                total_iterations = len(val_loader)
                #progress_bar = tqdm(enumerate(val_loader), total=total_iterations, desc="Validating")
    
                _acc = []
    
                for n, (inputs, target) in enumerate(val_loader):
    
    
                    inputs = inputs.to(DEVICE)
                    target = target.to(DEVICE)
    
                    out, h = model(inputs)
    
                    current_acc = roc_score(torch.sigmoid(out[:, -1, 0]), target)
                    _acc.append(current_acc)
                    if n == nmaxit:
                        break
            
            results['val_acc'].append(_acc)
            results['mean_val_acc'].append(np.mean(_acc))
            
        pickle.dump(results, open(f'gridsearch_results/result_transformer_FST_{SEED}.pkl', 'wb'))
        

seed_n = 0

for n_blocks in [1,2,3]:
    for hid_feature_size in [32, 64, 128]:
        for hid_seq_size in [32, 64, 128]:
            run_trial(seed_n, hid_feature_size=hid_feature_size, hid_seq_size=hid_seq_size, transformer_nhead=1, hid_rnn_size=-1, blocks=n_blocks, dropout=0.2,  use_fst=False, use_transformer=True)
            seed_n += 1