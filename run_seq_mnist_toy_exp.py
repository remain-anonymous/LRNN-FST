

import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from torchvision import datasets, transforms

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

for EXP_NUM in range(0, 5):

    SEED = EXP_NUM//2
    use_transformer = bool(1 - EXP_NUM%2)
    
    print(use_transformer)
    
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    
    
    start_time = time.time()
    #max_duration = 27000
    
    # Transform to convert images to tensors and flatten the image pixels
    transform = transforms.Compose([
        transforms.ToTensor(), # converts to (C, H, W) format and scales to [0, 1]
        transforms.Lambda(lambda x: x.view(-1, 1)) # flatten to (784, 1)
    ])

    # Load MNIST dataset
    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # DataLoader
    batch_size = 32 # You can modify this as needed
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_data, batch_size=batch_size)
    
    DTYPE = torch.complex64
    DEVICE=1 #'cpu'
    
    model = ParallelRNNBlock(input_dim=1, out_dim=10, seq_size=784, hid_feature_size=512, hid_seq_size=512, transformer_nhead=1, hid_rnn_size=128, blocks=1, dropout=0.2, seq_expansion= 1, FST=False, transformer= use_transformer).to(DEVICE)
    print(sum(p.numel() for p in model.parameters()))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)
    
    model.to(DEVICE)
    
    nmpara = 0
    for n,p in model.named_parameters():
        if p.requires_grad==True:
            nmpara += np.prod(p.size())

    print(f'Nparams: {nmpara}' )
    
    #fig, (ax1, ax2) = plt.subplots(1, 2)
    #camera = Camera(fig)#
    
    #fig.tight_layout()
    #ax1.axis('off')
    #ax2.axis('off')
    
    results = {'epoch' : [],
        'train_acc' : [],
        'val_acc' : [],
        'mean_train_acc' : [],
        'mean_val_acc' : [],
        'mean_train_acc_last' : [],
        'mean_val_acc_last' : []}
    
    nmaxit = 100
    max_val = 0.5
    
    
    for epoch in tqdm(range(100)):
        model.train()
        print('Training')
        results['epoch'].append(epoch)
    
        _acc = []
        
        total_iterations = nmaxit #len(train_loader)
        #progress_bar = tqdm(enumerate(train_loader), total=total_iterations, desc="Training")
    
        for n, (inputs, target) in enumerate(train_loader):
            optimizer.zero_grad() 

            inputs = inputs.to(torch.float32).to(DEVICE)
            target = target.to(torch.long).to(DEVICE)

            out, h = model(inputs)

            # Change here: remove F.softmax
            loss = F.cross_entropy(out[:, -1, :], target)

            loss.backward()
            optimizer.step()

            _, predicted = torch.max(out[:, -1, :], 1) # Change here: remove F.softmax

            # Calculate accuracy
            current_acc = (predicted == target).float().mean()
            _acc.append(current_acc)

            #progress_bar.set_postfix(accuracy=f'{current_acc:.2f}', refresh=True)
            if n == nmaxit:
                break
        
        results['train_acc'].append(_acc)
        _acc_cpu = [acc_item.cpu() for acc_item in _acc]
        results['mean_train_acc'].append(np.mean(_acc_cpu))
    
        if True:
            print('Validating')
            with torch.no_grad():
                model.eval()
    
                total_iterations = len(val_loader)
                #progress_bar = tqdm(enumerate(val_loader), total=total_iterations, desc="Validating")
    
                _acc = []
    
                for n, (inputs, target) in enumerate(val_loader):
    
                    inputs = inputs.to(torch.float32).to(DEVICE)
                    target = target.to(torch.long).to(DEVICE)

                    out, h = model(inputs)

                    # Change here: remove F.softmax
                    loss = F.cross_entropy(out[:, -1, :], target)

                    _, predicted = torch.max(out[:, -1, :], 1) # Change here: remove F.softmax

                    # Calculate accuracy
                    current_acc = (predicted == target).float().mean()
                    _acc.append(current_acc)

                    #progress_bar.set_postfix(accuracy=f'{current_acc:.2f}', refresh=True)
                    if n == nmaxit:
                        break
            
            results['val_acc'].append(_acc)
            _acc_cpu = [acc_item.cpu() for acc_item in _acc]
            results['mean_val_acc'].append(np.mean(_acc_cpu))
        
            print(results['mean_train_acc'][-1], results['mean_val_acc'][-1])
    
        else:
            print(results['mean_train_acc'][-1])
        
        pickle.dump(results, open(f'results/NEW_MNIST_{use_transformer}_result_{SEED}.pkl', 'wb'))
        



