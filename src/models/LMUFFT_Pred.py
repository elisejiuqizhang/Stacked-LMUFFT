import torch
import torch.nn as nn

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ssm_modules.lmufft import *

class LMUFFT_Pred(nn.Module):
    """ LMU parallelized implementation using FFT"""
    def __init__(self, in_dim, out_len, hidden_size, memory_size, in_len, num_layers, theta, device, skip_connection=True):
        super(LMUFFT_Pred, self).__init__()
        self.dain = DAIN_Layer(mode='full', input_dim=in_dim)
        self.num_layers = num_layers
        self.lmufft_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.lmufft_layers.append(LMUFFT(in_dim, hidden_size, memory_size, in_len, theta, device))
            else:
                self.lmufft_layers.append(LMUFFT(hidden_size, hidden_size, memory_size, in_len, theta, device))
        self.dropout = nn.Dropout(0.1)
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(num_layers*hidden_size, out_len).to(device)
        self.skip_connection = skip_connection
        if skip_connection and num_layers > 1:
            self.skip_fc = nn.Linear(in_dim, hidden_size).to(device)

    def forward(self, x): # [batch_size, seq_len, in_dim]
        identity = self.skip_fc(x) if self.skip_connection and self.num_layers>1 else 0 # skip connection: [batch_size, seq_len, hidden_size]
        x = self.dain(x)
        if self.num_layers > 1:
            h_n_list = [] # to stack all the hidden states at all levels
            for i in range(self.num_layers):

                if i<self.num_layers-1:
                    x, h_n = self.lmufft_layers[i](x) # x: [batch_size, seq_len, hidden_size]; h_n: [batch_size, hidden_size]
                    h_n_list.append(h_n)
                    x = x + identity 
                    x=self.tanh(x)
                    x=self.dropout(x)
                else:
                    x, h_n = self.lmufft_layers[i](x) # x: [batch_size, seq_len, hidden_size]; h_n: [batch_size, hidden_size]
                    h_n_list.append(h_n)
                
            h_n = torch.cat(h_n_list, dim=1) # [batch_size, num_layers*hidden_size]

        else:
            x, h_n = self.lmufft_layers[0](x) # x: [batch_size, seq_len, hidden_size]; h_n: [batch_size, hidden_size]


        out=self.tanh(h_n) # [batch_size, num_layers*hidden_size]
        out=self.dropout(h_n) # [batch_size, num_layers*hidden_size]
        out=self.fc(out) # out: [batch_size, out_len]
        
        return out 
