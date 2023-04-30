import torch
import torch.nn as nn

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ssm_modules.lmu import *

class LMU_Pred(torch.nn.Module):
    """ Original LMU - recurrent implementation"""
    def __init__(self, in_dim, out_len, hidden_size, memory_size, num_layers, theta, device, skip_connection):
        super(LMU_Pred, self).__init__()
        self.num_layers = num_layers
        self.lmu_layers = LMU(in_dim, hidden_size, memory_size, num_layers, theta, device, skip_connection)
        self.fc = torch.nn.Linear(num_layers*hidden_size, out_len)
        self.dropout = nn.Dropout(0.1)
        self.tanh = nn.Tanh()

    def forward(self, x): # x: [batch_size, seq_len, in_dim]
        out, _ = self.lmu_layers(x) # out: [batch_size, seq_len, num_layers*hidden_size]
        out = self.tanh(out)
        out= self.dropout(out) 
        out = self.fc(out[:, -1, :]) # out: [batch_size, out_len]
        return out