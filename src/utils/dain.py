# Normalization layer "DAIN" from https://github.com/passalis/dain
# Paper https://arxiv.org/pdf/1902.07892.pdf

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class DAIN_Layer(nn.Module):
    def __init__(self, mode='adaptive_avg', mean_lr=0.00001, gate_lr=0.001, scale_lr=0.00001, input_dim=144):
        super(DAIN_Layer, self).__init__()
        print("Mode = ", mode)

        self.mode = mode
        self.mean_lr = mean_lr
        self.gate_lr = gate_lr
        self.scale_lr = scale_lr

        # Parameters for adaptive average
        self.mean_layer = nn.Linear(input_dim, input_dim, bias=False)
        self.mean_layer.weight.data = torch.FloatTensor(data=np.eye(input_dim, input_dim))

        # Parameters for adaptive std
        self.scaling_layer = nn.Linear(input_dim, input_dim, bias=False)
        self.scaling_layer.weight.data = torch.FloatTensor(data=np.eye(input_dim, input_dim))

        # Parameters for adaptive scaling
        self.gating_layer = nn.Linear(input_dim, input_dim)

        self.eps = 1e-8

    def forward(self, x):
        # Expecting  (batch_size, input_dim, input_len)

        # Nothing to normalize
        if self.mode == None:
            pass

        # Do simple average normalization
        elif self.mode == 'avg':
            avg=torch.mean(x,1)
            avg=torch.reshape(avg,(avg.size(0),1,avg.size(1)))
            x=x-avg

        # Perform only the first step (adaptive averaging)
        elif self.mode == 'adaptive_avg':
            avg=torch.mean(x,1)
            adaptive_avg=self.mean_layer(avg)
            adaptive_avg=torch.reshape(adaptive_avg,(adaptive_avg.size(0), 1, adaptive_avg.size(1)))
            x=x-adaptive_avg

        # Perform the first + second step (adaptive averaging + adaptive scaling )
        elif self.mode == 'adaptive_scale':

            # Step 1:
            avg=torch.mean(x,1)
            adaptive_avg=self.mean_layer(avg)
            adaptive_avg=torch.reshape(adaptive_avg,(adaptive_avg.size(0), 1, adaptive_avg.size(1)))
            x=x-adaptive_avg

            # Step 2:
            std=torch.mean(x**2,1)
            std=torch.sqrt(std+self.eps)
            adaptive_std=self.scaling_layer(std)
            adaptive_std[adaptive_std<=self.eps]=1

            adaptive_std=torch.reshape(adaptive_std,(adaptive_std.size(0), 1, adaptive_avg.size(2)))
            x=x/(adaptive_std)

        elif self.mode == 'full':

            # Step 1:
            avg=torch.mean(x,1)
            adaptive_avg=self.mean_layer(avg)
            adaptive_avg=torch.reshape(adaptive_avg,(adaptive_avg.size(0), 1, adaptive_avg.size(1)))
            x=x-adaptive_avg

            # Step 2:
            std=torch.mean(x**2,1)
            std=torch.sqrt(std+self.eps)
            adaptive_std=self.scaling_layer(std)
            adaptive_std[adaptive_std<=self.eps]=1

            adaptive_std=torch.reshape(adaptive_std,(adaptive_std.size(0), 1, adaptive_avg.size(2)))
            x=x/adaptive_std

            # Step 3:
            avg=torch.mean(x,1)
            gate=torch.sigmoid(self.gating_layer(avg))
            gate=torch.reshape(gate,(gate.size(0), 1, adaptive_avg.size(2)))
            x=x*gate

        else:
            assert False

        return x
