# Implementation References: 
# [1] https://github.com/hrshtv/pytorch-lmu
# [2] https://github.com/nengo/keras-lmu


import numpy as np

import torch
from torch import nn
from torch import fft
from torch.nn import init
from torch.nn import functional as F

from scipy.signal import cont2discrete



class LMUFFT(nn.Module):
    """
        Parameters:
        in_dim (int) : 
            Size of the input vector (x_t)
        hidden_size (int) : 
            Size of the hidden vector (h_t)
        memory_size (int) :
            Size of the memory vector (m_t)
        in_len (int) :
            Size of the sequence length (n)
        theta (int) :
            The number of timesteps in the sliding window that is represented using the LTI system, default=1
        device (str) :
            Device to run the model on, default="cuda
    """
    def __init__(self, in_dim, hidden_size, memory_size, in_len, theta=1, device="cuda"):
        super(LMUFFT, self).__init__()
        
        self.in_dim = in_dim
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.in_len = in_len
        self.theta = theta
        self.device = device

        self.A, self.B = self._gen_AB()
        self.H, self.H_fft = self._impulse()

        # self.W_u=nn.Linear(in_features = self.in_dim, out_features = 1)
        self.W_u=nn.Parameter(torch.randn(1, self.in_dim)).to(self.device)

        # self.W_h=nn.Linear(in_features = self.memory_size + self.in_dim, out_features = self.hidden_size)
        self.W_h=nn.Parameter(torch.randn(self.hidden_size, self.memory_size + self.in_dim)).to(self.device)

        self.f_u=nn.ReLU()
        self.f_h=nn.ReLU()


    def forward(self, x):
        """
        Parameters:
            x (torch.tensor): 
                Input of size [batch_size, in_len, in_dim];

        Returns:
            h (torch.tensor):
                Output of size [batch_size, in_len, hidden_size]; The parallelized/flattened hidden states of every timestep;
            h_n (torch.tensor):
                Output of size [batch_size, hidden_size]; The hidden state of the last timestep;
        """
        
        x=x.reshape(x.shape[0],x.shape[1],-1)
        batch_size, in_len, in_dim = x.shape


        # Equation 18 of the paper
        u=F.linear(x, self.W_u) # [batch_size, in_len, 1]
        u = self.f_u(u) # [batch_size, in_len, 1]

         # Equation 26 of the paper
        fft_input = u.permute(0, 2, 1) # [batch_size, 1, in_len]
        u_fft = fft.rfft(fft_input, n = 2*in_len, dim = -1) # [batch_size, in_len, in_len+1]

        # Element-wise multiplication (uses broadcasting)
        # fft_u:[batch_size, 1, in_len+1] 
        # self.H_fft: [memory_size, in_len+1] -> to be expanded in dimension 0
        # [batch_size, 1, in_len+1] * [1, memory_size, in_len+1]
        temp = u_fft * self.H_fft.unsqueeze(0) # [batch_size, memory_size, in_len+1]

        m = fft.irfft(temp, n = 2*in_len, dim = -1) # [batch_size, memory_size, in_len+1]
        m = m[:, :, :in_len] # [batch_size, memory_size, in_len]
        m = m.permute(0, 2, 1) # [batch_size, in_len, memory_size]

        # Equation 20 of the paper (W_m@m + W_x@x  W@[m;x])
        input_h = torch.cat((m, x), dim = -1) # [batch_size, in_len, memory_size + in_dim]

        # h = self.f_h(self.W_h(input_h)) # [batch_size, in_len, hidden_size]
        h=F.linear(input_h, self.W_h)
        h = self.f_h(h) # all hidden_states: [batch_size, in_len, hidden_size]

        h_n = h[:, -1, :] # hidden_state at the last time stamp: [batch_size, hidden_size]

        return h, h_n
        
    
    def _gen_AB(self):
        """
        Pade approximants of the state space matrices.

        Reference:
        [1] Partington, Jonathan R. "Some frequency-domain approaches to the model reduction of delay systems." Annual Reviews in Control 28.1 (2004): 65-73.
        [2] Voelker, Aaron R., and Chris Eliasmith. "Improving spiking dynamical networks: Accurate delays, higher-order synapses, and time cells." Neural computation 30.3 (2018): 569-609.
        """

        Q = np.arange(self.memory_size, dtype = np.float64).reshape(-1, 1)
        R = (2 * Q + 1)/self.theta
        i, j = np.meshgrid(Q, Q, indexing = "ij")

        A = np.where(i < j, -1, (-1.0) ** (i - j + 1)) * R
        B = ((-1.0) ** Q) * R

        # discretization
        A, B, _, _, _ = cont2discrete((A, B,np.eye(self.memory_size), np.zeros(self.memory_size)), dt = 1.0, method = "zoh")

        A = torch.from_numpy(A).float()
        B = torch.from_numpy(B).float()

        return A, B

    def _impulse(self):
        """
        Returns:
            H (torch.tensor): Matrix; Impulse response of the LTI system
            H_fft (torch.tensor): Matrix; FFT of the impulse response
        """

        H=torch.empty((self.memory_size, self.in_len)) # [memory_size, in_len]
        A_i=torch.eye(self.memory_size)

        for i in range(self.in_len):
            H_i = torch.matmul(A_i, self.B)
            H[:,i]=H_i.reshape(-1)
            A_i = torch.matmul(self.A, A_i)

        H_fft=torch.fft.rfft(H,n = 2*self.in_len, dim = -1) # [memory_size, in_len + 1]

        return H.to(self.device), H_fft.to(self.device)