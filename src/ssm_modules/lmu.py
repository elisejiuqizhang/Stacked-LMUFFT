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

# if torch.cuda.is_available():
#     device = "cuda"
#     # Clear cache if non-empty
#     torch.cuda.empty_cache()
#     # See which GPU has been allotted 
#     print(torch.cuda.get_device_name(torch.cuda.current_device()))
# else:
#     device = "cpu"


def leCunUniform(tensor):
    """ 
        LeCun Uniform Initializer
        References: 
        [1] https://keras.rstudio.com/reference/initializer_lecun_uniform.html
        [2] Source code of _calculate_correct_fan can be found in https://pytorch.org/docs/stable/_modules/torch/nn/init.html
        [3] Yann A LeCun, Léon Bottou, Genevieve B Orr, and Klaus-Robert Müller. Efficient backprop. In Neural networks: Tricks of the trade, pages 9–48. Springer, 2012
    """
    fan_in = init._calculate_correct_fan(tensor, "fan_in")
    limit = np.sqrt(3. / fan_in)
    init.uniform_(tensor, -limit, limit) # fills the tensor with values sampled from U(-limit, limit)

# ------------------------- LMU Cell ---------------------------
class LMUCell(nn.Module):
    """
        Parameters:
        in_dim (int) : 
            Size of the input vector (x_t)
        hidden_size (int) : 
            Size of the hidden vector (h_t)
        memory_size (int) :
            Size of the memory vector (m_t)
        theta (int) :
            The number of timesteps in the sliding window that is represented using the LTI system, default=1
        device (str) :
            Device to run the model on, default="cuda"
    """
    def __init__(self, in_dim, hidden_size, memory_size, theta=1, device="cuda"):
        super(LMUCell, self).__init__()
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.theta = theta
        self.device = device
        self.A, self.B = self._gen_AB()

        self.f = nn.Tanh()

        # Model Parameters
        ## Encoding vectors
        self.e_x = nn.Parameter(torch.empty(1, in_dim))
        self.e_h = nn.Parameter(torch.empty(1, hidden_size))
        self.e_m = nn.Parameter(torch.empty(1, memory_size))
        ## Kernels
        self.W_x = nn.Parameter(torch.empty(hidden_size, in_dim))
        self.W_h = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.W_m = nn.Parameter(torch.empty(hidden_size, memory_size))  
        self._init_Parameters()

    def forward(self, x, state):
        """
        Parameters:
            x (torch.tensor): 
                Input of size [batch_size, seq_len, in_dim]
            state (tuple):
                h (torch.tensor) : [batch_size, hidden_size]
                m (torch.tensor) : [batch_size, memory_size]
        """
        h, m = state

        # Equation (7) of the paper
        u = F.linear(x, self.e_x) + F.linear(h, self.e_h) + F.linear(m, self.e_m) # [batch_size, 1]

        # Equation (4) of the paper
        m = F.linear(m, self.A) + F.linear(u, self.B) # [batch_size, memory_size]

        # Equation (6) of the paper
        h = self.f(
            F.linear(x, self.W_x) +
            F.linear(h, self.W_h) + 
            F.linear(m, self.W_m)
        ) # [batch_size, hidden_size]

        return (h, m)    


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

        return A.to(self.device), B.to(self.device)

    def _init_Parameters(self):
        """ Initialize the cell's parameters """

        # Initialize encoders
        leCunUniform(self.e_x)
        leCunUniform(self.e_h)
        init.constant_(self.e_m, 0)
        # Initialize kernels
        init.xavier_normal_(self.W_x)
        init.xavier_normal_(self.W_h)
        init.xavier_normal_(self.W_m)

# ------------------------- LMU Layer ---------------------------
class LMU(nn.Module):
    def __init__(self, in_dim, hidden_size, memory_size, num_layers, theta, device, skip_connection=True):

        """ 
            LMU layer

            Parameters:
                in_dim (int) : 
                    Size of the input vector (x_t)
                hidden_size (int) : 
                    Size of the hidden vector (h_t)
                memory_size (int) :
                    Size of the memory vector (m_t)
                num_layers (int) :
                    Number of LMU layers
                theta (int) :
                    The number of timesteps in the sliding window that is represented using the LTI system
                skip_connection (bool) :
                    Whether to use skip connection or not, default=True
        """

        super(LMU, self).__init__()
        self.hidden_size = hidden_size
        self.memory_size = memory_size

        self.num_layers = num_layers
        self.layers= nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(LMUCell(in_dim, hidden_size, memory_size, theta, device))
            else:
                self.layers.append(LMUCell(hidden_size, hidden_size, memory_size, theta, device))

        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=0.1)

        self.skip_connection = skip_connection
        if skip_connection:
            self.skip_fc = nn.Linear(in_dim, hidden_size)

    def forward(self, x, states=None):
        """
        Parameters:
            x (torch.tensor): 
                Input of size [batch_size, seq_len, in_dim]
            states (list of tuples):
                each tuple contains:
                    h (torch.tensor) : [batch_size, hidden_size]
                    m (torch.tensor) : [batch_size, memory_size]
        """
        # Assuming batch_first = True
        batch_size, seq_len, _ = x.shape

        # Initialize hidden and memory states (if not provided)
        if states is None:
            states=[]
            for i in range(self.num_layers):
                h0 = torch.zeros(batch_size, self.hidden_size, device = x.device)
                m0 = torch.zeros(batch_size, self.memory_size, device = x.device)
                states.append((h0, m0))

        # Iterate over the sequence / time steps
        outputs = []
        
        for t in range(seq_len):

            h_t_list = [] # to stack all the hidden states at all levels

            x_t= x[:, t, :] # [batch_size, in_dim]
            identity = self.skip_fc(x_t) if self.skip_connection else 0

            if self.num_layers > 1:
                for i in range(self.num_layers):
                    h_t, m_t = self.layers[i](x_t, states[i]) # h_t: [batch_size, hidden_size]
                    h_t_list.append(h_t)

                    x_t = h_t+identity
                    x_t = self.tanh(x_t)
                    x_t= self.dropout(x_t)
                    states[i] = (h_t, m_t)
            else:
                h_t, m_t = self.layers[i](x_t, states[0])
                h_t_list.append(h_t)

            h_t_stack = torch.cat(h_t_list, dim = 1) # [batch_size, num_layers*hidden_size]
            
            outputs.append(h_t_stack)
            output= torch.stack(outputs, dim = 1) # [batch_size, seq_len, num_layers*hidden_size]

        
        return output, states 
