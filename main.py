import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/src")

import numpy as np
import torch
import random

import argparse
from exps.Exp import Exp

def main():
    
    parser=argparse.ArgumentParser(description="Stacked-LMUFFT as Sequence Encoder")

    # Dataset path
    parser.add_argument('--root_path', type=str, default=os.path.dirname(os.path.abspath(__file__))+'/data/', help='Root path')
    parser.add_argument('--dataset', type=str, default='ETTh1', 
                        help='the name of your data csv file, e.g. ETTh1, ETTh2, ETTm1, ETTm2, weather')

    # Data split
    parser.add_argument('--scale', type=bool, default=True, help='Whether to scale the data')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Train size')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Test size') # validation ratio is going to be inferred automatically from the train and test ratio
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--down_rate', type=int, default=1, help='Downsample rate for the data')

    parser.add_argument('--noise_std', type=float, default=0, help='Noise level, by standard deviation')

    # Model settings
    parser.add_argument('--model', type=str, default='LMUFFT', help='Model name, options: "LMU", "LMUFFT"')
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden size')
    parser.add_argument('--memory_size', type=int, default=256, help='Memory size')
    parser.add_argument('--lmu_theta', type=float, default=1, help='theta for LMU and LMUFFT')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--skip_connection', type=str, default="True", help='Whether to use skip connection')

    # Prediction task settings
    parser.add_argument('--history_size', type=int, default=48, help='History size')
    parser.add_argument('--pred_size', type=int, default=1, help='Prediction size')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--viz', type=str, default="False", help='Whether to show visualization')

    # Optimizer settings
    parser.add_argument('--early_stop', type=int, default=15, help='Early stopping count')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')

    # Save settings
    parser.add_argument('--save_path', type=str, default=os.path.dirname(os.path.abspath(__file__))+"/out/", help='save path')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='whether to use GPU')

    # Set random seed
    parser.add_argument('--seed', type=int, default=25, help='Random seed for experiments')

    args=parser.parse_args()

    # Fix seed for experiments
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Set GPU
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    
    # Create experiment
    experiment = Exp(args) 
    experiment.exp_train()
    experiment.exp_test()


if __name__ == '__main__':
    main()