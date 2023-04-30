import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import time # to track runtime
from datetime import datetime # to track date and time while saving logs

from data_loaders.basic_dataloader import *
from datetime import datetime
from models.LMU_Pred import *
from models.LMUFFT_Pred import *

from utils.metrics import *
from utils.train_eval_utils import *

class Exp(object):
    def __init__(self, args):
        self.args = args
        self.device = self._get_device()

        self.skip_connection=self._get_bool_str(self.args.skip_connection)
        self.viz=self._get_bool_str(self.args.viz)
        
        self.input_dim=self._get_input_dim()
        self.train_loader, self.val_loader, self.test_loader= self._load_data()
        
        self.model=self._build_model().to(self.device)
        
        self.save_path=self._get_save_path()

    def _get_bool_str(self, s):
        if s=="True":
            return True
        elif s=="False":
            return False
        else:
            raise ValueError("Invalid boolean string")
    
    def _get_device(self):
        if self.args.use_gpu:
            torch.cuda.empty_cache()
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")
        return device

    def _get_save_path(self):
        # saving location
        # will be under "out" folder, inside be like "dataset/model_name/", 
        # log name contains date and time, 
        log_path = self.args.save_path+"/"+self.args.dataset+"/"+self.args.model+"/"
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        timestr = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")

        # naming: timestr, model, theta, dataset, batch_size, history size, pred size, if viz, if skip (more than 1 layer), hidden size, memory size, num_layers, seed
        if self.args.num_layers>1:
            filename = timestr+"_"+self.args.model+"_theta"+str(self.args.lmu_theta)+"_"+self.args.dataset+"_batch"+str(self.args.batch_size)+"_hist"+str(self.args.history_size)+"_pred"+str(self.args.pred_size)+"_viz"+str(self.viz)+"_skip"+str(self.skip_connection)+"_hid"+str(self.args.hidden_size)+"_mem"+str(self.args.memory_size)+"_numlayers"+str(self.args.num_layers)+"_seed"+str(self.args.seed)
        else:
            filename = timestr+"_"+self.args.model+"_theta"+str(self.args.lmu_theta)+"_"+self.args.dataset+"_batch"+str(self.args.batch_size)+"_hist"+str(self.args.history_size)+"_pred"+str(self.args.pred_size)+"_viz"+str(self.viz)+"_hid"+str(self.args.hidden_size)+"_mem"+str(self.args.memory_size)+"_numlayers"+str(self.args.num_layers)+"_seed"+str(self.args.seed)

        save_path = log_path+filename # add different extensions for different files
        return save_path

    def _load_data(self):
        """ Load data"""

        # get the data path
        data_path = self.args.root_path+self.args.dataset+".csv"

        # load the data
        raw_dataset=BasicDataset(data_path=data_path, len_history=self.args.history_size, len_pred=self.args.pred_size, scale=self.args.scale, down_rate=self.args.down_rate, noise_std=self.args.noise_std)

        # specify train, val, set sizes
        training_size = int(self.args.train_ratio * len(raw_dataset))
        test_size = int(self.args.test_ratio * len(raw_dataset))
        val_size=len(raw_dataset)-training_size-test_size

        if self.viz is False: # if not going to need visualization, can shuffle
            # create dataloaders
            train_set, val_set, test_set = torch.utils.data.random_split(raw_dataset, [training_size, val_size, test_size])
            train_loader=DataLoader(train_set, batch_size=self.args.batch_size, shuffle=True)
            val_loader=DataLoader(val_set, batch_size=self.args.batch_size, shuffle=True)
            test_loader=DataLoader(test_set, batch_size=self.args.batch_size, shuffle=False)

        else: # if we want to visualize the data - no shuffle
            # get a continuous subset for testing with no shuffling
            test_start_idx = random.randint(0, len(raw_dataset)) # get a random starting index for the test set
            test_set = torch.utils.data.Subset(raw_dataset, range(test_start_idx, test_start_idx+test_size))
            test_loader=DataLoader(test_set, batch_size=self.args.batch_size, shuffle=False)
            # concatenate the rest of the data for training and validation
            subset1 = torch.utils.data.Subset(raw_dataset, range(0, test_start_idx))
            subset2 = torch.utils.data.Subset(raw_dataset, range(test_start_idx+test_size, len(raw_dataset)))
            temp = torch.utils.data.ConcatDataset([subset1, subset2])
            # create train and val dataloaders
            train_set, val_set = torch.utils.data.random_split(temp, [training_size, val_size])
            train_loader=DataLoader(train_set, batch_size=self.args.batch_size, shuffle=True)
            val_loader=DataLoader(val_set, batch_size=self.args.batch_size, shuffle=True)

        return train_loader, val_loader, test_loader
    
    def _get_input_dim(self):
        dataset_dict = {"ETTh1": 7, 
                        "ETTh2": 7, 
                        "ETTm1": 7, 
                        "ETTm2": 7, 
                        "exchange_rate":8,
                        "weather":21}
        return dataset_dict[self.args.dataset]
    
    def _build_model(self):
        input_dim = self._get_input_dim()
        model_dict = {"LMU": LMU_Pred(in_dim=input_dim, out_len=self.args.pred_size, hidden_size=self.args.hidden_size, memory_size=self.args.memory_size, num_layers=self.args.num_layers, theta=self.args.lmu_theta, device=self.device, skip_connection=self.skip_connection),
                      "LMUFFT": LMUFFT_Pred(in_dim=input_dim, out_len=self.args.pred_size, hidden_size=self.args.hidden_size, memory_size=self.args.memory_size, in_len=self.args.history_size, num_layers=self.args.num_layers, theta=self.args.lmu_theta, device=self.device, skip_connection=self.skip_connection),
                     }
        return model_dict[self.args.model]
    
    def exp_train(self):

        earlystopping = EarlyStopping(patience=self.args.early_stop, verbose=True)

        # optimizer and criterion
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr)
        criterion = torch.nn.L1Loss()

        # train model
        time_start = time.time() # track the training time - start

        train_losses = []
        val_losses = []

        for epoch in range(self.args.epochs):
            epoch_start = time.time() # track the per-epoch training time - start
            # Train
            train_loss, train_mape, y_true_train, y_pred_train = train(self.model, self.train_loader, optimizer, criterion, device=self.device)
            train_losses.append(train_loss)

            # Validate
            val_loss, val_mape, y_true_val, y_pred_val = validate(self.model, self.val_loader, criterion, device=self.device)
            val_losses.append(val_loss)
            epoch_train_end = time.time() # track the per-epoch training time - end

            epoch_train_end = time.time() # track the per-epoch training time - end

            # Print
            with open(self.save_path+".log", 'a') as f:
                print("Epoch: {}/{} \t Avg. Train Loss: {:.4f}; Avg. Train sMAPE: {:.4f}; \t Avg. Val Loss: {:.4f}; Avg. Val sMAPE: {:.4f}".format(epoch+1, self.args.epochs, train_loss, train_mape, val_loss, val_mape), file=f)
                print("Epoch {} Spent Time: {:.4f} seconds".format(epoch, epoch_train_end-epoch_start), file=f)
            
            # Early Stopping
            earlystopping(val_loss, self.model, self.save_path, model_name=self.args.model)
            if earlystopping.early_stop:
                print("Early stopping!")
                break

        time_end = time.time() # track the training time - end
        with open(self.save_path+".log", 'a') as f:
            print("\nTotal Training Time: {:.4f} seconds".format(time_end-time_start), file=f)
            print("Average Training Time per Epoch: {:.4f} seconds".format((time_end-time_start)/(epoch+1)), file=f)


    def exp_test(self):
        # load model
        self.model.load_state_dict(torch.load(self.save_path+".pt"))

        if self.viz is True:
            predict_viz_eval(self.model, self.test_loader, self.args.pred_size, device=self.device, save_path=self.save_path)
        else:
            predict_eval(self.model, self.test_loader, self.args.pred_size, device=self.device, save_path=self.save_path)

