import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class BasicDataset(Dataset):
    def __init__(self, data_path, len_history, len_pred, target_col="OT", scale=True, down_rate=1, noise_std=0):
        """
        Parameters:
        data_path (str) : 
            Path to the data
        len_history (int) : 
            Length of the history sequence
        len_pred (int) :
            Length of the prediction sequence
        target_col (str) :
            Name of the target column
        scale (bool) :
            Whether to scale the data, default is True
        down_rate (int) :
            Downsample rate for the data
        """

        self.down_rate = down_rate
        self.len_history = len_history
        self.len_pred = len_pred
        self.target_col = target_col
        self.scale=scale 

        self.df = pd.read_csv(data_path, index_col=0, delimiter=",").astype(np.float32)
        # inject noise if needed
        if noise_std>0:
            self.df += np.random.normal(0, noise_std, self.df.shape)
        self.df=self.df.astype(np.float32)

        self.df.index = pd.to_datetime(self.df.index)
        self.raw_len = len(self.df)       

        self.df = self.__downsample(self.df) # to be scaled
        self.target= self.df[self.target_col].values # target won't be scaled

    def __downsample(self, df):
        """ Downsample the data by rolling mean"""
        if self.down_rate==1:
            return df
        else:
            return df.groupby(pd.Grouper(freq='{}H'.format(self.down_rate))).mean()

    def __len__(self):
        """ Length of the dataset"""
        if self.down_rate==1: # No downsampling
            return self.raw_len - self.len_history - self.len_pred + 1
        else: # Downsampled
            return self.raw_len // self.down_rate - self.len_history - self.len_pred + 1

    def __scale__(self, data):
        """ Scale the data"""
        self.data_scaler = StandardScaler()
        self.data_scaler.fit(data)
        return self.data_scaler.transform(data)

    def __getitem__(self, idx):
        """ Get the data at index idx"""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get the list of column names
        columns = list(self.df.columns)
        # Get the data
        data = self.df[columns].values
        # Scale the data
        if self.scale:
            data = self.__scale__(data)
        # Get the history sequence (features - [len_history, num_features]])
        history = data[idx:idx+self.len_history]
        # Get the prediction sequence (target - [len_pred, 1])
        pred = self.target[idx+self.len_history:idx+self.len_history+self.len_pred]
        # Get the time stamp
        timestamp = self.df.index[idx+self.len_history]

        # sample = {'history': history, 'pred': pred, 'timestamp': timestamp}
        sample = {'history': history, 'pred': pred}

        return sample 
