""" Training step, vaidation step and evaluation step"""
import numpy as np
import torch
import matplotlib.pyplot as plt

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.metrics import *

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ------------------- Early Stopping -------------------
class EarlyStopping:
    def __init__(self, patience=15, verbose=False, delta=0):
        """
        Early stops the training if validation loss doesn't improve after a given patience.
        Parameters:
        ----------
            patience (int): How long to wait after last time validation loss improved.
                            Default: 15
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
    
    def save_checkpoint(self, val_loss, model, save_path):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss has decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), save_path+'.pt')
        self.val_loss_min = val_loss
        
    def __call__(self, val_loss, model, save_path, model_name):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, save_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping count: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, save_path)
            self.counter = 0



#------------------- Training Epoch -------------------
def train(model, dataloader, optimizer, criterion, device):
    """ One epoch of training """
    epoch_loss = 0 # the criterion loss
    epoch_smape = 0 # the (s-)MAPE
    y_true = []
    y_pred = []

    model.train()
    for sample in dataloader:
        torch.cuda.empty_cache()

        # Get the data
        history = sample['history'].to(device)
        pred = sample['pred'].squeeze(1).to(device)

        # Forward pass
        output = model(history)
        output = output.squeeze(1) # [batch_size, pred_size]
        
        loss = criterion(output, pred) 

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Save the loss
        epoch_loss += loss.item()
        epoch_smape += smape(pred.cpu().detach().numpy(), output.cpu().detach().numpy())

        # Save the predictions
        y_true.append(pred.cpu().detach().numpy())
        y_pred.append(output.cpu().detach().numpy())

    return epoch_loss/len(dataloader), epoch_smape/len(dataloader),  y_true, y_pred # Return the average loss, average (s-)MAPE, the true sequence and the predictions

#------------------- Validation Epoch -------------------
def validate(model, dataloader, criterion, device):
    """ One epoch of validation """
    epoch_loss = 0 # the criterion loss
    epoch_smape = 0 # the (s-)MAPE
    y_true = []
    y_pred = []

    model.eval()
    with torch.no_grad():
        for sample in dataloader:
            torch.cuda.empty_cache()

            # Get the data
            history = sample['history'].to(device)
            pred = sample['pred'].squeeze(1).to(device)

            # Forward pass
            output= model(history)
            output = output.squeeze(1) # [batch_size, pred_size]
            
            loss = criterion(output, pred) 

            # Save the loss 
            epoch_loss += loss.item() 
            epoch_smape += smape(pred.cpu().detach().numpy(), output.cpu().detach().numpy())

            # Save the predictions
            y_true.append(pred.cpu().detach().numpy())
            y_pred.append(output.cpu().detach().numpy())

    return epoch_loss/len(dataloader), epoch_smape/len(dataloader), y_true, y_pred # Return the average loss, average (s-)MAPE, the true sequence and the predictions

#------------------- Evaluation on test set (No visualization) -------------------
def predict_eval(model, dataloader, pred_size, device, save_path=None):
    """ use the metrics as follows: rMSE, MAE, (s)MAPE """
    
    epoch_mse = 0
    epoch_mae = 0
    epoch_smape = 0
    outputs = []
    y_true = []
    model.eval()

    with torch.no_grad():
        for sample in dataloader:
            torch.cuda.empty_cache()

            # Get the data
            history = sample['history'].to(device)
            pred = sample['pred'].squeeze(1).to(device)

            # Forward pass
            output= model(history) # [batch_size, pred_size]
            output = output.squeeze(1) 

            output = output.cpu().detach() # [batch_size, pred_size]
            pred = pred.cpu().detach() # [batch_size, pred_size]

            # Save the loss 
            epoch_mse += mse(pred.numpy(), output.numpy())
            epoch_mae += mae(pred.numpy(), output.numpy())
            epoch_smape += smape(pred.numpy(), output.numpy())

            # Save the predictions
            if(pred_size == 1):
                # single step
                outputs.append(output)
                y_true.append(pred)
            else:
            # multi-step
                if len(outputs) == 0: # First prediction, hence empty the list - append all the elements to outputs and y_true
                    for i in range(output.shape[0]):
                        outputs.append(output[i])
                        y_true.append(pred[i])
                else: # Append only the last element to the existing list
                    outputs.append(output[:,-1])
                    y_true.append(pred[:,-1])

    outputs = np.concatenate(outputs)
    y_true = np.concatenate(y_true)

    #print the average loss
    if save_path is None:
        print('\nTest rMSE: {:.4f}, MAE: {:.4f}, (s-)MAPE: {:.4f}'.format(np.sqrt(epoch_mse/len(dataloader)), epoch_mae/len(dataloader), epoch_smape/len(dataloader)))
    else:
        with open(save_path+".log", 'a') as f:
            print('\nTest rMSE: {:.4f}, MAE: {:.4f}, (s-)MAPE: {:.4f}'.format(np.sqrt(epoch_mse/len(dataloader)), epoch_mae/len(dataloader), epoch_smape/len(dataloader)), file=f)

    return outputs, y_true

#------------------- Evaluation on test set (With visualization) -------------------
def predict_viz_eval(model, dataloader, pred_size, device, save_path=None):
    """ use the metrics as follows: rMSE, MAE, (s)MAPE """
    
    epoch_mse = 0
    epoch_mae = 0
    epoch_smape = 0
    outputs = []
    y_true = []
    model.eval()

    with torch.no_grad():
        for sample in dataloader:
            torch.cuda.empty_cache()

            # Get the data
            history = sample['history'].to(device)
            pred = sample['pred'].squeeze(1).to(device)

            # Forward pass
            output= model(history) # [batch_size, pred_size]
            output = output.squeeze(1) 

            output = output.cpu().detach() # [batch_size, pred_size]
            pred = pred.cpu().detach() # [batch_size, pred_size]

            # Save the loss 
            epoch_mse += mse(pred.numpy(), output.numpy())
            epoch_mae += mae(pred.numpy(), output.numpy())
            epoch_smape += smape(pred.numpy(), output.numpy())

            # Save the predictions
            if(pred_size == 1):
                # single step
                outputs.append(output)
                y_true.append(pred)
            else:
            # multi-step
                if len(outputs) == 0: # First prediction, hence empty the list - append all the elements to outputs and y_true
                    for i in range(output.shape[0]):
                        outputs.append(output[i])
                        y_true.append(pred[i])
                else: # Append only the last element to the existing list
                    outputs.append(output[:,-1])
                    y_true.append(pred[:,-1])

    outputs = np.concatenate(outputs)
    y_true = np.concatenate(y_true)

    #print the average loss
    if save_path is None:
        print('\nTest rMSE: {:.4f}, MAE: {:.4f}, (s-)MAPE: {:.4f}'.format(np.sqrt(epoch_mse/len(dataloader)), epoch_mae/len(dataloader), epoch_smape/len(dataloader)))
    else:
        with open(save_path+".log", 'a') as f:
            print('\nTest rMSE: {:.4f}, MAE: {:.4f}, (s-)MAPE: {:.4f}'.format(np.sqrt(epoch_mse/len(dataloader)), epoch_mae/len(dataloader), epoch_smape/len(dataloader)), file=f)

    # Plot the results
    plt.figure(figsize=(15, 5))
    plt.plot(y_true, label='True')
    plt.plot(outputs, label='Predicted')
    plt.legend()
    # plt.show()
    plt.savefig(save_path+'.png')

    return outputs, y_true
    