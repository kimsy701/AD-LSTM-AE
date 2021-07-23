#https://curiousily.com/posts/time-series-anomaly-detection-using-lstm-autoencoder-with-pytorch-in-python/
#https://data-newbie.tistory.com/567

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from LSTM import RecurrentAutoencoder
import torch.nn as nn
import torch
import copy
from torch.utils.data import DataLoader
import random
import torch.backends.cudnn as cudnn
#from pytorchtools import EarlyStopping

###EarlyStopping
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


###########dataset with dataloader
from torch.utils.data import Dataset

train_dataset_x = pd.read_csv('dataC_train_x.csv', index_col=0)
train_dataset = torch.tensor(train_dataset_x.values)
train_dataset = train_dataset.unsqueeze(dim=2)
train_dataset = train_dataset.float()
"""
class timeseries(Dataset):
    def __init__(self,x):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.len = x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx]
    def __len__(self):
        return self.len

train_dataset = timeseries(train_dataset_x.values)
"""

#bring validation data
val_dataset_x = pd.read_csv('dataC_val_x.csv', index_col=0)
val_dataset = torch.tensor(val_dataset_x.values)
val_dataset = val_dataset.unsqueeze(dim=2)
val_dataset = val_dataset.float()

# LSTM Autoencoder
timesteps = 48
epochs = 2
batch = 128
lr = 0.001
seq_len =timesteps
n_features=1
patience =7


"""
#bring train data with dataloader
from torch.utils.data import DataLoader
train_loader = DataLoader(train_dataset, shuffle =False, batch_size = 128)

#train the model
def train_model(model, train_dataset, n_epochs):
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  criterion = nn.L1Loss(reduction='sum').to(device)
  history = dict(train=[], val=[])
  best_model_wts = copy.deepcopy(model.state_dict())
  best_loss = 10000.0

  early_stopping = EarlyStopping(patience = patience, verbose=True)

  train_losses=[]
  valid_losses=[]
  avg_train_losses=[]
  avg_valid_losses=[]

  for epoch in range(1, n_epochs + 1):
      #train the model
    model = model.train()
    train_losses = []#for seq_true in train_dataset:
    for j, seq_true in enumerate(train_loader,1): #train dataset을 batch 단위로 만들
      optimizer.zero_grad()
      seq_true = seq_true.to(device)
      seq_pred = model(seq_true)
      loss = criterion(seq_pred, seq_true)
      loss.backward()
      optimizer.step()
      train_losses.append(loss.item())

          #validate the model
    model.eval()
    for j, data in enumerate(val_dataset,1) : #valid data to batch
        output = model(data)
        loss = criterion(output, data)
        valid_losses.append(loss.item())

    train_loss = np.average(train_losses)
    valid_loss = np.average(valid_losses)
    avg_train_losses.append(train_loss)
    avg_valid_losses.append(valid_loss)

    epoch_len = len(str(n_epochs))

    print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}]'+
                 f'train_loss:{train_loss: .5f}'+
                 f'valid_loss :{valid_loss:.5f}')
    print(print_msg)

    #clear lists to tack next epoch
    train_losses = []
    valid_losses=[]

    #early stopping needs the validation loss to check if it has decreased
    #and if it has, it will make a checkpoint of the current model
    early_stopping(valid_loss,model)

    if early_stopping.early_stop:
        print("Early Stopping")
        break

  model.load_state_dict((torch.load('checkpoint.pt')))

  return model.eval(), history, model, avg_train_losses, avg_valid_losses

#train_dataset (2d ->3d)
#train_dataset = train_dataset.unsqueeze(dim=2)

#train
model, history = train_model(
  model,
  train_dataset,
  n_epochs=1
)
"""

#DataLoader
class AutoencoderDataset(Dataset):
    def __init__(self,x):
        self.x = x
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x[idx,:,:])
        return x

#training
def train_model(model, train_dataset, val_dataset, n_epochs,batch_size):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.L1Loss(reduction='sum').to(device)
    history = dict(train=[], val=[])
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000.0
    print("start!")
    train_dataset_ae = AutoencoderDataset(train_dataset)
    tr_dataloader = DataLoader(train_dataset_ae, batch_size=batch_size,
                               shuffle=False,num_workers=4)
    val_dataset_ae = AutoencoderDataset(val_dataset)
    va_dataloader = DataLoader(val_dataset_ae, batch_size=len(val_dataset),
                               shuffle=False,num_workers=4)
    for epoch in range(1, n_epochs + 1):
        model = model.train()
        train_losses = []
        for batch_idx, batch_x in enumerate(tr_dataloader):
            optimizer.zero_grad()
            batch_x_tensor = batch_x.to(device)
            seq_pred = model(batch_x_tensor)
            loss = criterion(seq_pred, batch_x_tensor)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        val_losses = []
        model = model.eval()
        with torch.no_grad():
            va_x  =next(va_dataloader.__iter__())
            va_x_tensor = va_x.to(device)
            seq_pred = model(va_x_tensor)
            loss = criterion(seq_pred, va_x_tensor)
            val_losses.append(loss.item())
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history['train'].append(train_loss)
        history['val'].append(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
        print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')

    model.load_state_dict(best_model_wts)

    return model, model.eval()

area_list=[]

#5번 반복실험
for i in range(5):

    seed= 10**i
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark=True


    model = RecurrentAutoencoder(seq_len, n_features, 128)
    model = model.to(device)

    model, history = train_model(model, train_dataset , val_dataset , n_epochs = epochs, batch_size=batch)

    #save model
    MODEL_PATH = 'dataC_model.pth'
    torch.save(model, MODEL_PATH)

    #bring test data
    test_dataset_x = pd.read_csv('dataC_test_x.csv', index_col=0)
    test_dataset = torch.tensor(test_dataset_x.values)
    test_dataset = test_dataset.unsqueeze(dim=2)
    test_dataset = test_dataset.float()

    y_test = pd.read_csv('dataC_test_y.csv', index_col=0)

    """
    #predict function
    #predict
    def predict(model, dataset):
      predictions, losses = [], []
      criterion = nn.L1Loss(reduction='sum').to(device)
      with torch.no_grad():
        model = model.eval()
        for seq_true in dataset:
          seq_true = seq_true.to(device)
          seq_pred = model(seq_true)
          loss = criterion(seq_pred, seq_true)
          predictions.append(seq_pred.cpu().numpy().flatten())
          losses.append(loss.item())
      return predictions, losses
    
    #predict
    predictions, pred_losses = predict(model, test_dataset)
    """

    test_dataset_ae = AutoencoderDataset(test_dataset)
    test_dataloader = DataLoader(test_dataset_ae, batch_size=len(test_dataset),
                                shuffle=False, num_workers=4)

    test_losses=[]
    model=model.eval()
    with torch.no_grad():
        test_x = next(test_dataloader.__iter__())
        test_x_tensor = test_x.to(device)
        test_seq_pred = model(test_x_tensor)


    #reconstruction error인

    #prediction from (1344144,) to (28003,48)
    predictions = np.array(test_seq_pred.cpu())
    predictions = predictions.reshape(test_seq_pred.shape[0],test_seq_pred.shape[1])

    mse = np.mean(np.power(np.array(test_dataset.squeeze()) - predictions, 2), axis=1)

    error_df = pd.DataFrame({'Reconstruction_error': mse,
                             'True_class': y_test[:test_seq_pred.shape[0]].values.squeeze()})
    groups = error_df.groupby('True_class')

    loss_list = mse.tolist()



    ###성능 내기###
    def roc(loss_list, threshold):
        test_score_df = pd.DataFrame(index=range(len(loss_list)))
        test_score_df['loss'] = [loss / timesteps for loss in loss_list]  # 29027
        test_score_df['y'] =  y_test[:test_seq_pred.shape[0]].values.squeeze()
        test_score_df['threshold'] = threshold
        test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold
        #test_score_df['t'] = [x[47] for x in test_dataset.x]  # x[59]

        start_end = []
        state = 0
        for idx in test_score_df.index:
            if state == 0 and test_score_df.loc[idx, 'y'] == 1:
                state = 1
                start = idx
            if state == 1 and test_score_df.loc[idx, 'y'] == 0:
                state = 0
                end = idx
                start_end.append((start, end))

        for s_e in start_end:
            if sum(test_score_df[s_e[0]:s_e[1] + 1]['anomaly']) > 0:
                for i in range(s_e[0], s_e[1] + 1):
                    test_score_df.loc[i, 'anomaly'] = 1

        actual = np.array(test_score_df['y'])
        predicted = np.array([int(a) for a in test_score_df['anomaly']])

        return actual, predicted


    # AUROC 구하기

    # threshold = 1,2,3....100으로 해주기
    final_loss = [(loss / timesteps) for loss in loss_list]
    final_loss = pd.array(final_loss)
    min_value = int(np.min(final_loss))
    max_value = int(np.max(final_loss))


    #[loss / timesteps for loss in loss_list]
    #threshold_list = []
    #for i in range(min_value, max_value, (max_value-min_value)/100):
    #    threshold_list.append(i)

    import decimal

    def drange(x,y,jump):
        while x <y:
            yield float(x)
            x += decimal.Decimal(jump)

    threshold_list = list(drange(min_value, max_value, (max_value-min_value)/100))

    final_actual11 = []
    final_predicted11 = []

    TPR = []
    FPR = []

    for i in range(len(threshold_list)):
        ac, pr = roc(error_df.Reconstruction_error, threshold_list[i])
        final_actual11.append(ac)
        final_predicted11.append(pr)

        TP = 0
        FP = 0
        TN = 0
        FN = 0
        # compare final_actual11[i] and final_predicted11[i]
        for j in range(len(final_actual11[i])):
            if final_actual11[i][j] == 1 and final_predicted11[i][j] == 1:
                TP += 1
            elif final_actual11[i][j] == 1 and final_predicted11[i][j] == 0:
                FN += 1
            elif final_actual11[i][j] == 0 and final_predicted11[i][j] == 1:
                FP += 1
            elif final_actual11[i][j] == 0 and final_predicted11[i][j] == 0:
                TN += 1

        TPR.append(TP / (TP + FN))
        FPR.append(FP / (FP + TN))

    # 최종 면적 구하기
    from sklearn.metrics import auc

    area = auc(FPR, TPR)

    print('area under curve:', area)


    area_list.append(area)
    
print(area_list)