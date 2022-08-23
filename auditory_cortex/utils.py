import numpy as np
import torch
import torch.nn as nn
import os
import pandas as pd
from scipy import linalg

def down_sample(data, k):
    #down samples 'data' by factor 'k' along dim=0 
    n_dim = data.ndim
    if n_dim == 1:
        out = np.zeros(int(np.ceil(data.shape[0]/k)))
    elif n_dim ==2:
        out = np.zeros((int(np.ceil(data.shape[0]/k)), data.shape[1]))
    for i in range(out.shape[0]):
      #Just add the remaining samples at the end...!
      if (i == out.shape[0] -1):
        out[i] = data[k*i:].sum(axis=0)
      else:  
        out[i] = data[k*i:k*(i+1)].sum(axis=0)
    return out

@torch.no_grad()
def poisson_regression_score(model, X, Y):
    # Poisson Prediction with Poisson Score....!
    eta = model(X)
    Y_hat = np.exp(eta)
    # eta = np.log(Y_hat)
    Y_mean = Y.mean()
    score = (Y_hat - Y*np.log(Y_hat)).mean() - (Y_mean - Y*np.log(Y_mean)).mean()
    
    return score.item()

def gaussian_cross_entropy(Y, Y_hat):
    # Gaussian Predictions with Gaussain Loss
    #loss_fn = nn.GaussianNLLLoss(full=True)
    sq_error = (Y - Y_hat)**2
    var = sq_error.sum(axis=0)
    cross_entropy = 0.5*(np.log(2*np.pi) + np.log(var) + 1)
    #cross_entropy = loss_fn(Y.squeeze(), Y_hat.squeeze(), var.squeeze()).item()
    
    return cross_entropy

def poisson_cross_entropy(Y, Y_hat):
    # Poisson predictions with Poisson Loss
    loss_fn = nn.PoissonNLLLoss(log_input=False, full=True)
    cross_entropy = loss_fn(Y, Y_hat).item()
    
    return cross_entropy

def linear_regression_score(Y, Y_hat):
    # using Y_hat 'prediction from linear regression' with Poisson Loss 
    Y_hat[Y_hat <= 0] = 1.0e-5
    Y_mean = Y.mean()
    score = (Y_hat - Y*np.log(Y_hat)).mean() - (Y_mean - Y*np.log(Y_mean)).mean()
    
    return score

def MSE_poisson_predictions(Y, poisson_pred):
    # using Y_hat 'prediction from linear regression' with Poisson Loss 
    Y_hat = np.exp(poisson_pred)
    loss_fn = nn.MSELoss()
    score = loss_fn(Y, Y_hat)
    
    return score

def MSE_Linear_predictions(Y, linear_reg_pred):
    # using Y_hat 'prediction from linear regression' with Poisson Loss     
    loss_fn = nn.MSELoss()
    score = loss_fn(Y, linear_reg_pred)
    
    return score

def poiss_regression(x_train, y_train):
    X = torch.tensor(x_train, dtype=torch.float32)
    Y = torch.tensor(y_train, dtype=torch.float32)
    N,d = X.shape
    model = nn.Linear(d, 1, bias=True)
    loss_fn = nn.PoissonNLLLoss(log_input = True, full=True)
    optimizer = torch.optim.Adam(model.parameters())
    
    state = model.state_dict()
    state['bias'] = torch.zeros(1)
    state['weight'] = torch.zeros((1,d))
    model.load_state_dict(state)
    num_epochs = 2000
    criteria = 1e-4
    loss_history = []
    P_scores = []
    
    count = 0
    for i in range(num_epochs):
        Y_hat = model(X)
        loss = loss_fn(Y_hat, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i >= 50 and (np.linalg.norm(loss_history[-1] - loss.item()) < criteria):
            print(loss_history[-1] - loss.item())
            count += 1
        loss_history.append(loss.item())
        P_scores.append(poisson_regression_score(model, X, Y))
        if count >= 3:
            break
      
    return model, loss_history, P_scores
def write_df_to_disk(df, file_path):
    """
    Takes in any pandas dataframe 'df' and writes as csv file 'file_path',
    appends data to existing file, if it already exists.

    Args:
        df (dataframe): dataframe containing data to write
        file_path (str): name of the file to write to.
    """
    if os.path.isfile(file_path):
        data = pd.read_csv(file_path)
        action = 'appended'
    else:
        data = pd.DataFrame(columns= df.columns)
        action = 'written'
    data = pd.concat([data,df], axis=0, ignore_index=True)
    data.to_csv(file_path, index=False)
    print(f"Dataframe {action} to {file_path}.")

def write_to_disk(corr, file_path):
    """
    | Takes in the 'corr' dict and stores the results
    | at the 'file_path', (concatenates if file already exists)
    | corr: dict of correlation scores
    | win: float
    | delay: float
    | file_path: path of csv file
    """
    if os.path.isfile(file_path):
        data = pd.read_csv(file_path)
    else:
        data = pd.DataFrame(columns=['layer','channel','bin_width','delay','train_cc', 'val_cc','test_cc'])
    win = corr['win']
    delay = corr['delay']
    ch = np.arange(corr['train'].shape[1])
    layers = np.arange(corr['train'].shape[0])
    for layer in layers:
        df = pd.DataFrame(np.array([np.ones_like(ch)*layer,ch, 
                        np.ones_like(ch)*win, 
                        np.ones_like(ch)*delay,
                        corr['train'][layer,:],
                        corr['val'][layer,:],
                        corr['test'][layer,:]]).transpose(),
                        columns=['layer','channel','bin_width','delay','train_cc', 'val_cc','test_cc']
                        )
        data = pd.concat([data,df], axis=0, ignore_index=True)
    data.to_csv(file_path, index=False)
    print(f"Data saved for bin-width: {win}, delay: {delay} at file: '{file_path}'")
    return data

def cc_norm(y_hat, y, sp=1, normalize=False):
    """
    Args:   
        y_hat (ndarray): (n_samples, channels) or (n_samples, channels, repeats) for null dist
        y (ndarray): (n_samples, channels)  
        sp & normalize are redundant...! 
    """
    # if 'normalize' = True, use signal power as factor otherwise use normalize CC formula i.e. 'un-normalized'
    try:
        n_channels = y.shape[1]
    except:
        n_channels=1
        y = np.expand_dims(y,axis=1)
        y_hat = np.expand_dims(y_hat,axis=1)
        
    corr_coeff = np.zeros(y_hat.shape[1:])
    for ch in range(n_channels):
        corr_coeff[ch] = cc_single_channel(y_hat[:,ch],y[:,ch])
    return corr_coeff

def cc_single_channel(y_hat, y):
    """
    computes correlations for the given spikes and predictions 'single channel'

    Args:   
        y_hat (ndarray): (n_sampes,) or (n_samples,repeats) spike predictions
        y (ndarray): (n_samples) actual spikes for single channel 

    Returns:  
        ndarray: (1,) or (repeats, ) correlation value or array (for repeats). 
    """
    try:
        y_hat = np.transpose(y_hat,(1,0))
    except:
        y_hat = np.expand_dims(y_hat, axis=0)
    return np.cov(y_hat, y)[0,1:] / np.sqrt(np.var(y)*np.var(y_hat, axis=1))


def regression_param(X, y):
    """
    Computes the least-square solution to the equation Xz = y,
  
    Args:
        X (ndarray): (M,N) left-hand side array
        y (adarray): (M,) or (M,K) right-hand side array
    Returns:
        ndarray: (N,) or (N,K)
    """
    B = linalg.lstsq(X, y)[0]
    return B


# def regression_param(X, y):
#     B = linalg.lstsq(X, y)[0]
#     return B

def predict(X, B):
    """
    Args:
        X (ndarray): X (ndarray): (M,N) left-hand side array
        B (ndarray): (N,) or (N,K)
    Returns:
        ndarray: (M,) or (M,K)
    """
    return X@B

def fit_and_score(X, y):
    x_train, y_train, x_test, y_test = train_test_split(X,y, split=0.7)
    B = regression_param(x_train,y_train)
    y_hat = predict(x_test,B)
    cc = cc_norm(y_hat, y_test)
    return cc

# def fit_and_score(X, y):
#     B = regression_param(X,y)
#     y_hat = predict(X,B)
#     cc = cc_norm(y_hat, y)
#     return cc

def train_test_split(x,y, split=0.7):
    split = int(x.shape[0]*split)
    return x[0:split], y[0:split], x[split:], y[split:]

