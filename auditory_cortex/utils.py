import numpy as np
import torch
import torch.nn as nn
import os
import pandas as pd

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

def regression_param(X, y):
    B = linalg.lstsq(X, y)[0]
    return B

def predict(X, B):
    return X@B

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