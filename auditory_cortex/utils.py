import numpy as np
import cupy as cp
import torch
import torch.nn as nn
import os
import pandas as pd
from scipy import linalg
import torchaudio
import pickle
import matplotlib as mpl
import auditory_cortex.helpers as helpers


def get_2d_cmap(session, clrm1 ='YlGnBu', clrm2 = 'YlOrRd'):
    
    cmap1 = mpl.cm.get_cmap(clrm1)
    cmap2 = mpl.cm.get_cmap(clrm2)    
    # make a copy of session to coordinates...
    session_to_coordinates =  helpers.session_to_coordinates.copy()
    """"maps coordinates to 2d color map."""
    session = int(float(session))
    coordinates = session_to_coordinates[session]

    # mapping to 0-1 range
    coords_x = (coordinates[0] + 2)/4.0
    coords_y = (coordinates[1] + 2)/4.0
    c1 = cmap1(coords_x)
    c2 = cmap2(coords_y)
    # c3 = (c1[0],c2[1] ,0.5, c1[3])
    # c3 = ((c1[0] + c2[0])/2.0,(c1[1] + c2[1])/2.0 ,0.5, c1[3])
    # c3 = (c1[0],c2[1] ,0.0, c1[3])
    c3 = (c1[0],c2[1] ,0.0, c1[3])
    # c3 = cmap_2d(c1, c2)
    return c3

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
    # using Y_hat 'prediction from linear regression' with MSE Loss     
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

def write_to_disk(corr_dict, file_path, normalizer=None):
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
        data = pd.DataFrame(columns=['session','layer','channel','bin_width',
                                    'delay','train_cc_raw','test_cc_raw', 'normalizer', 'N_sents'])
    session = corr_dict['session']
    model_name = corr_dict['model']
    win = corr_dict['win']
    delay = corr_dict['delay']
    ch = np.arange(corr_dict['test_cc_raw'].shape[1])
    layers = np.arange(corr_dict['test_cc_raw'].shape[0])
    N_sents = corr_dict['N_sents']
    if normalizer is None:
        normalizer = np.zeros_like(ch)
    for layer in layers:
        df = pd.DataFrame(np.array([np.ones_like(ch)*int(session),
                                    np.ones_like(ch)*layer,
                                    ch, 
                                    np.ones_like(ch)*win, 
                                    np.ones_like(ch)*delay,
                                    corr_dict['train_cc_raw'][layer,:],
                                    corr_dict['test_cc_raw'][layer,:],
                                    normalizer,
                                    np.ones_like(ch)*N_sents
                                    ]).transpose(),
                        columns=data.columns
                        )
        data = pd.concat([data,df], axis=0, ignore_index=True)
    data.to_csv(file_path, index=False)
    print(f"Data saved for model: '{model_name}', session: '{session}',\
    bin-width: {win}ms, delay: {delay}ms at file: '{file_path}'")
    return data

def cc_norm(y, y_hat, sp=1, normalize=False):
    """
    Args:   
        y_hat (ndarray): (n_samples, channels) or (n_samples, channels, repeats) for null dist
        y (ndarray): (n_samples, channels)  
        sp & normalize are redundant...! 
    """
    #check if incoming array is np or cp,
    #and decide which module to use...!
    if type(y).__module__ == np.__name__:
        module = np
    else:
        module = cp
    # if 'normalize' = True, use signal power as factor otherwise use normalize CC formula i.e. 'un-normalized'
    try:
        n_channels = y.shape[1]
    except:
        n_channels=1
        y = module.expand_dims(y,axis=1)
        y_hat = module.expand_dims(y_hat,axis=1)
        
    corr_coeff = module.zeros(y_hat.shape[1:])
    for ch in range(n_channels):
        corr_coeff[ch] = cc_single_channel(y[:,ch],y_hat[:,ch])
    return cp.asnumpy(corr_coeff)

# def cc_norm_cp(y, y_hat, sp=1, normalize=False):
#     """
#     Args:   
#         y_hat (ndarray): (n_samples, channels) or (n_samples, channels, repeats) for null dist
#         y (ndarray): (n_samples, channels)  
#         sp & normalize are redundant...! 
#     """
#     # if 'normalize' = True, use signal power as factor otherwise use normalize CC formula i.e. 'un-normalized'
#     try:
#         n_channels = y.shape[1]
#     except:
#         n_channels=1
#         y = cp.expand_dims(y,axis=1)
#         y_hat = cp.expand_dims(y_hat,axis=1)
        
#     corr_coeff = cp.zeros(y_hat.shape[1:])
#     for ch in range(n_channels):
#         corr_coeff[ch] = cc_single_channel_cp(y[:,ch],y_hat[:,ch])
#     return corr_coeff

# def cc_single_channel_cp(y, y_hat):
#     """
#     computes correlations for the given spikes and predictions 'single channel'

#     Args:   
#         y_hat (ndarray): (n_sampes,) or (n_samples,repeats) spike predictions
#         y (ndarray): (n_samples) actual spikes for single channel 

#     Returns:  
#         ndarray: (1,) or (repeats, ) correlation value or array (for repeats). 
#     """
#     try:
#         y_hat = cp.transpose(y_hat,(1,0))
#     except:
#         y_hat = cp.expand_dims(y_hat, axis=0)
#     return cp.cov(y, y_hat)[0,1:] / (cp.sqrt(cp.var(y)*cp.var(y_hat, axis=1)) + 1.0e-8)

def cc_single_channel(y, y_hat):
    """
    computes correlations for the given spikes and predictions 'single channel'

    Args:   
        y_hat (ndarray): (n_sampes,) or (n_samples,repeats) spike predictions
        y (ndarray): (n_samples) actual spikes for single channel 

    Returns:  
        ndarray: (1,) or (repeats, ) correlation value or array (for repeats). 
    """
    #check if incoming array is np or cp,
    #and decide which module to use...!
    if type(y).__module__ == np.__name__:
        module = np
    else:
        module = cp
    try:
        y_hat = module.transpose(y_hat,(1,0))
    except:
        y_hat = module.expand_dims(y_hat, axis=0)
    return module.cov(y, y_hat)[0,1:] / (module.sqrt(module.var(y)*module.var(y_hat, axis=1)) + 1.0e-8)

# def regression_param(X, y):
#     """
#     Computes the least-square solution to the equation Xz = y,
  
#     Args:
#         X (ndarray): (M,N) left-hand side array
#         y (adarray): (M,) or (M,K) right-hand side array
#     Returns:
#         ndarray: (N,) or (N,K)
#     """
#     B = linalg.lstsq(X, y)[0]
#     return B
def reg(X,y, lmbda=0):

    #check if incoming array is np or cp,
    #and decide which module to use...!
    if type(X).__module__ == np.__name__:
        module = np
    else:
        module = cp
    
    if X.ndim ==2:
        X = module.expand_dims(X,axis=0)
    d = X.shape[2]
    m = X.shape[1]
    I = module.eye(d)
    X_T = X.transpose((0,2,1))
    a = module.matmul(X_T, X) + m*lmbda*I
    b = module.matmul(X_T, y)
    B = module.linalg.solve(a,b)
    return B.squeeze()

# def reg_cp(X,y, lmbda=0):
#     # takes in cupy arrays and uses gpu...!
#     if X.ndim ==2:
#         X = cp.expand_dims(X,axis=0)
#     d = X.shape[2]
#     m = X.shape[1]
#     I = cp.eye(d)
#     X_T = X.transpose((0,2,1))
#     a = cp.matmul(X_T, X) + m*lmbda*I
#     b = cp.matmul(X_T, y)
#     B = cp.linalg.solve(a,b)
#     return B.squeeze()

# def regression_param(X, y):
#     B = linalg.lstsq(X, y)[0]
#     return B

def predict(X, B):
    """
    Args:
        X (ndarray): (M,N) left-hand side array
        B (ndarray): (N,) or (N,K)
    Returns:
        ndarray: (M,) or (M,K)
    """

    #check if incoming array is np or cp,
    #and decide which module to use...!
    if type(X).__module__ == np.__name__:
        module = np
    else:
        module = cp
    pred = module.matmul(X,B)
    if pred.ndim ==3:
        return pred.transpose(1,2,0) 
    return pred 

# def predict_cp(X, B):
#     """
#     Args:
#         X (ndarray): (M,N) left-hand side array
#         B (ndarray): (N,) or (N,K)
#     Returns:
#         ndarray: (M,) or (M,K)
#     """
#     # pred = X@B
#     # cp.matmul is supposed to be faster...!
#     pred = cp.matmul(X,B)
#     if pred.ndim ==3:
#         return pred.transpose(1,2,0) 
#     return pred 

def fit_and_score(X, y):
    # x_train, y_train, x_test, y_test = train_test_split(X,y, split=0.7)
    # B = reg(x_train,y_train)
    # y_hat = predict(x_test,B)
    # cc = cc_norm(y_hat, y_test)

    B = reg(X,y)
    y_hat = predict(X,B)
    cc = cc_norm(y_hat, y)
    return cc

# def fit_and_score(X, y):
#     B = regression_param(X,y)
#     y_hat = predict(X,B)
#     cc = cc_norm(y_hat, y)
#     return cc

def train_test_split(x,y, split=0.7):
    split = int(x.shape[0]*split)
    return x[0:split], y[0:split], x[split:], y[split:]


def mse_loss(y, y_hat):
    #check if incoming array is np or cp,
    #and decide which module to use...!
    if type(y).__module__ == np.__name__:
        module = np
    else:
        module = cp
    if y.ndim < y_hat.ndim:
        y = module.expand_dims(y, axis=-1)
    return (module.sum((y - y_hat)**2, axis=0))/y_hat.shape[0]

# def mse_loss_cp(y, y_hat):
#     if y.ndim < y_hat.ndim:
#         y = cp.expand_dims(y, axis=-1)
#     return (cp.sum((y - y_hat)**2, axis=0))/y_hat.shape[0]

def inter_trial_corr(spikes, n=1000):
    """Compute distribution of inter-trials correlations.

    Args: 
        spikes (ndarray): (repeats, samples/time, channels)

    Returns:
        trials_corr (ndarray): (n, channels) distribution of inter-trial correlations
    """
    trials_corr = np.zeros((n, spikes.shape[2]))
    for t in range(n):
        trials = np.random.choice(np.arange(0,spikes.shape[0]), size=2, replace=False)
        trials_corr[t] = cc_norm(spikes[trials[0]].squeeze(), spikes[trials[1]].squeeze())

    return trials_corr

def normalize(x):
    # Normalize for spectrogram
    mean = x.mean(axis=0)
    square_sums = (x ** 2).sum(axis=0)
    x = np.subtract(x, mean)
    var = square_sums / x.shape[0] - mean ** 2
    std = np.sqrt(np.maximum(var, 1e-10))
    x = np.divide(x, std)

    return x

def spectrogram(aud):
    waveform = torch.tensor(aud, dtype=torch.float32).unsqueeze(dim=0)
    waveform = waveform * (2 ** 15)
    kaldi = torchaudio.compliance.kaldi.fbank(waveform, num_mel_bins=80, window_type='hanning')
    kaldi = normalize(kaldi)

    return kaldi.transpose(0,1)

def write_optimal_delays(filename, result):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            prev_result = pickle.load(f)
        result['corr'] = np.concatenate([prev_result['corr'], result['corr']], axis=0)
        result['loss'] = np.concatenate([prev_result['loss'], result['loss']], axis=0)
        result['delays'] = np.concatenate([prev_result['delays'], result['delays']], axis=0)
        # temporary change...should be removed after run..!
        # result['delays'] = np.concatenate([np.arange(0,201,10), result['delays']], axis=0)
        

    with open(filename, 'wb') as file:
        pickle.dump(result, file)
