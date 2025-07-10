import os
import yaml
import pickle
import torch
import torchaudio
import torch.nn as nn
import numpy as np
import cupy as cp
import pandas as pd
import matplotlib.pylab as plt

# local
# from auditory_cortex import session_to_coordinates#, #CMAP_2D
from auditory_cortex import aux_dir, saved_corr_dir

import sys
import logging
logger = logging.getLogger(__name__)

def set_up_logging(level=None):
    # Set up logging configuration

    fmt = '%(levelname)s:%(message)s'
    if level is None or level == 'info':
        level = logging.INFO 
    elif level == 'debug':
        level = logging.DEBUG
        fmt = '%(levelname)s:%(name)s:%(message)s'
    elif level == 'warning':
        level = logging.WARNING
    logging.basicConfig(
        level=level,    
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),  
        ]
    )
    # Suppress DEBUG logs
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("fsspec").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

def load_dnn_config(model_name=None, filename=None):
    """Reads the configuration file for the model."""
    assert model_name is not None or filename is not None, \
        f"Either model_name or filename must be provided."
    if filename is None:
        filename = model_name + '_config.yml'
    config_file = os.path.join(aux_dir, filename)
    with open(config_file, 'r') as f:
        config = yaml.load(f, yaml.FullLoader)
    return config


class SyntheticInputUtils:
    """Contains utility functions for analysis of Synthetic inputs.
    """
    @staticmethod
    def normalize(x):
        """
        ONLY USED FOR VISUALIZING THE SPECTROGRAM...!
        Normalizes the spectrogram (obtained using kaldi transform),
        done to match the spectrogram exactly to the Speec2Text transform
        """
        mean = x.mean(axis=0)
        square_sums = (x ** 2).sum(axis=0)
        x = np.subtract(x, mean)
        var = square_sums / x.shape[0] - mean ** 2
        std = np.sqrt(np.maximum(var, 1e-10))
        x = np.divide(x, std)

        return x
    
    @classmethod
    def get_spectrogram(cls, waveform):
        """Returns spectrogram for the input waveform (ndarray or tensor)"""
        if not torch.is_tensor(waveform):
            waveform = torch.tensor(waveform)
        waveform = torch.atleast_2d(waveform)
            # waveform = waveform.unsqueeze(dim=0)
        waveform = waveform * (2 ** 15)
        spect = torchaudio.compliance.kaldi.fbank(waveform, num_mel_bins=80, window_type='hanning')
        spect = cls.normalize(spect)
        return spect


    @classmethod
    def plot_spect(cls, waveform, cmap='viridis', ax=None):
        """Takes in a waveform (as ndarray or tensor) and plots its 
        spectrogram
        """
        waveform = waveform.squeeze()
        if ax is None:
            fig, ax = plt.subplots()
        if not torch.is_tensor(waveform):
            waveform = torch.tensor(waveform)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(dim=0)
            waveform = cls.get_spectrogram(waveform)
        # waveform = waveform * (2 ** 15)
        # kaldi = torchaudio.compliance.kaldi.fbank(waveform, num_mel_bins=80, window_type='hanning')
        # kaldi = cls.normalize(kaldi)
        x_ticks = np.arange(0, waveform.shape[0], 20)
        data = ax.imshow(waveform.transpose(1,0), cmap=cmap, origin='lower')
        ax.set_xticks(x_ticks, 10*x_ticks)
        ax.set_xlabel('time (ms)')
        ax.set_ylabel('mel filters')
        return data, ax
    
    @classmethod
    def align_add_2_signals(cls, sig1, sig2):
        """Smartly adds (averages) two signals having slight offset,
        by using the cross-correlation to align them before adding. 
        This might have to trim the extra lengths on one side of either 
        signal (depending upon the offset for perfect alignment)."""
        sig1 = sig1.squeeze()
        sig2 = sig2.squeeze()
        cross_corr = np.correlate(sig1, sig2, mode='same')
        # identify the shift for peak of cross-correlation
        if sig1.shape[0] > sig2.shape[0]:
            peak_id = np.argmax(cross_corr) - sig2.shape[0]/2
            sig1 = sig1[:sig2.shape[0]]
        elif sig2.shape[0] > sig1.shape[0]:
            peak_id = np.argmax(cross_corr) - sig1.shape[0]/2
            sig2 = sig2[:sig1.shape[0]]
        else:
            cross_corr = np.correlate(sig1, sig2, mode='same')
            peak_id = np.argmax(cross_corr) - sig1.shape[0]/2
        peak_id = int(peak_id)
        # align and average....
        if peak_id > 0:
            sig1_al = sig1[peak_id:]
            sig2_al = sig2[:-1*peak_id]
        elif peak_id < 0:
            sig1_al = sig1[:peak_id]
            sig2_al = sig2[-1*peak_id:]
        else:
            sig1_al = sig1
            sig2_al = sig2
        
        out_sig = (sig1_al + sig2_al)/2

        fig, ax = plt.subplots(ncols=2)
        cls.plot_spect(sig1_al, cmap='jet', ax=ax[0])
        cls.plot_spect(sig2_al, cmap='jet', ax=ax[1])

        # return the correlation of aligned signals...
        corr = np.corrcoef(sig1_al, sig2_al)[0,1]
                
        return out_sig, corr


    @classmethod
    def align_add_signals(cls, signals_list):
        """Takes in a list of signals, and aligns and combines them
        pairwise, keeps doing it until only one signal is left.
        In short, it align and add first 2, 4 , 8 or any highest 
        possible power of 2.
        For example, given 5 signals, it will only use first 4."""
        while len(signals_list) > 1:
            new_list = []
            m = int(len(signals_list)/2)
            for i in range(m):
                sig1 = signals_list[2*i]
                sig2 = signals_list[2*i + 1]
                new_list.append(cls.align_add_2_signals(sig1, sig2)[0])
            signals_list = new_list
        
        return signals_list[0]

    @staticmethod
    def cross_correlation(sig1, sig2):
        """Computes cross-correlation both for 1D and 2D signals, 
        for 2D case shifts across axis=0 only (taken as time axis).
        Results in cross correlation similar to 'scipy.signal.correlate'
        when used with mode='same', and for boundry conditions wraps the 
        shifted signal, so we may call it as circular cross correlations. 
        
        Args:
            sig1 (ndarray): having dimensions (t,) 1D case or (t, f) in 2D case
            sig2 (ndarray): having dimensions (t,) 1D case or (t, f) in 2D case
        
        Returns:
            cross_corr (list): cross-correlation at different shifts with zero-shift
                            at the middle.
        """
        logger.info(f"Computing cross-correlation using local implemenation....")
        # making sure both the signals have equal lengths.
        cross_corr = []
        if sig1.shape[0] > sig2.shape[0]:
            sig1 = sig1[:sig2.shape[0]]
        elif sig1.shape[0] < sig2.shape[0]:
            sig2 = sig2[:sig1.shape[0]]

        # computing 2d cross-correlation (shift only along time axis (axis=0))
        sig_length = sig1.shape[0]
        max_neg_shift = -1*int(sig_length/2)
        max_pos_shift = int(sig_length/2 + 0.5) # we want to round-up.
        for shift in np.arange(max_neg_shift, max_pos_shift, 1):
            sig2_shifted = np.concatenate([sig2[-shift:],sig2[:-shift]], axis=0)
            cross_corr.append(np.sum(sig1*sig2_shifted))

        return cross_corr
    





# class CorrelationUtils:
#     """Contains utility functions for correlations analysis.
#     """
#     def merge_correlation_results(model_name, file_identifiers, idx):
#         """
#         Args:

#             model_name: Name of the pre-trained network
#             file_identifiers: List of filename identifiers 
#             idx:    id of the file identifier to use for saving the merged results
#         """
#         # results_dir = '/depot/jgmakin/data/auditory_cortex/correlation_results/cross_validated_correlations'

#         corr_dfs = []
#         for identifier in file_identifiers:
#             filename = f"{model_name}_{identifier}_corr_results.csv"
#             file_path = os.path.join(saved_corr_dir, filename)

#             corr_dfs.append(pd.read_csv(file_path))

#             # remove the file
#             os.remove(file_path)

#         # save the merged results at the very first filename...
#         output_identifer = file_identifiers[idx]    
#         filename = f"{model_name}_{output_identifer}_corr_results.csv"
#         file_path = os.path.join(saved_corr_dir, filename)

#         data = pd.concat(corr_dfs)
#         data.to_csv(file_path, index=False)
#         logger.info(f"Output saved at: \n {file_path}")

    # @staticmethod
    # def add_layer_types(model_name, results_identifer):

    #     # reading layer_types from aux config...
    #     layer_types = {}
    #     config_file = os.path.join(aux_dir, f"{model_name}_config.yml")
    #     with open(config_file, 'r') as f:
    #         config = yaml.load(f, yaml.FullLoader)

    #     # config['layers']
    #     for layer_config in config['layers']:
    #         layer_types[layer_config['layer_id']] = layer_config['layer_type']

    #     # reading results directory...
    #     if results_identifer != '':
    #         model = f'{model_name}_{results_identifer}'
    #     else:
    #         model = model_name 
    #     filename = f"{model}_corr_results.csv"
    #     file_path = os.path.join(saved_corr_dir, filename)
    #     data = pd.read_csv(file_path)
    #     logger.info(f"reading from {file_path}")

    #     # remove 'Unnamed' columns
    #     data = data.loc[:, ~data.columns.str.contains('Unnamed')]

    #     # add 'layer_type' as a column
    #     for layer, type in layer_types.items():
    #         ids = data[data['layer']==layer].index
    #         data.loc[ids, 'layer_type'] = type

    #     logger.info("Writing back...!")
    #     data.to_csv(file_path, index=False)

    # @staticmethod
    # def copy_normalizer(model_name, results_identifer=''):
    #     # reading results directory...
    #     if results_identifer != '':
    #         corr_file = f'{model_name}_{results_identifer}'
    #     else:
    #         corr_file = model_name

    #     filename = f'{corr_file}_corr_results.csv'
    #     corr_file_path = os.path.join(saved_corr_dir, filename)
    #     data1 = pd.read_csv(corr_file_path)
    #     logger.info(f"Reading file from: \n {corr_file_path}")
    #     # normalizer
    #     normalizer_file = 'wav2letter_modified_normalizer2_corr_results.csv'
    #     norm_file_path = os.path.join(saved_corr_dir, normalizer_file)
    #     data2 = pd.read_csv(norm_file_path)
    #     logger.info(f"Reading normalizers from: \n {norm_file_path}")

    #     sessions = data1['session'].unique()
    #     for session in sessions:
    #         select_data = data1[data1['session']==session]
    #         channels = select_data['channel'].unique()
    #         for ch in channels:
    #             ids = select_data[select_data['channel'] == ch].index

    #             norm = data2[(data2['session']==session) &(data2['channel']==ch)]['normalizer'].head(1).item() 

    #             data1.loc[ids, 'normalizer'] = norm
        
    #     data1.to_csv(corr_file_path, index=False)
    #     logger.info(f"Normalizer updated and written back to file: \n {corr_file_path}")



def _get_layer_receptive_field(kernels, strides, layer_id):
    """Computes receptive field for the layer at index 'layer_id'.

    Args:
        kernels: list = kernels sizes of convolution layers in order.
        strides: list = strides of convolution layers in order.
        layer_id: int = index of layer to compute the receptive field for.

    Returns:
        Receptive field (number of samples of input).
    """
    samples = kernels[layer_id]
    for i in range(layer_id,0,-1):
        samples = (samples - 1)*strides[i-1] + kernels[i-1]

    return samples

def get_receptive_fields(kernels, strides, fs=16000):
    """Computes receptive fields for all the layers of the network,
    the given arrays fo kernels and strides.
    Args:
        kernels: list = kernels sizes of convolution layers in order.
        strides: list = strides of convolution layers in order.

    """
    logger.info("Calculating receptive fields for all layers...")
    samping_rates = np.zeros(len(kernels))
    samping_rates[0] = fs/strides[0]
    for i in range(0,len(strides)):
        rf_samples = _get_layer_receptive_field(kernels, strides, i)
        rf_ms = rf_samples*1000/fs
        if i>0:
            samping_rates[i] = samping_rates[i-1]/strides[i] 
        logger.info(f"Layer {i}, RF: {rf_samples:5d} samples, {rf_ms:4.2f} ms," +
              f" sampling_rate: {samping_rates[i]:.0f}Hz, sampling_time: {(1000/samping_rates[i]):.3f}ms",)


def coordinates_to_color(cmap_2d, coordinates):
    return cmap_2d(coordinates[0], coordinates[1])/255.0


# def down_sample(data, k):
# 	#down samples 'data' by factor 'k' along dim=0 
# 	n_dim = data.ndim
# 	if n_dim == 1:
# 		out = np.zeros(int(np.ceil(data.shape[0]/k)))
# 	elif n_dim ==2:
# 		out = np.zeros((int(np.ceil(data.shape[0]/k)), data.shape[1]))
# 	for i in range(out.shape[0]):
# 		  #Just add the remaining samples at the end...!
# 		if (i == out.shape[0] -1):
# 			out[i] = data[k*i:].sum(axis=0)
# 		else:  
# 			out[i] = data[k*i:k*(i+1)].sum(axis=0)
# 	return out

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

# def poisson_cross_entropy(Y, Y_hat):
#     # Poisson predictions with Poisson Loss
#     loss_fn = nn.PoissonNLLLoss(log_input=False, full=True)
#     cross_entropy = loss_fn(Y, Y_hat).item()
    
#     return cross_entropy

def poisson_cross_entropy(ref, predictions):
    """Computes the poisson cross entropy on the predicted outputs,
    that can come from multiple model candidates (layers). 

    Args:
        ref: ndarray or cupy array: (samples, response_channels)
        predictions: ndarray or cupy array: (samples, respose_channels, layers)
    
    Returns:
        loss: ndarray = (channels, layers)
    """
    poisson_loss = torch.nn.PoissonNLLLoss(log_input=True, full=True, reduction='none')

    with torch.no_grad():
        ref_tensor = torch.tensor(ref)
        predictions_tensor = torch.tensor(predictions)
        ref_tensor = ref_tensor.transpose(0,1).unsqueeze(dim=1)
        predictions_tensor = predictions_tensor.transpose(0,1).transpose(1,2)

        loss = poisson_loss(predictions_tensor, ref_tensor)
        loss = torch.mean(loss, dim=2).cpu().numpy()
    return loss

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
            logger.info(loss_history[-1] - loss.item())
            count += 1
        loss_history.append(loss.item())
        P_scores.append(poisson_regression_score(model, X, Y))
        if count >= 3:
            break
      
    return model, loss_history, P_scores

# def write_df_to_disk(df, file_path):
#     """
#     Takes in any pandas dataframe 'df' and writes as csv file 'file_path',
#     appends data to existing file, if it already exists.

#     Args:
#         df (dataframe): dataframe containing data to write
#         file_path (str): name of the file to write to.
#     """
#     if os.path.isfile(file_path):
#         data = pd.read_csv(file_path)
#         action = 'appended'
#     else:
#         data = pd.DataFrame(columns= df.columns)
#         action = 'written'
#     data = pd.concat([data,df], axis=0, ignore_index=True)
#     data.to_csv(file_path, index=False)
#     logger.info(f"Dataframe {action} to {file_path}.")

# def write_STRF(corr_dict, file_path, normalizer=None):

#     columns_list = [
#         'session','channel','bin_width', 'delay',
#         'num_freqs', 'tmin', 'tmax', 'lmbda',
#         'test_cc_raw', 'normalizer',
#         'mVocs_test_cc_raw', 'mVocs_normalizer',
#         ]
#     if os.path.isfile(file_path):
#         logger.info(f"Reading existing result from: {file_path}")
#         data = pd.read_csv(file_path)
#         # making sure there is no extra columns in the existing dataframe.
#         for column in data.columns:
#             if column not in columns_list:
#                 data.drop(columns=column, inplace=True)
#     else:
#         data = pd.DataFrame(columns=columns_list)
#     session = corr_dict['session']
#     win = corr_dict['win']
#     delay = corr_dict['delay']
#     ch = np.arange(corr_dict['strf_corr'].shape[0])
#     num_freqs = corr_dict['num_freqs']
#     tmin = corr_dict['tmin']
#     tmax = corr_dict['tmax']
#     lmbda = corr_dict['lmbda']
#     if normalizer is None:
#         normalizer = np.zeros_like(ch)

#     df = pd.DataFrame(np.array([np.ones_like(ch)*int(session),
#                                     ch, 
#                                     np.ones_like(ch)*win, 
#                                     np.ones_like(ch)*delay,
#                                     np.ones_like(ch)*num_freqs,
#                                     np.ones_like(ch)*tmin,
#                                     np.ones_like(ch)*tmax,
#                                     np.ones_like(ch)*lmbda,
#                                     corr_dict['strf_corr'],
#                                     normalizer,
#                                     corr_dict['mVocs_strf_corr'],
#                                     normalizer,
#                                     ]).transpose(),
#                         columns=data.columns
#                         )
#     data = pd.concat([data,df], axis=0, ignore_index=True)
#     data.to_csv(file_path, index=False)
#     logger.info(f"Data saved for session: '{session}',\
#     bin-width: {win}ms, delay: {delay}ms at file: '{file_path}'")
#     return data

def write_to_disk(corr_dict, file_path):
    """Takes in the 'corr' dict and stores the results
    at the 'file_path', (concatenates if file already exists)
    
    Args:
        corr_dict (dict): 
        file_path: path of csv file to write results to.
    """
    columns= list(corr_dict.keys())
    df = pd.DataFrame(corr_dict)
    if os.path.isfile(file_path):
        data = pd.read_csv(file_path)[columns]
        data = pd.concat([data,df], axis=0, ignore_index=True)
    else:
        data = df
    data.to_csv(file_path, index=False)
    logger.info(f"Data saved to: '{file_path}'")
    return data


# def write_to_disk(corr_dict, file_path, normalizer=None):
#     """
#     | Takes in the 'corr' dict and stores the results
#     | at the 'file_path', (concatenates if file already exists)
#     | corr: dict of correlation scores
#     | win: float
#     | delay: float
#     | file_path: path of csv file
#     """
#     columns=[
#             'session','layer','channel','bin_width',
#             'delay', 'test_cc_raw', 'normalizer', 
#             'mVocs_test_cc_raw', 'mVocs_normalizer',
#             'opt_lag', 'opt_lmbda', 'N_sents', 

#             # Deprecated
#             # 'poiss_entropy', 'uncertainty_per_spike', 'bits_per_spike_NLB',
#             ]


#     if os.path.isfile(file_path):
#         data = pd.read_csv(file_path)[columns]
#     else:
#         data = pd.DataFrame(columns=columns)
#     session = corr_dict['session']
#     model_name = corr_dict['model']
#     win = corr_dict['win']
#     delay = corr_dict['delay']
#     ch = np.arange(corr_dict['test_cc_raw'].shape[1])
#     layers = np.arange(corr_dict['test_cc_raw'].shape[0])
#     N_sents = corr_dict['N_sents']
#     layer_ids = corr_dict['layer_ids']
#     if normalizer is None:
#         normalizer = np.zeros_like(ch)

#     for layer in layers:
#         if 'opt_lag' in corr_dict.keys():
#             opt_lag = corr_dict['opt_lag']
#             opt_lags = np.ones_like(ch)*opt_lag
#         else:
#             opt_lags = corr_dict['opt_delays'][layer]
        
#         if 'opt_lmbda' in corr_dict.keys():
#             opt_lmbda = corr_dict['opt_lmbda']
#             opt_lmbdas = np.ones_like(ch)*opt_lmbda
#         else:
#             opt_lmbdas = corr_dict['opt_lmbdas'][layer]

#         df = pd.DataFrame(np.array([np.ones_like(ch)*int(session),
#                                     np.ones_like(ch)*layer_ids[int(layer)],
#                                     ch, 
#                                     np.ones_like(ch)*win, 
#                                     np.ones_like(ch)*delay,
#                                     corr_dict['test_cc_raw'][layer,:],
#                                     normalizer,
#                                     corr_dict['mVocs_test_cc_raw'][layer,:],
#                                     normalizer,
#                                     opt_lags,
#                                     opt_lmbdas,
#                                     np.ones_like(ch)*N_sents,
#                                     # Deprecated
#                                     # corr_dict['poiss_entropy'][layer,:],
#                                     # corr_dict['uncertainty_per_spike'][layer,:],
#                                     # corr_dict['bits_per_spike_NLB'][layer,:],
#                                     ]).transpose(),
#                         columns=data.columns
#                         )
#         data = pd.concat([data,df], axis=0, ignore_index=True)
#     data.to_csv(file_path, index=False)
#     logger.info(f"Data saved for model: '{model_name}', session: '{session}',\
#     bin-width: {win}ms, delay: {delay}ms at file: '{file_path}'")
#     return data

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
        cupy_return = False
    else:
        cupy_return = True
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

    if cupy_return:
        return cp.asnumpy(corr_coeff)
    else:
        return corr_coeff
    
# def compute_avg_test_corr(y_all_trials, y_pred, test_trial=None, mVocs=False):
# 	"""Computes correlation for each trial and averages across all trials.
    
# 	Args:
# 		y_all_trials: (num_trials, num_bins)
# 		y_pred: (num_bins,)
# 		tr: int = integer in range=[0, 11], Default=None.

# 	"""
def compute_avg_test_corr(y_all_trials, y_pred, n_test_trials=None):
    """Computes correlation for each trial and averages across all trials.
    
    Args:
        y_all_trials: (num_trials, num_bins)
        y_pred: (num_bins,)
        n_test_trials: int = number of trials to be tested on.
            Choices=[0, num_repeats], If None, test on all trial    
            repeats. Default=None

    Returns:
        trial_corr: ndarray = (num_channels,) correlation values averaged across trials.
    """
    trial_corr = []
    total_trial_repeats = y_all_trials.shape[0]
    if n_test_trials is None:
        trial_ids = np.arange(total_trial_repeats)
    else:
        trial_ids = np.random.choice(total_trial_repeats, size=n_test_trials, replace=True) # with replacement for bootstrapping
    for tr in trial_ids:
        trial_corr.append(cc_norm(y_all_trials[tr], y_pred))
    trial_corr = np.stack(trial_corr, axis=0)
    trial_corr = np.mean(trial_corr, axis=0)
    return trial_corr

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
    """Fits linear regression parameters using the given data.
    Depending on the type of X and y, it uses numpy or cupy for computation.
    For linear model y = XB, it solves for B using the equation X^T X B = X^T y.
    
    Args:
        X (ndarray): (M,N) or (L,M,N) left-hand side array
        y (adarray): (M,) or (M,K) right-hand side array
        lmbda (float): regularization parameter (default=0)

    Returns:
        B (ndarray): (N,) or (N,K) or (L,N) or (L,N,K)
    """

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
    # X_t = X.transpose((0,2,1))

    return module.linalg.solve(
        module.matmul(X.transpose((0,2,1)), X) + m*lmbda*module.eye(d),
        module.matmul(X.transpose((0,2,1)), y)
        ).squeeze()
    # a = module.matmul(X_t, X) + m*lmbda*module.eye(d)
    # del X # to save memory...
    # gc.collect()
    # b = module.matmul(X_t, y)

    # # a = module.matmul(X.transpose((0,2,1)), X) + m*lmbda*module.eye(d)
    # # b = module.matmul(X.transpose((0,2,1)), y)

    # return module.linalg.solve(a,b).squeeze()
    # # return B.squeeze()

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

    B = reg(X,y)
    y_hat = predict(X,B)
    cc = cc_norm(y_hat, y)
    return cc


def train_test_split(x,y, split=0.7):
    split = int(x.shape[0]*split)
    return x[0:split], y[0:split], x[split:], y[split:]


def mse_loss(y, y_hat):
    module = np
    if y.ndim < y_hat.ndim:
        y = module.expand_dims(y, axis=-1)
    loss = (module.sum((y - y_hat)**2, axis=0))/y_hat.shape[0]
    return cp.asnumpy(loss)

# def mse_loss_cp(y, y_hat):
#     if y.ndim < y_hat.ndim:
#         y = cp.expand_dims(y, axis=-1)
#     return (cp.sum((y - y_hat)**2, axis=0))/y_hat.shape[0]


#######################################################################
######## MOVED to Regression
# ######################################
# def inter_trial_corr(spikes, n=1000):
#     """Compute distribution of inter-trials correlations.

#     Args: 
#         spikes (ndarray): (repeats, samples/time, channels)

#     Returns:
#         trials_corr (ndarray): (n, channels) distribution of inter-trial correlations
#     """
#     trials_corr = np.zeros((n, spikes.shape[2]))
#     for t in range(n):
#         trials = np.random.choice(np.arange(0,spikes.shape[0]), size=2, replace=False)
#         trials_corr[t] = cc_norm(spikes[trials[0]].squeeze(), spikes[trials[1]].squeeze())

#     return trials_corr

#############################
##############

# def normalize(x):
#     # Normalize for spectrogram
#     mean = x.mean(axis=0)
#     square_sums = (x ** 2).sum(axis=0)
#     x = np.subtract(x, mean)
#     var = square_sums / x.shape[0] - mean ** 2
#     std = np.sqrt(np.maximum(var, 1e-10))
#     x = np.divide(x, std)

#     return x

# def spectrogram(aud):
#     waveform = torch.tensor(aud, dtype=torch.float32).unsqueeze(dim=0)
#     waveform = waveform * (2 ** 15)
#     kaldi = torchaudio.compliance.kaldi.fbank(waveform, num_mel_bins=80, window_type='hanning')
#     kaldi = normalize(kaldi)

#     return kaldi.transpose(0,1)

# def write_optimal_delays(filename, result):
#     if os.path.exists(filename):
#         with open(filename, 'rb') as f:
#             prev_result = pickle.load(f)
#         result['corr'] = np.concatenate([prev_result['corr'], result['corr']], axis=0)
#         result['loss'] = np.concatenate([prev_result['loss'], result['loss']], axis=0)
#         result['delays'] = np.concatenate([prev_result['delays'], result['delays']], axis=0)
#         # temporary change...should be removed after run..!
#         # result['delays'] = np.concatenate([np.arange(0,201,10), result['delays']], axis=0)
        

#     with open(filename, 'wb') as file:
#         pickle.dump(result, file)
