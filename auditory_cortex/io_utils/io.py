import os
import re
import numpy as np
import pandas as pd
import gzip
import pickle
from auditory_cortex import opt_inputs_dir, results_dir, cache_dir, normalizers_dir
from auditory_cortex import valid_model_names
from memory_profiler import profile
import logging
logger = logging.getLogger(__name__)


def sanitize_string(s):
    """Sanitize strings for safe filenames."""
    return re.sub(r'[^\w.-]', '_', str(s))

def settings_to_name(settings: dict) -> str:
    parts = [f"{k}-{sanitize_string(v)}" for k, v in sorted(settings.items())]
    return "_".join(parts)

def write_dict(dict, filepath):
    """Writes a dictionary to a file using gzip compression."""
    with gzip.open(filepath, 'wb') as f:
        pickle.dump(dict, f)
    logger.info(f"Dictionary saved to {filepath}")

def read_dict(filepath):
    try:
        with gzip.open(filepath, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        logger.warning(f"File not found: {filepath}")
        return None
################################################################
######      Normalizer read/write functions....STARTS HERE
def read_inter_trial_corr_dists(
        session, bin_width, delay, mVocs=False, dataset_name='ucsf'
        ):
    """Writes distributions (True & Null both) of trial-trial correlations
    for the given selection.
    """
    bin_width = int(bin_width)
    delay = int(delay)
    if dataset_name != 'ucsf':
        parent_dir = os.path.join(normalizers_dir, dataset_name)
    else:
        parent_dir = normalizers_dir
    if mVocs:
        parent_dir = os.path.join(parent_dir, 'mVocs')

    norm_dir = os.path.join(parent_dir, 'norm_dist')
    null_dir = os.path.join(parent_dir, 'null_dist')

    norm_dist = read_dict(
        os.path.join(norm_dir, f"norm_bw_{bin_width}ms_sess_{session}.pkl.gz")
        )
    null_dist = read_dict(
        os.path.join(null_dir, f"null_bw_{bin_width}ms_sess_{session}.pkl.gz")
        )
    return norm_dist, null_dist

# def write_inter_trial_corr_dists(
#         norm_dist, null_dist, 
#         session, bin_width, delay, mVocs=False, dataset_name='ucsf'
#         ):
#     """Writes distributions (True & Null both) of trial-trial correlations
#     for the given selection.
#     """
#     bin_width = int(bin_width)
#     delay = int(delay)
#     if dataset_name != 'ucsf':
#         parent_dir = os.path.join(normalizers_dir, dataset_name)
#     else:
#         parent_dir = normalizers_dir
#     if mVocs:
#         parent_dir = os.path.join(parent_dir, 'mVocs')

#     norm_dir = os.path.join(parent_dir, 'norm_dist')
#     null_dir = os.path.join(parent_dir, 'null_dist')

#     os.makedirs(norm_dir, exist_ok=True)
#     os.makedirs(null_dir, exist_ok=True)

#     write_dict(
#         norm_dist,
#         os.path.join(norm_dir, f"norm_bw_{bin_width}ms_sess_{session}.pkl.gz")
#     )
#     write_dict(
#         null_dist,
#         os.path.join(null_dir, f"null_bw_{bin_width}ms_sess_{session}.pkl.gz")
#     )


def read_inter_trial_corr_dists(
        session, bin_width, mVocs=False, dataset_name='ucsf', 
        **kwargs,
        ):
    """Reads the distribution of normalizers (both True & Null) from cache.

    Args:
        session: int = session ID (e.g. 200206)
        bin_width: int = bin width in ms
        mVocs: bool = if True, uses mVocs directory
        dataset_name: str = name of the dataset (default: 'ucsf')

        Kwargs:
            bootstrap: bool = if True, uses bootstrap method (default: False)
            epoch: int = epoch index (default: None)   
                None means no bootstrap, just save the distribution
            percent_dur: int = percentage of duration to consider for bootstrap (default: None)
            num_trial: int = number of trials to consider for bootstrap (default: None)

    Returns:
        norm_dist: dict = distribution of normalizers
        null_dist: dict = distribution of null correlations
    """
    bootstrap = kwargs.get('bootstrap', False)
    epoch = kwargs.get('epoch', None)
    percent_dur = kwargs.get('percent_dur', None)
    num_trial = kwargs.get('num_trial', None)
    if bootstrap:
        assert epoch is not None, "epoch id must be specified for bootstrap method"
        assert percent_dur is not None, "percent_dur must be specified for bootstrap method"
        assert num_trial is not None, "num_trial must be specified for bootstrap method"

        percent_dur = int(percent_dur)
        num_trial = int(num_trial)
        epoch = int(epoch)

    bin_width = int(bin_width)
    session = int(session)
    
    if dataset_name != 'ucsf':
        parent_dir = os.path.join(normalizers_dir, dataset_name)
    else:
        parent_dir = normalizers_dir
    if mVocs:
        parent_dir = os.path.join(parent_dir, 'mVocs')
    if bootstrap:
        parent_dir = os.path.join(parent_dir, 'bootstrap')

    norm_dir = os.path.join(parent_dir, 'norm_dist')
    null_dir = os.path.join(parent_dir, 'null_dist')

    os.makedirs(norm_dir, exist_ok=True)
    os.makedirs(null_dir, exist_ok=True)

    settings = {
        'epoch': epoch, 
        'percent_dur': percent_dur,
        'num_trial': num_trial, 
        'session': session,
        'bin_width': bin_width, 
    }
    filename = f"{settings_to_name(settings)}.pkl.gz"
    norm_dist = read_dict(os.path.join(norm_dir, filename))
    null_dist = read_dict(os.path.join(null_dir, filename))
    return norm_dist, null_dist


def write_inter_trial_corr_dists(
        norm_dist, null_dist, 
        session, bin_width, mVocs=False, dataset_name='ucsf', 
        **kwargs,
        ):
    """Writes the distribution of normalizers (both True & Null) to cache.

    Args:
        norm_dist: dict = distribution of normalizers
        null_dist: dict = distribution of null correlations
        session: int = session ID (e.g. 200206)
        bin_width: int = bin width in ms
        mVocs: bool = if True, uses mVocs directory
        dataset_name: str = name of the dataset (default: 'ucsf')

        Kwargs:
            bootstrap: bool = if True, uses bootstrap method (default: False)
            itr: int = number of bootstrap iterations (default: None)   
                None means no bootstrap, just save the distribution
            percent_dur: int = percentage of duration to consider for bootstrap (default: None)
            num_trial: int = number of trials to consider for bootstrap (default: None)
    """
    bootstrap = kwargs.get('bootstrap', False)
    epoch = kwargs.get('epoch', None)
    percent_dur = kwargs.get('percent_dur', None)
    num_trial = kwargs.get('num_trial', None)
    if bootstrap:
        assert epoch is not None, "itr must be specified for bootstrap method"
        assert percent_dur is not None, "percent_dur must be specified for bootstrap method"
        assert num_trial is not None, "num_trial must be specified for bootstrap method"

        percent_dur = int(percent_dur)
        num_trial = int(num_trial)
        epoch = int(epoch)

    bin_width = int(bin_width)
    session = int(session)
    
    if dataset_name != 'ucsf':
        parent_dir = os.path.join(normalizers_dir, dataset_name)
    else:
        parent_dir = normalizers_dir
    if mVocs:
        parent_dir = os.path.join(parent_dir, 'mVocs')
    if bootstrap:
        parent_dir = os.path.join(parent_dir, 'bootstrap')

    norm_dir = os.path.join(parent_dir, 'norm_dist')
    null_dir = os.path.join(parent_dir, 'null_dist')

    os.makedirs(norm_dir, exist_ok=True)
    os.makedirs(null_dir, exist_ok=True)

    settings = {
        'epoch': epoch, 
        'percent_dur': percent_dur,
        'num_trial': num_trial, 
        'session': session,
        'bin_width': bin_width, 
    }
    filename = f"{settings_to_name(settings)}.pkl.gz"
    write_dict(norm_dist, os.path.join(norm_dir, filename))
    write_dict(null_dist, os.path.join(null_dir, filename))

#-----------      cache bootstrap median dist    -----------

def read_bootstrap_median_dist(
        model_name, bin_width=50, mVocs=False, verbose=True,
        test=False, dataset_name='ucsf'
        ):
    """Reads distribution of medians for standard error of mean using
     bootstrap method.

     Args:
        test: bool = if True, reads from test directory
    """
    bin_width = int(bin_width)
    # layer_ID = int(layer_ID)
    path_dir = os.path.join(cache_dir, 'bootstrap', f'{model_name}')
    if dataset_name != 'ucsf':
        path_dir = os.path.join(path_dir, dataset_name)
    if test:
        path_dir = os.path.join(path_dir, 'test')
    if mVocs:
        path_dir = os.path.join(path_dir, 'mVocs')
    filename = f'{model_name}_bootstrap_medians_{bin_width}ms.pkl'   
    file_path = os.path.join(path_dir, filename)
    
    if os.path.exists(file_path):
        if verbose:
            logger.info(f"Reading from file: {file_path}")
        with open(file_path, 'rb') as F: 
            reg_results = pickle.load(F)
        return reg_results
    else:
        logger.info(f"Results not found.")
        return None
    
def write_bootstrap_median_dist(
        median_dist, model_name, bin_width=50, mVocs=False, verbose=True,
        test=False, dataset_name='ucsf'
        ):
    """Reads distribution of medians for standard error of mean using
     bootstrap method.
    """
    bin_width = int(bin_width)
    # layer_ID = int(layer_ID)
    path_dir = os.path.join(cache_dir, 'bootstrap', f'{model_name}')
    if dataset_name != 'ucsf':
        path_dir = os.path.join(path_dir, dataset_name)
    if test:
        path_dir = os.path.join(path_dir, 'test')
    if mVocs:
        path_dir = os.path.join(path_dir, 'mVocs')
    filename = f'{model_name}_bootstrap_medians_{bin_width}ms.pkl'   
    file_path = os.path.join(path_dir, filename)

    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
        logger.info(f"Directory path created: {path_dir}")
    

    with open(file_path, 'wb') as F: 
        pickle.dump(median_dist, F)
    logger.info(f"trf parameters saved for {model_name} at path: \n {file_path}.")


#-----------      Null distribution using poisson sequences    -----------#

def read_normalizer_null_distribution_using_poisson(bin_width, spike_rate, mVocs=False, dataset_name='ucsf'):
    """Retrieves null distribution of correlations computed using poisson sequences."""
    bin_width = int(bin_width)
    # path_dir = os.path.join(results_dir, 'normalizers', 'null_distribution')
    if dataset_name != 'ucsf':
        parent_dir = os.path.join(normalizers_dir, dataset_name)
    else:
        parent_dir = normalizers_dir
        
    post_str = ''
    if mVocs:
        parent_dir = os.path.join(parent_dir, 'mVocs')
        post_str = ' (mVocs)'

    path_dir = os.path.join(parent_dir, 'null_distribution')
    file_path = os.path.join(path_dir, f"normalizers_null_dist_poisson_bw_{bin_width}ms_spike_rate_{spike_rate}hz.pkl")
    if os.path.exists(file_path):
        with open(file_path, 'rb') as F: 
            norm_null_dist = pickle.load(F)
        return norm_null_dist
    else:
        logger.info(f"Null dist.{post_str} not found: for bin-width {bin_width}ms and {spike_rate}Hz spike rate.")
        return None

def write_normalizer_null_distribution_using_poisson(
        bin_width, spike_rate, null_dist_poisson, mVocs=False, dataset_name='ucsf'):
    """Writes null distribution of correlations computed using poisson sequences for the given selection."""
    bin_width = int(bin_width)
    # path_dir = os.path.join(results_dir, 'normalizers', 'null_distribution')
    if dataset_name != 'ucsf':
        parent_dir = os.path.join(normalizers_dir, dataset_name)
    else:
        parent_dir = normalizers_dir
        
    if mVocs:
        parent_dir = os.path.join(parent_dir, 'mVocs')
    
    path_dir = os.path.join(parent_dir, 'null_distribution')
    if not os.path.exists(path_dir):
        logger.info(f"Path not found, creating directories...")
        os.makedirs(path_dir)
    file_path = os.path.join(path_dir, f"normalizers_null_dist_poisson_bw_{bin_width}ms_spike_rate_{spike_rate}hz.pkl")
    
    with open(file_path, 'wb') as F: 
        pickle.dump(null_dist_poisson, F)
    logger.info(f"Null dist. poisson saved to: {file_path}")


# #-----------      Null distribution using sequence shifts    -----------#

# def read_normalizer_null_distribution_random_shifts(
#         session, bin_width, dataset_name='ucsf'
#         ):
#     """Retrieves null distribution of correlations computed using randomly 
#         shifted spike sequence of one trial vs (non-shifted) seconds trial."""
#     bin_width = int(bin_width)
#     session = str(int(float(session)))
#     if dataset_name != 'ucsf':
#         parent_dir = os.path.join(normalizers_dir, dataset_name)
#     else:
#         parent_dir = normalizers_dir
#     path_dir = os.path.join(parent_dir, 'null_distribution', 'shifted_sequence')
#     file_path = f"shifted_null_bw_{bin_width}ms_sess_{session}.npz"
#     file_path = os.path.join(path_dir, file_path)
#     if os.path.exists(file_path):
#         data = np.load(file_path)
#         null_dist = data['null_dist']
#         return null_dist
#     else:
#         logger.info(f"Null dist. not found: for bin-width {bin_width}ms and session {session}.")
#         return None

# def write_normalizer_null_distribution_using_random_shifts(
#         session, bin_width, null_dist_sess, dataset_name='ucsf'
#         ):
#     """Writes null distribution of correlations computed using poisson sequences for the given selection."""
#     bin_width = int(bin_width)
#     session = str(int(float(session)))
#     if dataset_name != 'ucsf':
#         parent_dir = os.path.join(normalizers_dir, dataset_name)
#     else:
#         parent_dir = normalizers_dir
#     path_dir = os.path.join(parent_dir, 'null_distribution', 'shifted_sequence')
#     os.makedirs(path_dir, exist_ok=True)
#     file_path = f"shifted_null_bw_{bin_width}ms_sess_{session}.npz"
#     file_path = os.path.join(path_dir, file_path)

#     np.savez_compressed(file_path, null_dist=null_dist_sess)
#     logger.info(f"Shifted Null dist. saved to: {file_path}")

# #-----------  Normalizer distribution using all possible pairs of trials  ----------#

# def read_normalizer_distribution(
#         bin_width, delay, session, method='app', mVocs=False, dataset_name='ucsf'
#         ):
#     """Retrieves distribution of normalizers for the given selection.
    
#     Args:
#         bin_width: int = bin width in ms
#         delay: int = delay in ms
#         session: str = session ID (e.g. 200206)
#     """
#     bin_width = int(bin_width)
#     delay = int(delay)
#     if method == 'app':
#         subdir = 'all_possible_pairs'
#     else:
#         subdir = 'random_pairs'

#     if dataset_name != 'ucsf':
#         parent_dir = os.path.join(normalizers_dir, dataset_name)
#     else:
#         parent_dir = normalizers_dir

#     if mVocs:
#         parent_dir = os.path.join(parent_dir, 'mVocs')

#     path_dir = os.path.join(parent_dir, subdir)
#     file_path = os.path.join(path_dir, f"normalizers_bw_{bin_width}ms_delay_{delay}ms_sess_{session}.npz")
#     if os.path.exists(file_path):
#         data = np.load(file_path)
#         norm_dist = data['dist']
#         return norm_dist
#     else:
#         logger.info(f"Normalizer not found: for bw {bin_width}ms, delay {delay}ms and session {session}.")
#         return None
#     # file_path = os.path.join(path_dir, f"normalizers_bw_{bin_width}ms_delay_{delay}ms_sess_{session}.pkl")
#     # if os.path.exists(file_path):
#     #     with open(file_path, 'rb') as F: 
#     #         normalizers_dist = pickle.load(F)
#     #     return normalizers_dist
#     # else:
#     #     logger.info(f"Normalizers not found: for bin-width {bin_width}ms and delay {delay}ms.")
#     #     return None

# def write_normalizer_distribution(
#         session, bin_width, delay, normalizer_dist, method='app', mVocs=False, dataset_name='ucsf'
#         ):
#     """Writes distribution of normalizers for the given selection."""
#     bin_width = int(bin_width)
#     delay = int(delay)
#     if method == 'app':
#         subdir = 'all_possible_pairs'
#     else:
#         subdir = 'random_pairs'
#     # path_dir = os.path.join(results_dir, 'normalizer', subdir)
#     if dataset_name != 'ucsf':
#         parent_dir = os.path.join(normalizers_dir, dataset_name)
#     else:
#         parent_dir = normalizers_dir

#     if mVocs:
#         parent_dir = os.path.join(parent_dir, 'mVocs')

#     path_dir = os.path.join(parent_dir, subdir)
#     if not os.path.exists(path_dir):
#         logger.info(f"Path not found, creating directories...")
#         os.makedirs(path_dir)

#     file_path = os.path.join(path_dir, f"normalizers_bw_{bin_width}ms_delay_{delay}ms_sess_{session}.npz")
#     np.savez_compressed(file_path, dist=normalizer_dist)
#     logger.info(f"Writing normalizer dictionary to the {file_path}")
#     # file_path = os.path.join(path_dir, f"normalizers_bw_{bin_width}ms_delay_{delay}ms.pkl")
#     # norm_dict_all_sessions = read_normalizer_distribution(
#     #     bin_width, delay, method=method, mVocs=mVocs, dataset_name=dataset_name
#     #     )
    
#     # if norm_dict_all_sessions is None:
#     #     norm_dict_all_sessions = {}

#     # norm_dict_all_sessions[session] = normalizer_dist
#     # with open(file_path, 'wb') as F: 
#     #     pickle.dump(norm_dict_all_sessions, F)
#     # logger.info(f"Writing normalizer dictionary to the {file_path}")


######      Normalizer read/write functions....END HERE
################################################################


    


#-----------      cache TRF parameters    -----------#

def read_trf_parameters(
        model_name, session, bin_width=50,
        shuffled=False, layer_ID=None, LPF=False,
        mVocs=False, dataset_name='ucsf',
        lag=300
        ):
    """Reads parameters of GLM alongthwith neural spikes and natural parameters 
    model_name, returns a dictionary.
    """
    session = int(session)
    bin_width = int(bin_width)
    # layer_ID = int(layer_ID)
    logger.info(f"Reading TRF parameters for {model_name}, session-{session}," +\
           f"bin-width-{bin_width}ms, shuffled-{shuffled}, LPF-{LPF}")
    path_dir = os.path.join(cache_dir, 'trf', f'{model_name}')
    if dataset_name != 'ucsf':
        path_dir = os.path.join(path_dir, dataset_name)
    if mVocs:
        path_dir = os.path.join(path_dir, 'mVocs')
    if shuffled:
        path_dir = os.path.join(path_dir, 'shuffled')
    if LPF:
        path_dir = os.path.join(path_dir, 'LPF')

    if model_name == 'strf':
        filename = f'{model_name}_sess_{session}_trf{lag}_{bin_width}ms.pkl' 
    else:
        filename = f'{model_name}_layer_{layer_ID}_sess_{session}_trf{lag}_{bin_width}ms.pkl'
    
    file_path = os.path.join(path_dir, filename)
    if os.path.exists(file_path):
        logger.info(f"Reading from file: {file_path}")
        with open(file_path, 'rb') as F: 
            parameters = pickle.load(F)
        return parameters
    else:
        logger.info(f"Results not found.")
        return None

def write_trf_parameters(
        model_name, session, parameters, bin_width=50,
        shuffled=False, layer_ID=None, LPF=False,
        mVocs=False, dataset_name='ucsf',
        lag=300
        ):
    """writes the lmbdas, separate file for every model..
    """
    session = int(session)
    if layer_ID is not None:
        layer_ID = int(layer_ID)

    path_dir = os.path.join(cache_dir, 'trf', f'{model_name}')
    if dataset_name != 'ucsf':
        path_dir = os.path.join(path_dir, dataset_name)
    if mVocs:
        path_dir = os.path.join(path_dir, 'mVocs')
    if shuffled:
        path_dir = os.path.join(path_dir, 'shuffled')
    if LPF:
        path_dir = os.path.join(path_dir, 'LPF')
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
        logger.info(f"Directory path created: {path_dir}")

    if model_name == 'strf':
        filename = f'{model_name}_sess_{session}_trf{lag}_{bin_width}ms.pkl' 
    else:
        filename = f'{model_name}_layer_{layer_ID}_sess_{session}_trf{lag}_{bin_width}ms.pkl'
    file_path = os.path.join(path_dir, filename)
    
    with open(file_path, 'wb') as F: 
        pickle.dump(parameters, F)
    logger.info(f"trf parameters saved for {model_name} at path: \n {file_path}.")

# def write_alphas(
#         model_name, session, alphas, bin_width=50,
#         shuffled=False, layer_ID=None, LPF=False,
#         mVocs=False, dataset_name='ucsf',
#         lag=300
#         ):
#     """writes the lmbdas, separate file for every model..
#     """
#     session = int(session)
#     if layer_ID is not None:
#         layer_ID = int(layer_ID)

#     path_dir = os.path.join(cache_dir, 'trf', f'{model_name}')
#     if dataset_name != 'ucsf':
#         path_dir = os.path.join(path_dir, dataset_name)
#     if mVocs:
#         path_dir = os.path.join(path_dir, 'mVocs')
#     if shuffled:
#         path_dir = os.path.join(path_dir, 'shuffled')
#     if LPF:
#         path_dir = os.path.join(path_dir, 'LPF')
#     path_dir = os.path.join(path_dir, 'alphas')
#     if not os.path.exists(path_dir):
#         os.makedirs(path_dir)
#         logger.info(f"Directory path created: {path_dir}")


#     if model_name == 'strf':
#         filename = f'{model_name}_sess_{session}_trf{lag}_{bin_width}ms.pkl' 
#     else:
#         filename = f'{model_name}_layer_{layer_ID}_sess_{session}_trf{lag}_{bin_width}ms.pkl'

#     file_path = os.path.join(path_dir, filename)
#     with open(file_path, 'wb') as F: 
#         pickle.dump(alphas, F)
#     logger.info(f"trf parameters saved for {model_name} at path: \n {file_path}.")

# def read_alphas(
#         model_name, session, bin_width=50,
#         shuffled=False, layer_ID=None, LPF=False,
#         mVocs=False, bias=False, dataset_name='ucsf',
#         lag=300
#         ):
#     """writes the lmbdas, separate file for every model..
#     """
#     session = int(session)
#     if layer_ID is not None:
#         layer_ID = int(layer_ID)

#     path_dir = os.path.join(cache_dir, 'trf', f'{model_name}')
#     if dataset_name != 'ucsf':
#         path_dir = os.path.join(path_dir, dataset_name)
#     if mVocs:
#         path_dir = os.path.join(path_dir, 'mVocs')
#     if shuffled:
#         path_dir = os.path.join(path_dir, 'shuffled')
#     if LPF:
#         path_dir = os.path.join(path_dir, 'LPF')
#     path_dir = os.path.join(path_dir, 'alphas')
#     if not os.path.exists(path_dir):
#         os.makedirs(path_dir)
#         logger.info(f"Directory path created: {path_dir}")


#     if model_name == 'strf':
#         filename = f'{model_name}_sess_{session}_trf{lag}_{bin_width}ms.pkl' 
#     else:
#         filename = f'{model_name}_layer_{layer_ID}_sess_{session}_trf{lag}_{bin_width}ms.pkl'

#     file_path = os.path.join(path_dir, filename)
#     if os.path.exists(file_path):
#         logger.info(f"Reading from file: {file_path}")
#         with open(file_path, 'rb') as F: 
#             alphas = pickle.load(F)
#         return alphas
#     else:
#         logger.info(f"Results not found.")
#         return None




#-----------      Null distribution using poisson sequences    -----------#

def read_significant_sessions_and_channels(bin_width, p_threshold, use_poisson_null=True, mVocs=False):
    """Retrieves significant sessions and channels at the specified bin width."""

    if use_poisson_null:
        subdir = 'using_poisson_null'
    else:
        subdir = 'using_shifts_null'
    # path_dir = os.path.join(results_dir, 'normalizers', 'significant_neurons', subdir)
    if mVocs:
        path_dir = os.path.join(normalizers_dir, 'significant_neurons', 'mVocs', subdir)
    else:
        path_dir = os.path.join(normalizers_dir, 'significant_neurons', subdir)
    file_path = os.path.join(path_dir, f"significant_sessions_and_channels_bw_{bin_width}ms_pvalue_{p_threshold}.pkl")
    logger.info(f"Reading sig sessions/channels from: {file_path}")
    if os.path.exists(file_path):
        with open(file_path, 'rb') as F: 
            significant_sessions_and_channels = pickle.load(F)
        return significant_sessions_and_channels
    else:
        logger.info(f"Sigificant sessions/channels data not found: for bin-width {bin_width}ms.")
        return None

def write_significant_sessions_and_channels(
        bin_width, p_threshold, significant_sessions_and_channels,
        use_poisson_null=True, mVocs=False):
    """Writes significant sessions and channels at the specified bin width."""
    if use_poisson_null:
        subdir = 'using_poisson_null'
    else:
        subdir = 'using_shifts_null'

    # path_dir = os.path.join(results_dir, 'normalizers', 'significant_neurons', subdir)
    # path_dir = os.path.join(normalizers_dir, 'significant_neurons', subdir)
    if mVocs:
        path_dir = os.path.join(normalizers_dir, 'significant_neurons', 'mVocs', subdir)
    else:
        path_dir = os.path.join(normalizers_dir, 'significant_neurons', subdir)
    if not os.path.exists(path_dir):
        logger.info(f"Path not found, creating directories...")
        os.makedirs(path_dir)

    file_path = os.path.join(path_dir, f"significant_sessions_and_channels_bw_{bin_width}ms_pvalue_{p_threshold}.pkl")
    with open(file_path, 'wb') as F: 
        pickle.dump(significant_sessions_and_channels, F)
    logger.info(f"Sigificant sessions/channels saved to: {file_path}")





def read_cached_glm_parameters(model_name, session, bin_width=20, shuffled=False):
    """Reads parameters of GLM alongthwith neural spikes and natural parameters 
    model_name, returns a dictionary.
    """
    session = int(session)
    bin_width = int(bin_width)
    # layer_ID = int(layer_ID)
    if shuffled:
        path_dir = os.path.join(cache_dir, 'glm', f'{model_name}', 'shuffled')
    else:
        path_dir = os.path.join(cache_dir, 'glm', f'{model_name}')
    filename = f'{model_name}_sess_{session}_glm_parameters_{bin_width}ms.pkl'   
    file_path = os.path.join(path_dir, filename)
    
    if os.path.exists(file_path):
        logger.info(f"Reading from file: {file_path}")
        with open(file_path, 'rb') as F: 
            reg_results = pickle.load(F)
        return reg_results
    else:
        logger.info(f"Results not found.")
        return None

def cache_glm_parameters(model_name, layer_ID, session,
        neural_spikes, glm_coefficients, natural_param, bin_width=20,
        shuffled=False
        ):
    """writes the lmbdas, separate file for every model..
    """
    session = int(session)
    layer_ID = int(layer_ID)
    if shuffled:
        path_dir = os.path.join(cache_dir, 'glm', f'{model_name}', 'shuffled')
    else:
        path_dir = os.path.join(cache_dir, 'glm', f'{model_name}')
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
        logger.info(f"Directory path created: {path_dir}")

    filename = f'{model_name}_sess_{session}_glm_parameters_{bin_width}ms.pkl' 
    file_path = os.path.join(path_dir, filename)
    
    exisiting_results = read_cached_glm_parameters(
        model_name, session, bin_width=bin_width, shuffled=shuffled)
    if exisiting_results is None:
        exisiting_results = {}

    # save/update results for model_name
    layer_results = {
        'neural_spikes': neural_spikes,
        'glm_coefficients': glm_coefficients,
        'natural_param': natural_param,
    }

    exisiting_results[layer_ID] = layer_results
    with open(file_path, 'wb') as F: 
        pickle.dump(exisiting_results, F)
    logger.info(f"Results saved for {model_name} at path: \n {file_path}.")



def read_lmbdas(
        model_name, layer_ID, session, bin_width=20, shuffled=False
        ):
    """Reads optimal lmbdas and lmbda losses, for the 
    model_name, returns a dictionary.
    """
    session = int(session)
    layer_ID = int(layer_ID)
    if shuffled:
        path_dir = os.path.join(results_dir, 'lmbdas', 'shuffled')
    else:
        path_dir = os.path.join(results_dir, 'lmbdas')
    filename = f'{model_name}_l{layer_ID}_sess_{session}_optimal_lmbdas_and_losses_{bin_width}ms.pkl'   
    file_path = os.path.join(path_dir, filename)
    
    if os.path.exists(file_path):
        logger.info(f"Reading from file: {file_path}")
        with open(file_path, 'rb') as F: 
            reg_results = pickle.load(F)
        return reg_results
    else:
        logger.info(f"Results not found.")
        return None


def write_lmbdas(
        model_name, layer_ID, session, optimal_lmbdas, lmbda_loss,
        bin_width=20, shuffled=False
                 
                 ):
        """writes the lmbdas, separate file for every model..
        """
        session = int(session)
        layer_ID = int(layer_ID)
        if shuffled:
            path_dir = os.path.join(results_dir, 'lmbdas', 'shuffled')
        else:
            path_dir = os.path.join(results_dir, 'lmbdas')
        if not os.path.exists(path_dir):
            os.makedirs(path_dir)
            logger.info(f"Directory path created: {path_dir}")
        
        filename = f'{model_name}_l{layer_ID}_sess_{session}_optimal_lmbdas_and_losses_{bin_width}ms.pkl'    
        file_path = os.path.join(path_dir, filename)
        
        exisiting_results = read_lmbdas(
            model_name, layer_ID, session, bin_width=bin_width, shuffled=shuffled
            )
        if exisiting_results is None:
            exisiting_results = {}

        # save/update results for model_name
        exisiting_results['optimal_lmbdas'] = optimal_lmbdas
        exisiting_results['lmbda_loss'] = lmbda_loss
        with open(file_path, 'wb') as F: 
            pickle.dump(exisiting_results, F)
        logger.info(f"Results saved for {model_name} at path: \n {file_path}.")


def read_WER():

    path_dir = os.path.join(results_dir, 'task_optimization')
    filename = f'pretrained_networks_WERs.csv'    
    file_path = os.path.join(path_dir, filename)
    if os.path.isfile(file_path):
        logger.info("Reading existing WER results")
        return pd.read_csv(file_path, index_col=0)
    else:
        return None
    
def write_WER(model_name, benchmark, wer):
    """Writes WER evaluated on the benchmark specified 
    for the model_name"""
    path_dir = os.path.join(results_dir, 'task_optimization')
    filename = f'pretrained_networks_WERs.csv'    
    file_path = os.path.join(path_dir, filename)
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
        logger.info(f"Directory path created: {path_dir}")

    df = read_WER()
    if df is None:
        df = pd.DataFrame()
    if benchmark not in df.columns:
        df[benchmark] = np.nan
    if model_name not in df.index:
        df.loc[model_name] = pd.Series(np.nan)
    
    df.at[model_name, benchmark] = wer
    df.to_csv(file_path)
    logger.info(f"WER for {model_name}, on {benchmark} saved to {file_path}")
    



def read_reg_corr():
    """Reads the saved regression correlation results, 
    at 20ms bin width.
    Results are saved as dictionary of dictionaries,
    organized as follows:
        'deepspeech2:   
                'core':
                    0: ndarray
                    1: ndarray
                    .
                'belt':
                    0: ndarray
                    1: ndarray


    """
    path_dir = os.path.join(results_dir, 'Reg')
    filename = f'reg_correlations_normalized_20ms.pkl'    
    file_path = os.path.join(path_dir, filename)
    
    if os.path.exists(file_path):
        logger.info(f"Reading from file: {file_path}")
        with open(file_path, 'rb') as F: 
            reg_results = pickle.load(F)
        return reg_results
    else:
        logger.info(f"Results not found.")
        return None
def write_reg_corr(model_name, results_dict):
    """writes the regression correlation results, 
    at 20ms bin width.
    Results are saved as dictionary of dictionaries,
    organized as follows:
        'deepspeech2:   
                'core':
                    0: ndarray
                    1: ndarray
                    .
                'belt':
                    0: ndarray
                    1: ndarray


    """
    path_dir = os.path.join(results_dir, 'Reg')
    
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
        logger.info(f"Directory path created: {path_dir}")

    filename = f'reg_correlations_normalized_20ms.pkl'    
    file_path = os.path.join(path_dir, filename)
    
    exisiting_results = read_reg_corr()
    if exisiting_results is None:
        exisiting_results = {}

    # save/update results for model_name
    exisiting_results[model_name] = results_dict
    with open(file_path, 'wb') as F: 
        pickle.dump(exisiting_results, F)
    logger.info(f"Results saved for {model_name} at path: \n {file_path}.")


def read_model_parameters(model_name):
    """Retrieves model parameters/betas at path 
    determined using the model_name.
    """
    dirpath = os.path.join(opt_inputs_dir, model_name)
    filepath = os.path.join(dirpath, f"{model_name}_beta_bank.pkl")
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            beta_bank = pickle.load(f)
        return beta_bank
    else:
        return None


def write_model_parameters(
        model_name, session, coefficents):
    """Writes/updates model parameters/betas at path 
    determined using the model_name.
    """
    # check if path exists, create if doesn't exist.
    dirpath = os.path.join(opt_inputs_dir, model_name)
    if not os.path.exists(dirpath):
        logger.info("Does not exist")
        os.makedirs(dirpath)

    # loading existing betas or creating new (if not available already)
    beta_bank = read_model_parameters(model_name=model_name)
    if beta_bank is None:
        logger.info(f"Creating new beta bank")
        beta_bank = {}

    beta_bank[session] = coefficents

    filepath = os.path.join(dirpath, f"{model_name}_beta_bank.pkl")
    with open(filepath, 'wb') as f:
        pickle.dump(beta_bank, f) 

    logger.info(f"Parameters computed and saved for {model_name}, sess-{session}")



def read_RDM(model_name, identifier, bin_width=None, verbose=True):
    """Retrieves the dictionary of RSA matrices for the given
    model.
    
    Args:
        model_name: str = model_name from ['neural']U[*regression_models]

    Returns:
        dict = RSA matrices for different layers or neural areas.
    """
    if 'global' in identifier:
        id = '_global_clipped'
    elif 'average' in identifier:
        id = '_spikes_averaged'
    else:
        id = ''

    path_dir = os.path.join(results_dir, 'RSA', model_name)
    if bin_width is not None:
        model_name += f'_{bin_width}_ms_'
    filename = f'{model_name}_RSA{id}_results.pkl'    
    file_path = os.path.join(path_dir, filename)
    if verbose:
        logger.info(f"Reading from file: {file_path}")
    if os.path.exists(file_path):
        with open(file_path, 'rb') as F: 
            rsa_dict = pickle.load(F)
        return rsa_dict
    else:
        None


def write_RDM(model_name, key, matrix, identifier, bin_width=None,
                       verbose=True):
    """Writes given RSA matrix to the dictionary of RSA matrices 
    for the given model, pointed to by the given key.
    
    Args:
        model_name: str = model_name from ['neural']U[*regression_models]
        key: str or int = Layer ID for models or neural area.
        matrix: ndarray = num_stimuli x num_stimuli
    """
    if 'global' in identifier:
        id = '_global_clipped'
    elif 'average' in identifier:
        id = '_spikes_averaged'
    else:
        id = ''

    path_dir = os.path.join(results_dir, 'RSA', model_name)
    # check if directory structure exists..
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
        logger.info(f"Directory path created: {path_dir}")
    # read existing dict of matrices
    rsa_dict = read_RDM(model_name, identifier=identifier,
                                bin_width=bin_width)
    if rsa_dict is None:
        rsa_dict = {}

    # filename to write back to..
    if bin_width is not None:
        model_name += f'_{bin_width}_ms_'
    filename = f'{model_name}_RSA{id}_results.pkl'    
    file_path = os.path.join(path_dir, filename)

    # updata value and write back
    rsa_dict[key] = matrix
    with open(file_path, 'wb') as F: 
        pickle.dump(rsa_dict, F)
    if verbose:
        logger.info(f"RSA saved to '{file_path}'")

def delete_saved_RDM(model_name, identifier, bin_width):
    """
    Removes file containing RSA results for the given selection,
    """
    if 'global' in identifier:
        id = '_global_clipped'
    elif 'average' in identifier:
        id = '_spikes_averaged'
    else:
        id = ''

    path_dir = os.path.join(results_dir, 'RSA', model_name)
    if bin_width is not None:
        model_name += f'_{bin_width}_ms_'
    filename = f'{model_name}_RSA{id}_results.pkl'    
    file_path = os.path.join(path_dir, filename)

    if os.path.exists(file_path):
        os.remove(file_path)
        logger.info(f"Removed: {file_path}")
    else:
        logger.info(f"File does not exist.")


def read_cached_features(model_name, dataset_name, contextualized=False, shuffled=False, mVocs=False):
    """Retrieves cached features from the cache_dir, returns None if 
    features not cached already. 

    Args:
        model_name: str specifying model name, possible choices are
            ['wav2letter_modified', 'wav2vec2', 'speech2text',
            'deepspeech2', 'whiper_tiny', 'whisper_base', 'whisper_small']
    """
    assert model_name in valid_model_names, f"Invalid model name '{model_name}' specified!"
    logger.info(f"Reading features for model: {model_name}")
    
    # if contextualized:
    #     logger.info(f"Reading contextualized features...")
    #     file_name = f"{model_name}_raw_features_contextualized.pkl"
    # else:
    #     file_name = f"{model_name}_raw_features.pkl"
    
    if mVocs:
        directory = os.path.join(cache_dir, 'mVocs')
    else:
        directory = cache_dir

    directory = os.path.join(directory, dataset_name)
    if shuffled:
        dir_path = os.path.join(directory, model_name, 'shuffled')
    else:
        dir_path = os.path.join(directory, model_name)
    # make sure directory structure is in place...
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    features = {}
    filenames = os.listdir(dir_path)
    if len(filenames) > 0:
        filenames.sort()
    for filename in filenames:
        if '.npz' in filename:
            loaded_data = np.load(os.path.join(dir_path, filename), allow_pickle=True)
            # loaded_dict = dict(loaded_data)
            loaded_dict = loaded_data['layer_features'].item()
            layer_id = filename.split('layer')[-1].split('.')[0]
            features[int(layer_id)] = loaded_dict
    
    if len(features) > 0:
        return features
    else:   
        return None

    # file_path = os.path.join(dir_path, file_name+'.gz')
    # if os.path.exists(file_path):
    #     logger.info(f"Reading raw features from {file_path}")
    #     # Load and decompress
    #     with gzip.open(file_path, 'rb') as f:
    #         features = pickle.load(f)
    #     # with open(file_path, 'rb') as F:
    #     #     features = pickle.load(F)
    #     return features
    # else:
    #     return None

# @profile
def write_cached_features(
        model_name, features, dataset_name, verbose=True, contextualized=False, shuffled=False,
        mVocs=False):
    """Writes features to the cache_dir,

    Args:
        model_name: str specifying model name, possible choices are
            ['wav2letter_modified', 'wav2vec2', 'speech2text',
            'deepspeech2', 'whiper_tiny', 'whisper_base', 'whisper_small']
        features: list = features for each layers as a list of dictionaries 
    """
    assert model_name in valid_model_names, f"Invalid model name '{model_name}' specified!"
    logger.info(f"Saving features for model: {model_name}")
    if contextualized:
        logger.info(f"writing contextualized features...")
        file_name = f"{model_name}_raw_features_contextualized"
    else:
        file_name = f"{model_name}_raw_features"

    if mVocs:
        directory = os.path.join(cache_dir, 'mVocs')
    else:
        directory = cache_dir
    directory = os.path.join(directory, dataset_name)
    if shuffled:
        dir_path = os.path.join(directory, model_name, 'shuffled')
    else:
        dir_path = os.path.join(directory, model_name)
    # make sure directory structure is in place...
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    for layer_id, layer_features in features.items():
        logger.info(f"Saving features for layer: {layer_id}")
        file_path = os.path.join(dir_path, f"{file_name}_layer{layer_id:02}.npz")
        np.savez_compressed(file_path, layer_features=layer_features)

    logger.info(f"All layer features saved to: {dir_path}")


    # file_path = os.path.join(dir_path, file_name+'.gz')
    # # Compress and save
    # with gzip.open(file_path, 'wb') as f:
    #     pickle.dump(features, f, protocol=pickle.HIGHEST_PROTOCOL)
    # logger.info(f"Features saved to file: {file_path}")
    
    # # # writing features to file
    # # with open(file_path, 'wb') as F:
    # #     pickle.dump(features, F)
    # logger.info(f"Features saved to file: {file_path}")
    
def read_cached_spikes(bin_width=20, threshold=0.068):
    """Retrieves neural spikes for the bin_width and area specified,
    this returns neural spikes in a format that is used for RSA.

    Args:
        bin_width: int = bin_width in ms
    """
    file_name = f"neural_spikes_{bin_width}ms_threshold_{threshold:.3f}.pkl"
    file_path = os.path.join(cache_dir, 'neural', file_name)

    if os.path.exists(file_path):
        with open(file_path, 'rb') as F:
            spikes = pickle.load(F)
        return spikes   
    return None

def write_cached_spikes(spikes, bin_width=20, area='all', threshold=0.068, verbose=True):
    """Retrieves neural spikes for the bin_width and area specified,
    this returns neural spikes in a format that is used for RSA.

    Args:
        spikes:  dict= spikes for significant channels (threshold=0.068)
        bin_width: int = bin_width in ms
        area: str = specify neural area, possible choices are
            ['all', 'core', 'belt'].
    """
    area_choices = ['core', 'belt', 'all']
    assert area in area_choices, f"Invalid neural area '{area}' specified!"

    file_name = f"neural_spikes_{bin_width}ms_threshold_{threshold:.3f}.pkl"
    file_path = os.path.join(cache_dir, 'neural', file_name)
    # make sure directory structure is in place...
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))

    cached_spikes = read_cached_spikes(bin_width=bin_width, threshold=threshold)
    if cached_spikes is None:
        cached_spikes = {}

    # update spikes for the area..
    cached_spikes[area] = spikes
    with open(file_path, 'wb') as F:
        pickle.dump(cached_spikes, F)
    if verbose:
        logger.info(f"Spikes for area: '{area}' saved to file: {file_path}")


def read_cached_spikes_session_wise(bin_width=20, delay=0):
    """Retrieves neural spikes for the given bin_width (sampling rate)
    and delay, returns a dictionary with session IDs as keys.

    Args:
        bin_width: int = bin_width in ms
        delay: int = neural delay in ms
    """
    file_name = f"neural_spikes_session_wise_bw_{bin_width}ms_delay_{delay}ms.pkl"
    file_path = os.path.join(cache_dir, 'neural', file_name)

    if os.path.exists(file_path):
        with open(file_path, 'rb') as F:
            spikes = pickle.load(F)
        return spikes   
    return None

def write_cached_spikes_session_wise(spikes, session, bin_width=20, 
            delay=0, verbose=True):
    """Saves neural spikes for the bin_width, delay and session specified,

    Args:
        spikes:  dict= spikes for all channels of the 'session'
        session: int = session ID 
        bin_width: int = bin_width in ms
        delay: int = neural delay in ms
    """
    session = str(int(session))
    file_name = f"neural_spikes_session_wise_bw_{bin_width}ms_delay_{delay}ms.pkl"
    file_path = os.path.join(cache_dir, 'neural', file_name)
    # make sure directory structure is in place...
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))

    cached_spikes = read_cached_spikes_session_wise(bin_width=bin_width)
    if cached_spikes is None:
        cached_spikes = {}

    # update spikes for the area..
    cached_spikes[session] = spikes
    with open(file_path, 'wb') as F:
        pickle.dump(cached_spikes, F)
    if verbose:
        logger.info(f"Spikes for session: '{session}' saved to file: {file_path}")



def read_cached_RDM_correlations(model_name, identifier, area, bin_width):
    """Retrieves cached RDM correlations (layer-wise) from the cache_dir,
    returns None if not cached already. 

    Args:
        model_name: str specifying model name, possible choices are
            ['wav2letter_modified', 'wav2vec2', 'speech2text',
            'deepspeech2', 'whiper_tiny', 'whisper_base']
        identifier: str specifying time alignment operation..
            ['', 'global','average']
    """
    assert identifier in ['', 'global','average'], logger.info(f"Please specify right identifier..")
    file_name = f"RDM_correlations_{model_name}_{identifier}_{area}_{bin_width}ms.pkl"
    file_path = os.path.join(cache_dir, model_name, file_name)
    if os.path.exists(file_path):
        with open(file_path, 'rb') as F:
            features = pickle.load(F)
        return features
    else:
        return None

def write_cached_RDM_correlations(corr_dict, model_name, identifier, area, bin_width):
    """Save RDM correlations (layer-wise) to the cache_dir,
    returns None if not cached already. 

    Args:
        model_name: str specifying model name, possible choices are
            ['wav2letter_modified', 'wav2vec2', 'speech2text',
            'deepspeech2', 'whiper_tiny', 'whisper_base']
    """
    model_choices = ['wav2letter_modified', 'wav2vec2', 'speech2text',
            'deepspeech2', 'whisper_tiny', 'whisper_base']
    assert model_name in model_choices, f"Invalid model name '{model_name}' specified!"
    assert identifier in ['', 'global','average'], logger.info(f"Please specify right identifier..")
    file_name = f"RDM_correlations_{model_name}_{identifier}_{area}_{bin_width}ms.pkl"
    file_path = os.path.join(cache_dir, model_name, file_name)

    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))

    # update spikes for the area..
    with open(file_path, 'wb') as F:
        pickle.dump(corr_dict, F)
    logger.info(f"RSA corr_dict saved to file: {file_path}")

    if os.path.exists(file_path):
        with open(file_path, 'rb') as F:
            features = pickle.load(F)
        return features
    else:
        return None
    

def read_normalizer_threshold(bin_width=20, poisson_normalizer=True):
    """Retrieves cached normalizer thresholds from the cache_dir,
    at specified bin_width.

    Args:
        bin_width: int = bin_width in ms
        poisson_normalizer: bool = specified method used for computing 
            normalizer threshold.
    """
    if poisson_normalizer:
        method = 'poisson'
    else:
        method = 'gaussian'

    file_name = f"normalizer_thresholds_{method}.pkl"
    file_path = os.path.join(cache_dir, 'normalizer', file_name)
    
    if os.path.exists(file_path):
        
        with open(file_path, 'rb') as F:
            normalizers_dict = pickle.load(F)
            logger.info(f"Reading exisiting normalizer thresholds...")
        return normalizers_dict
    return None

def write_normalizer_threshold(
        bin_width, poisson_normalizer, thresholds
    ):
    """Writes  normalizer thresholds to the cache_dir,
    at specified bin_width.

    Args:
        bin_width: int = bin_width in ms
        poisson_normalizer: bool = specified method used for computing 
            normalizer threshold.
        thresholds: ndarray =  result to be cached 
    """
    if poisson_normalizer:
        method = 'poisson'
    else:
        method = 'gaussian'

    file_name = f"normalizer_thresholds_{method}.pkl"
    file_path = os.path.join(cache_dir, 'normalizer', file_name)

    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))

    # read existing...
    existing_threshold = read_normalizer_threshold(
        bin_width=bin_width, poisson_normalizer=poisson_normalizer
        )
    if existing_threshold is None:
        existing_threshold = {}
    existing_threshold[bin_width] = thresholds

    logger.info(f"Writing normalizers to the cache...")
    # writing back..
    with open(file_path, 'wb') as F:
        pickle.dump(existing_threshold, F)



def read_context_dependent_normalizer(model_name, bin_width=20):
    """Reads cached context dependent variance of the correlations normalizer.
    Computed using contextful features of ANN, and using the features 
    corresponding to the repeated senteces.

    Args:
        model_name: str: model used for computing context variance
        bin_width: int = bin_width in ms
    """

    file_name = f"context_dependent_variance_{model_name}_{bin_width}.pkl"
    file_path = os.path.join(cache_dir, 'normalizer', file_name)
    
    if os.path.exists(file_path):
        
        with open(file_path, 'rb') as F:
            normalizers_dict = pickle.load(F)
            logger.info(f"Reading exisiting context normalizer ...")
        return normalizers_dict
    return None



def write_context_dependent_normalizer(
        model_name, context_normalizer, bin_width=20
    ):
    """Writes context dependent variance of the correlations normalizer.
    Computed using contextful features of ANN, and using the features 
    corresponding to the repeated senteces.

    Args:
        model_name: str: model used for computing context variance
        bin_width: int = bin_width in ms
        context_normalizer: dict = normalizers for different layers (keys)
    """

    file_name = f"context_dependent_variance_{model_name}_{bin_width}.pkl"
    file_path = os.path.join(cache_dir, 'normalizer', file_name)

    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))

    # read existing...
    existing_context_normalizers = read_context_dependent_normalizer(
        model_name=model_name, bin_width=bin_width
        )
    if existing_context_normalizers is None:
        existing_context_normalizers = {}
    for layer, normalizer in context_normalizer.items():
        existing_context_normalizers[layer] = normalizer

    logger.info(f"Writing context normalizers to the cache for {model_name} at {bin_width} ms...")
    # writing back..
    with open(file_path, 'wb') as F:
        pickle.dump(existing_context_normalizers, F)

    logger.info(f"Done.")


