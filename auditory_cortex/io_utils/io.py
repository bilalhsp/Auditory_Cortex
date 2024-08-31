import os
import numpy as np
import pandas as pd
import pickle
from auditory_cortex import opt_inputs_dir, results_dir, cache_dir, normalizers_dir
from auditory_cortex import valid_model_names



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
    print(f"Reading sig sessions/channels from: {file_path}")
    if os.path.exists(file_path):
        with open(file_path, 'rb') as F: 
            significant_sessions_and_channels = pickle.load(F)
        return significant_sessions_and_channels
    else:
        print(f"Sigificant sessions/channels data not found: for bin-width {bin_width}ms.")
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
        print(f"Path not found, creating directories...")
        os.makedirs(path_dir)

    file_path = os.path.join(path_dir, f"significant_sessions_and_channels_bw_{bin_width}ms_pvalue_{p_threshold}.pkl")
    with open(file_path, 'wb') as F: 
        pickle.dump(significant_sessions_and_channels, F)
    print(f"Sigificant sessions/channels saved to: {file_path}")


#-----------      Null distribution using poisson sequences    -----------#

def read_normalizer_null_distribution_using_poisson(bin_width, spike_rate, mVocs=False):
    """Retrieves null distribution of correlations computed using poisson sequences."""
    bin_width = int(bin_width)
    # path_dir = os.path.join(results_dir, 'normalizers', 'null_distribution')
    if mVocs:
        parent_dir = os.path.join(normalizers_dir, 'mVocs')
        post_str = ' (mVocs)'
    else:
        parent_dir = normalizers_dir
        post_str = ''
    path_dir = os.path.join(parent_dir, 'null_distribution')
    file_path = os.path.join(path_dir, f"normalizers_null_dist_poisson_bw_{bin_width}ms_spike_rate_{spike_rate}hz.pkl")
    if os.path.exists(file_path):
        with open(file_path, 'rb') as F: 
            norm_null_dist = pickle.load(F)
        return norm_null_dist
    else:
        print(f"Null dist.{post_str} not found: for bin-width {bin_width}ms and {spike_rate}Hz spike rate.")
        return None

def write_normalizer_null_distribution_using_poisson(bin_width, spike_rate, null_dist_poisson, mVocs=False):
    """Writes null distribution of correlations computed using poisson sequences for the given selection."""
    bin_width = int(bin_width)
    # path_dir = os.path.join(results_dir, 'normalizers', 'null_distribution')
    if mVocs:
        parent_dir = os.path.join(normalizers_dir, 'mVocs')
    else:
        parent_dir = normalizers_dir
    path_dir = os.path.join(parent_dir, 'null_distribution')
    if not os.path.exists(path_dir):
        print(f"Path not found, creating directories...")
        os.makedirs(path_dir)
    file_path = os.path.join(path_dir, f"normalizers_null_dist_poisson_bw_{bin_width}ms_spike_rate_{spike_rate}hz.pkl")
    
    with open(file_path, 'wb') as F: 
        pickle.dump(null_dist_poisson, F)
    print(f"Null dist. poisson saved to: {file_path}")


#-----------      Null distribution using sequence shifts    -----------#

def read_normalizer_null_distribution_random_shifts(
        bin_width, min_shift_frac, max_shift_frac
        ):
    """Retrieves null distribution of correlations computed using randomly 
        shifted spike sequence of one trial vs (non-shifted) seconds trial."""
    bin_width = int(bin_width)
    # path_dir = os.path.join(results_dir, 'normalizers', 'null_distribution', 'shifted_sequence')
    path_dir = os.path.join(normalizers_dir, 'null_distribution', 'shifted_sequence')
    file_path = os.path.join(
        path_dir,
        f"normalizers_null_dist_sequence_shifted_bw_{bin_width}ms_shift_range_{min_shift_frac:01.2f}_{max_shift_frac:01.2f}.pkl"
        )
    if os.path.exists(file_path):
        with open(file_path, 'rb') as F: 
            norm_null_dist = pickle.load(F)
        return norm_null_dist
    else:
        print(f"Null dist. not found: for bin-width {bin_width}ms and shift range {min_shift_frac:01.2f}--{max_shift_frac:01.2f}.")
        return None

def write_normalizer_null_distribution_using_random_shifts(
        session, bin_width, min_shift_frac,
        max_shift_frac, null_dist_sess,
        ):
    """Writes null distribution of correlations computed using poisson sequences for the given selection."""
    bin_width = int(bin_width)
    session = str(int(float(session)))
    # path_dir = os.path.join(results_dir, 'normalizers', 'null_distribution', 'shifted_sequence')
    path_dir = os.path.join(normalizers_dir, 'null_distribution', 'shifted_sequence')
    file_path = os.path.join(
        path_dir,
        f"normalizers_null_dist_sequence_shifted_bw_{bin_width}ms_shift_range_{min_shift_frac:01.2f}_{max_shift_frac:01.2f}.pkl"
        )
    
    if not os.path.exists(path_dir):
        print(f"Path not found, creating directories...")
        os.makedirs(path_dir)

    null_dict_all_sessions = read_normalizer_null_distribution_random_shifts(
        bin_width, min_shift_frac, max_shift_frac
        )
        
    if null_dict_all_sessions is None:
        null_dict_all_sessions = {}

    null_dict_all_sessions[session] = null_dist_sess
    with open(file_path, 'wb') as F: 
        pickle.dump(null_dict_all_sessions, F)
    print(f"Writing normalizer dictionary to the {file_path}")



#-----------  Normalizer distribution using all possible pairs of trials  ----------#

def read_normalizer_distribution(
        bin_width, delay, method='app', mVocs=False
        ):
    """Retrieves distribution of normalizers for the given selection."""
    bin_width = int(bin_width)
    delay = int(delay)
    if method == 'app':
        subdir = 'all_possible_pairs'
    else:
        subdir = 'random_pairs'

    if mVocs:
        parent_dir = os.path.join(normalizers_dir, 'mVocs')
    else:
        parent_dir = normalizers_dir

    path_dir = os.path.join(parent_dir, subdir)
    file_path = os.path.join(path_dir, f"normalizers_bw_{bin_width}ms_delay_{delay}ms.pkl")
    if os.path.exists(file_path):
        with open(file_path, 'rb') as F: 
            normalizers_dist = pickle.load(F)
        return normalizers_dist
    else:
        print(f"Normalizers not found: for bin-width {bin_width}ms and delay {delay}ms.")
        return None

def write_normalizer_distribution(
        session, bin_width, delay, normalizer_dist, method='app', mVocs=False
        ):
    """Writes distribution of normalizers for the given selection."""
    bin_width = int(bin_width)
    delay = int(delay)
    if method == 'app':
        subdir = 'all_possible_pairs'
    else:
        subdir = 'random_pairs'
    # path_dir = os.path.join(results_dir, 'normalizer', subdir)
    if mVocs:
        parent_dir = os.path.join(normalizers_dir, 'mVocs')
    else:
        parent_dir = normalizers_dir

    path_dir = os.path.join(parent_dir, subdir)
    if not os.path.exists(path_dir):
        print(f"Path not found, creating directories...")
        os.makedirs(path_dir)
    file_path = os.path.join(path_dir, f"normalizers_bw_{bin_width}ms_delay_{delay}ms.pkl")
    norm_dict_all_sessions = read_normalizer_distribution(
        bin_width, delay, method=method, mVocs=mVocs)
    
    if norm_dict_all_sessions is None:
        norm_dict_all_sessions = {}

    norm_dict_all_sessions[session] = normalizer_dist
    with open(file_path, 'wb') as F: 
        pickle.dump(norm_dict_all_sessions, F)
    print(f"Writing normalizer dictionary to the {file_path}")



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
        print(f"Reading from file: {file_path}")
        with open(file_path, 'rb') as F: 
            reg_results = pickle.load(F)
        return reg_results
    else:
        print(f"Results not found.")
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
        print(f"Directory path created: {path_dir}")

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
    print(f"Results saved for {model_name} at path: \n {file_path}.")



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
        print(f"Reading from file: {file_path}")
        with open(file_path, 'rb') as F: 
            reg_results = pickle.load(F)
        return reg_results
    else:
        print(f"Results not found.")
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
            print(f"Directory path created: {path_dir}")
        
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
        print(f"Results saved for {model_name} at path: \n {file_path}.")


def read_WER():

    path_dir = os.path.join(results_dir, 'task_optimization')
    filename = f'pretrained_networks_WERs.csv'    
    file_path = os.path.join(path_dir, filename)
    if os.path.isfile(file_path):
        print("Reading existing WER results")
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
        print(f"Directory path created: {path_dir}")

    df = read_WER()
    if df is None:
        df = pd.DataFrame()
    if benchmark not in df.columns:
        df[benchmark] = np.nan
    if model_name not in df.index:
        df.loc[model_name] = pd.Series(np.nan)
    
    df.at[model_name, benchmark] = wer
    df.to_csv(file_path)
    print(f"WER for {model_name}, on {benchmark} saved to {file_path}")
    



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
        print(f"Reading from file: {file_path}")
        with open(file_path, 'rb') as F: 
            reg_results = pickle.load(F)
        return reg_results
    else:
        print(f"Results not found.")
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
        print(f"Directory path created: {path_dir}")

    filename = f'reg_correlations_normalized_20ms.pkl'    
    file_path = os.path.join(path_dir, filename)
    
    exisiting_results = read_reg_corr()
    if exisiting_results is None:
        exisiting_results = {}

    # save/update results for model_name
    exisiting_results[model_name] = results_dict
    with open(file_path, 'wb') as F: 
        pickle.dump(exisiting_results, F)
    print(f"Results saved for {model_name} at path: \n {file_path}.")


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
        print("Does not exist")
        os.makedirs(dirpath)

    # loading existing betas or creating new (if not available already)
    beta_bank = read_model_parameters(model_name=model_name)
    if beta_bank is None:
        print(f"Creating new beta bank")
        beta_bank = {}

    beta_bank[session] = coefficents

    filepath = os.path.join(dirpath, f"{model_name}_beta_bank.pkl")
    with open(filepath, 'wb') as f:
        pickle.dump(beta_bank, f) 

    print(f"Parameters computed and saved for {model_name}, sess-{session}")



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
        print(f"Reading from file: {file_path}")
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
        print(f"Directory path created: {path_dir}")
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
        print(f"RSA saved to '{file_path}'")

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
        print(f"Removed: {file_path}")
    else:
        print(f"File does not exist.")


def read_cached_features(model_name, contextualized=False, shuffled=False, mVocs=False):
    """Retrieves cached features from the cache_dir, returns None if 
    features not cached already. 

    Args:
        model_name: str specifying model name, possible choices are
            ['wav2letter_modified', 'wav2vec2', 'speech2text',
            'deepspeech2', 'whiper_tiny', 'whisper_base', 'whisper_small']
    """
    assert model_name in valid_model_names, f"Invalid model name '{model_name}' specified!"

    
    if contextualized:
        print(f"Reading contextualized features...")
        file_name = f"{model_name}_raw_features_contextualized.pkl"
    else:
        file_name = f"{model_name}_raw_features.pkl"
    
    if mVocs:
        directory = os.path.join(cache_dir, 'mVocs')
    else:
        directory = cache_dir

    if shuffled:
        file_path = os.path.join(directory, model_name, 'shuffled', file_name)
    else:
        file_path = os.path.join(directory, model_name, file_name)

    if os.path.exists(file_path):
        print(f"Reading raw features from {file_path}")
        with open(file_path, 'rb') as F:
            features = pickle.load(F)
        return features
    else:
        return None

def write_cached_features(
        model_name, features, verbose=True, contextualized=False, shuffled=False,
        mVocs=False):
    """Writes features to the cache_dir,

    Args:
        model_name: str specifying model name, possible choices are
            ['wav2letter_modified', 'wav2vec2', 'speech2text',
            'deepspeech2', 'whiper_tiny', 'whisper_base', 'whisper_small']
        features: list = features for each layers as a list of dictionaries 
    """
    # model_choices = ['wav2letter_modified', 'wav2vec2', 'speech2text',
    #         'deepspeech2', 'whisper_tiny', 'whisper_base', 'whisper_small',
    #         'wav2letter_spect']
    assert model_name in valid_model_names, f"Invalid model name '{model_name}' specified!"
    
    if contextualized:
        print(f"writing contextualized features...")
        file_name = f"{model_name}_raw_features_contextualized.pkl"
    else:
        file_name = f"{model_name}_raw_features.pkl"
    # file_name = f"{model_name}_raw_features.pkl"
    if mVocs:
        directory = os.path.join(cache_dir, 'mVocs')
    else:
        directory = cache_dir

    if shuffled:
        file_path = os.path.join(directory, model_name, 'shuffled', file_name)
    else:
        file_path = os.path.join(directory, model_name, file_name)
    # make sure directory structure is in place...
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    
    # writing features to file
    with open(file_path, 'wb') as F:
        pickle.dump(features, F)
    if verbose:
        print(f"Features saved to file: {file_path}")
    
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
        print(f"Spikes for area: '{area}' saved to file: {file_path}")


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
        print(f"Spikes for session: '{session}' saved to file: {file_path}")



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
    assert identifier in ['', 'global','average'], print(f"Please specify right identifier..")
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
    assert identifier in ['', 'global','average'], print(f"Please specify right identifier..")
    file_name = f"RDM_correlations_{model_name}_{identifier}_{area}_{bin_width}ms.pkl"
    file_path = os.path.join(cache_dir, model_name, file_name)

    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))

    # update spikes for the area..
    with open(file_path, 'wb') as F:
        pickle.dump(corr_dict, F)
    print(f"RSA corr_dict saved to file: {file_path}")

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
            print(f"Reading exisiting normalizer thresholds...")
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

    print(f"Writing normalizers to the cache...")
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
            print(f"Reading exisiting context normalizer ...")
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

    print(f"Writing context normalizers to the cache for {model_name} at {bin_width} ms...")
    # writing back..
    with open(file_path, 'wb') as F:
        pickle.dump(existing_context_normalizers, F)

    print(f"Done.")


