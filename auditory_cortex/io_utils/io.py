import os
import pickle
from auditory_cortex import opt_inputs_dir, results_dir, cache_dir


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
    filepath = os.path.join(dirpath, f"{model_name}_beta_bank.pkl")
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            beta_bank = pickle.load(f)
            print("Loading file...")
    else:
        print(f"Creating new beta bank")
        beta_bank = {}

    beta_bank[session] = coefficents

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


def read_cached_features(model_name, contextualized=False):
    """Retrieves cached features from the cache_dir, returns None if 
    features not cached already. 

    Args:
        model_name: str specifying model name, possible choices are
            ['wave2letter_modified', 'wave2vec2', 'speech2text',
            'deepspeech2', 'whiper_tiny', 'whisper_base', 'whisper_small']
    """
    model_choices = ['wave2letter_modified', 'wave2vec2', 'speech2text',
            'deepspeech2', 'whisper_tiny', 'whisper_base', 'whisper_small']
    assert model_name in model_choices, f"Invalid model name '{model_name}' specified!"
    
    if contextualized:
        print(f"Reading contextualized features...")
        file_name = f"{model_name}_raw_features_contextualized.pkl"
    else:
        file_name = f"{model_name}_raw_features.pkl"
        
    file_path = os.path.join(cache_dir, model_name, file_name)
    if os.path.exists(file_path):
        with open(file_path, 'rb') as F:
            features = pickle.load(F)
        return features
    else:
        return None

def write_cached_features(model_name, features, verbose=True, contextualized=False):
    """Writes features to the cache_dir,

    Args:
        model_name: str specifying model name, possible choices are
            ['wave2letter_modified', 'wave2vec2', 'speech2text',
            'deepspeech2', 'whiper_tiny', 'whisper_base', 'whisper_small']
        features: list = features for each layers as a list of dictionaries 
    """
    model_choices = ['wave2letter_modified', 'wave2vec2', 'speech2text',
            'deepspeech2', 'whisper_tiny', 'whisper_base', 'whisper_small']
    assert model_name in model_choices, f"Invalid model name '{model_name}' specified!"
    
    if contextualized:
        print(f"writing contextualized features...")
        file_name = f"{model_name}_raw_features_contextualized.pkl"
    else:
        file_name = f"{model_name}_raw_features.pkl"
    # file_name = f"{model_name}_raw_features.pkl"
    file_path = os.path.join(cache_dir, model_name, file_name)
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
            ['wave2letter_modified', 'wave2vec2', 'speech2text',
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
            ['wave2letter_modified', 'wave2vec2', 'speech2text',
            'deepspeech2', 'whiper_tiny', 'whisper_base']
    """
    model_choices = ['wave2letter_modified', 'wave2vec2', 'speech2text',
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


