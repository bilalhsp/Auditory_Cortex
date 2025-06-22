# adjsut the basic logging lovel of notebook
import os
import logging
logging.basicConfig(level=logging.WARNING)

import numpy as np
from auditory_cortex import utils, saved_corr_dir, config
from auditory_cortex.deprecated.models import Regression
import pandas as pd

# using the following models (best layers for each) for ensemble results.
models_list = [
    {
        'model_name': 'wav2letter_modified',
        'layer': 6,
        'opt_neural_delay': 37.75,
    },
    {
        'model_name': 'speech2text',
        'layer': 4,
        'opt_neural_delay': 49.83,
    },
    # {
    #     'model_name': 'wav2vec',
    #     'layer': 8
    # },
    {
        'model_name': 'wav2vec2',
        'layer': 7,
        'opt_neural_delay': 49.23,
    },
    {
        'model_name': 'deepspeech2',
        'layer': 2,
        'opt_neural_delay': 54.73,
    },
]


def get_sessions_list():
    ## read the sessions available in data_dir
    data_dir = config['neural_data_dir']
    bad_sessions = config['bad_sessions']
    sessions = np.array(os.listdir(data_dir))
    sessions = np.delete(sessions, np.where(sessions == "out_sentence_details_timit_all_loudness.mat"))
    for s in bad_sessions:
        sessions = np.delete(sessions, np.where(sessions == s))
    sessions = np.sort(sessions)
    return sessions

def get_remaining_sessions(sessions, corr_filepath, bin_width=20, delay=0):
    if os.path.exists(corr_filepath):
        data = pd.read_csv(corr_filepath)
        sessions_done = data[
                    (data['delay']==delay) & \
                    (data['bin_width']==bin_width) 
                ]['session'].unique()
        sessions = sessions[
            np.isin(sessions,sessions_done.astype(int).astype(str), invert=True)
            ]
    return sessions


def create_regression_models(models_list):
    # creating regression objects for all networks...
    reg_objs = []
    for i, model in enumerate(models_list):
        model_name = model['model_name']
        print(f"Creating regression object for {model_name}")
        reg_objs.append(Regression(model_name, load_features=True))
    return reg_objs

def ensemble_corr_for_session(reg_objs, models_list, session, delay=0):
    """Computes correlations for predicitons using the ensemble of 
    networks (using best layer for each), as specified in 'models_list'."""

    session = str(session)
    print(f"Computing ensemble correlations for sess-{session}")
    
    correlations = []
    predictions_list = []
    # compute Beta
    for i, model in enumerate(models_list):
        if i==0:
            test_set = None
        else:
            # copying dataset object, so we dont have to load again for every model.
            reg_objs[i].spike_datasets[session] = reg_objs[0].spike_datasets[session]
            
        corr_coeff, B, loss, test_set = reg_objs[i].cross_validated_regression(
            session=session, num_lmbdas=10, iterations=1, test_sents=test_set,
            delay=delay
            )
        
        if i==0:
            spikes = reg_objs[i].get_neural_spikes(session, sents=test_set, numpy=True,
                                                   delay=delay)
        # get predictions from individual models..
        print(f"Computing neural predictions using {reg_objs[i].model_name}")
        predictions = reg_objs[i].neural_prediction(session, sent=test_set)
        layer = model['layer']
        layer_idx = reg_objs[i].get_layer_index(layer)
        
        predictions_list.append(predictions[:,:,layer_idx])
        correlations.append(utils.cc_norm(spikes, predictions[:, :, layer_idx]))

    ensemble_prediction = sum(predictions_list)/len(predictions_list)
    ensemble_corr = utils.cc_norm(spikes, ensemble_prediction)

    return ensemble_corr

def save_corr(corr_results, file_path, norm, session, bin_width=20, delay=0):
    # although there is no meaningful layer id here, adding 
    # an extra dimension for the layer, just for consistency.
    if corr_results.ndim == 1:
        corr_results = np.expand_dims(corr_results, axis=0)
    corr_dict = { 
        'test_cc_raw': corr_results,
        'train_cc_raw': corr_results, # not interested in these
        'win': bin_width,
        'delay': delay, 
        'session': session,
        'model': 'ensembe4',
        'N_sents': 499,
        'layer_ids': [0],       # no valid layer_id for ensemble
        'opt_delays': None
        }
    
    df = utils.write_to_disk(corr_dict, file_path, normalizer=norm)
    


def save_ensemble_correlations(models_list, bin_width=20, delay=0, identifier = ''):
    """Uses ensemble of models provided in the list, and
    computes correlations using the best layer of each model (
    also specified in the list)."""

    sessions = get_sessions_list()
    reg_objs = create_regression_models(models_list)

    # correlation results file...
    if identifier != '':
        identifier += '_'

    corr_filepath = os.path.join(saved_corr_dir, f'ensemble_{identifier}corr_results.csv')
    sessions = get_remaining_sessions(
        sessions, corr_filepath, bin_width=bin_width, delay=delay
        )

    for session in sessions:
        session = str(session)

        # using 1 model to compute normalers, these will be replaced with 
        # pre-saved results, with reduced variance, afterwards...!
        norm = reg_objs[0].get_normalizer(
            session, bin_width=bin_width,
            delay=0, n=1 # normalizer not needed, will be updated later
            )
            
        ensemble_corr = ensemble_corr_for_session(reg_objs, models_list, session)

        save_corr(ensemble_corr, corr_filepath, norm, session, bin_width=bin_width, delay=delay)
    print(f"Done for {len(sessions)} sessions.")

def avg_opt_delay(models_list):
    """Computes avg. optimal delay for for all 4 networks, 
    rounds-off to nearest integer."""
    opt_delays = []
    for model in models_list:
        opt_delays.append(model['opt_neural_delay'])
    avg_delay = sum(opt_delays)/len(opt_delays)
    return int(avg_delay + 0.5) 

if __name__ == "__main__":
    
    print(f"Using ensembe of {len(models_list)} models, to predict neural activity...")
    identifier = 'opt_neural_delay'
    delay = avg_opt_delay(models_list)
    print(f"Avg. optimal delay for {len(models_list)} models: {delay} ms.")
    save_ensemble_correlations(models_list, bin_width=20, delay=delay, identifier=identifier)

