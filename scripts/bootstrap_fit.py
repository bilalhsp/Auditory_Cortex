"""
This script is used to run the bootstrap analysis to finalize the 
details of stimulus design for data collection. Specifically,
it is meant to run two types of analyses:
1. Bootstrapping on the training set, where the proportion of training data
    to be used to fit the regression is specified as N_sents. 
    N_sents of 10 means 10% of the total duration of the training set.
2. Bootstrapping on the test set, where the number of test trials to be used
    is specified as N_test_trials.
    - N_test_trials: # of trial repeats used to evaluate the model.
        e.g. 1 means 1 trial repeat, 2 means 2 trial repeats, etc.
    - test_duration: fraction of duration of test set to be used for evaluation.
        e.g. 20 means 20% of the total duration of the test set.

Args:
    dataset_name: str ['ucsf', 'ucdavis'], -d
    model_name: str, -m
    layer_ID: int, -l
    bin_widths: list of int, -b
    identifier: str, default='', -i
    mVocs: bool, default=False, -v
    start_ind: int, default=0, --start
    end_ind: int, default=41, --end
    test_bootstrap: bool, default=False, --test_bootstrap
    N_test_trials: int, default=None, --N_test_trials
    percent_duration: int, default=None, --percent_duration
        duration of train/test stimuli to be used for training/evaluation.
        if test_bootstrap is True, this is the percent of the test set duration to be used.
        otherwise used for percent of training set duration to be used.


Usage examples:
- Training set bootstrap:
    python bootstrap_fit.py -d ucsf -m whisper_base -l 2 -b 50 -v --percent_duration 20 -i exp_design
"""


import logging
from auditory_cortex.utils import set_up_logging
set_up_logging()

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import argparse
import gc

# local
from auditory_cortex import saved_corr_dir
from auditory_cortex import config
import auditory_cortex.utils as utils
# import auditory_cortex.models as models
import auditory_cortex.io_utils.io as io
from auditory_cortex.io_utils.io import write_lmbdas
from auditory_cortex import valid_model_names
# from auditory_cortex.neural_data import NeuralMetaData
# from auditory_cortex.datasets import BaselineDataset, DNNDataset
# from auditory_cortex.computational_models.encoding import TRF

from auditory_cortex.io_utils import ResultsManager
from auditory_cortex.neural_data import create_neural_dataset, create_neural_metadata
from auditory_cortex.dnn_feature_extractor import create_feature_extractor
from auditory_cortex.data_assembler import STRFDataAssembler, DNNDataAssembler
from auditory_cortex.encoding import TRF


def compute_and_save_regression(args):

    # bin_widths = config['bin_widths']
    bin_widths = args.bin_widths
    model_name = args.model_name
    layer_ID = args.layer_ID
    # shuffled = args.shuffled
    # test_trial = args.test_trial
    identifier = args.identifier
    mVocs = args.mVocs
    # LPF = args.LPF  
    # N_sents = args.N_sents
    percent_duration = args.percent_duration
    test_bootstrap = args.test_bootstrap
    n_test_trials = args.n_test_trials
    dataset_name = args.dataset_name
    # itr = args.itr
    # fixed parameters..
    
    tmin=0
    delay=0
    # N_sents=500
    # num_workers=16
    num_folds=3
    LPF=False
    LPF_analysis_bw = 20
    shuffled = False
    # LPF_analysis_bw = 10
    
    lag=200
    # use_nonlinearity=False

    results_identifier = ResultsManager.get_run_id(
            dataset_name, bin_widths[0], identifier, mVocs=mVocs, shuffled=shuffled, lag=lag,
            bootstrap=True, test_bootstrap=test_bootstrap
        )
    csv_file_name = model_name + '_' + results_identifier + '_corr_results.csv'



    # CSV file to save the results at
    file_exists = False
    file_path = os.path.join(saved_corr_dir, csv_file_name)
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        file_exists = True

    if shuffled:
        logging.info(f"Running TRF for 'Untrained' networks...")
    else:
        logging.info(f"Running TRF for 'Trained' networks...")


    feature_extractor = create_feature_extractor(model_name, shuffled=shuffled)
    metadata = create_neural_metadata(dataset_name)


    if dataset_name == 'ucsf':
        sessions = np.array([
                '200205', '191121', '191210', # non-primary sessions 27/35 channels...
                '200206', '191113', '180814', '200213', '191206', '191125', '180731',
                '200207', '180807',      # primary sessions 195/227 channels...
                ])
    elif dataset_name == 'ucdavis':
        sessions = metadata.get_all_available_sessions()
    for bin_width in bin_widths:
        # Session in data_dir that we do not have results for...
        if file_exists:
            sessions_done = data[
                    (data['delay']==delay) & \
                    (data['bin_width']==bin_width) 
                ]['session'].unique()

            subjects = sessions[np.isin(sessions,sessions_done.astype(int).astype(str), invert=True)]
        else:
            subjects = sessions

        if len(subjects) == 0:
            logging.info(f"All sessions already done for bin_width: {bin_width}.")
            continue

        neural_dataset = create_neural_dataset(dataset_name)
        data_assembler = DNNDataAssembler(
                neural_dataset, feature_extractor, layer_ID, bin_width=bin_width, mVocs=mVocs,
                LPF=LPF, LPF_analysis_bw=LPF_analysis_bw
                )
        
        for session in subjects:
            logging.info(f"Working with '{session}'")
            
            if mVocs:
                excluded_sessions = ['190726', '200213']
                if session in excluded_sessions:
                    print(f"Excluding session: {session}")
                    continue
            
            if session != data_assembler.get_session_id():
                # no need to read features again...just reach spikes..
                dataset_obj = create_neural_dataset(dataset_name, session)
                data_assembler.read_session_spikes(dataset_obj)
            

            trf_obj = TRF(model_name, data_assembler)
            if test_bootstrap:
                trf_model = trf_obj.load_saved_model(
                    model_name, session, layer_ID, bin_width, shuffled=shuffled, dataset_name=dataset_name,
                    mVocs=mVocs, tmax=lag, LPF=LPF,
                    )
                if trf_model is None:
                    corr, opt_lmbda, trf_model = trf_obj.grid_search_CV(
                        lag=lag, tmin=tmin, num_folds=num_folds,
                    )
                    trf_obj.save_model_parameters(
                        trf_model, model_name, layer_ID, session, bin_width, shuffled=shuffled,
                        LPF=LPF, mVocs=mVocs, dataset_name=dataset_name, tmax=lag
                    )
                corr = trf_obj.evaluate(
                    trf_model, n_test_trials=n_test_trials, percent_duration=percent_duration
                    )
                # opt_lag = [0]*corr.size
                opt_lmbda = [1]*corr.size
            else:
                corr, opt_lmbda, trf_model = trf_obj.grid_search_CV(
                        lag=lag, tmin=tmin, num_folds=num_folds,
                        percent_duration=percent_duration,
                    )
            if mVocs:
                mVocs_corr = corr
                timit_corr = np.zeros_like(corr)
            else:
                mVocs_corr = np.zeros_like(corr)
                timit_corr = corr

            channel_ids = data_assembler.channel_ids
            num_channels = len(channel_ids)
            corr_dict = {
                'session': num_channels*[session],
                'layer': num_channels*[layer_ID],
                'channel': channel_ids,
                'bin_width': num_channels*[bin_width],
                'percent_duration': num_channels*[percent_duration],
                'test_cc_raw': timit_corr.squeeze(),
                'normalizer': num_channels*[0.0],  # placeholder for normalizer
                'mVocs_test_cc_raw': mVocs_corr.squeeze(),
                'mVocs_normalizer': num_channels*[0.0],  # placeholder for mVocs normalizer
                'opt_lag': num_channels*[lag],
                'opt_lmbda': np.log10(opt_lmbda).squeeze(),
                'n_test_trials': num_channels*[n_test_trials],
                }
            
            df = utils.write_to_disk(corr_dict, file_path)


            # corr_dict = {
            #     'test_cc_raw': timit_corr[None,...],
            #     'mVocs_test_cc_raw': mVocs_corr[None,...],
            #     'win': bin_width,
            #     'delay': delay, 
            #     'session': session,
            #     'model': model_name,
            #     'N_sents': N_sents,
            #     'layer_ids': [layer_ID],
            #     'opt_lag': opt_lag,
            #     'opt_lmbda': np.log10(opt_lmbda),
            #     'poiss_entropy': np.zeros_like(corr[None,...]),
            #     'uncertainty_per_spike': np.zeros_like(corr[None,...]),
            #     'bits_per_spike_NLB': np.zeros_like(corr[None,...]),
            #     }

            # df = utils.write_to_disk(corr_dict, file_path, normalizer=None)

            # make sure to delete the objects to free up memory
            # del dataset
            del trf_model
            del trf_obj
    
            gc.collect()




# ------------------  get parser ----------------------#

def get_parser():

    parser = argparse.ArgumentParser(
        description='This is to compute and save regression results for for layers '+
            'of DNN models and neural areas for the purpose of doing bootstrap analysis.'+
            'This script is useful for two types of bootstrapping analysis: '+
            '1. Bootstrapping on the training set, and '+
            '2. Bootstrapping on the test set. '+
            'For the first, specify the proportion of training data to be used to '+
            'fit the betas as N_sents. For the second, specify the number of test trials to be used.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

    parser.add_argument(
        '-m', '--model_name', dest='model_name', action='store',
        choices=valid_model_names,
        required=True,
        help='model to be used for Regression analysis.'
    )
    parser.add_argument(
        '-d','--dataset_name', dest='dataset_name', type= str, action='store',
        choices=['ucsf', 'ucdavis'], required=True,
        help = "Name of neural data to be used."
    )

    parser.add_argument(
        '-b','--bin_widths', dest='bin_widths', nargs='+', type= int, action='store',
        default=[50],
        help="Specify list of bin_widths."
    )
    parser.add_argument(
        '-l','--layer', dest='layer_ID', type=int, action='store',
        required=True,
        help="Specify the layer ID."
    )
    parser.add_argument(
        '-i','--identifier', dest='identifier', type= str, action='store',
        default='',
        help="Specify identifier for saved results."
    )
    parser.add_argument(
        '-v','--mVocs', dest='mVocs', action='store_true', default=False,
        help="Specify if spikes for mVocs are to be used."
    )
    parser.add_argument(
        '--start', dest='start_ind', type=int, action='store', 
        default=0,
        help="Choose sessions starting index to compute results at."
    )
    parser.add_argument(
        '--end', dest='end_ind', type=int, action='store', 
        default=41,
        help="Choose sessions ending index to compute results at."
    )
    parser.add_argument(
        '--test_bootstrap', dest='test_bootstrap', action='store_true', default=False,
        help="Specify if bootstrap on the test set is required."
    )
    parser.add_argument(
        '--n_test_trials', dest='n_test_trials', type=int, action='store', default=None,
        choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        help="Specify the number of test trials to be used."
    )
    parser.add_argument(
        '--percent_duration', dest='percent_duration', type=int, action='store', default=None,
        choices=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        help="Specify the \%\ of total duration of train/test stimuli to be used for training/evaluation."
    )
    return parser




# ------------------  main function ----------------------#

if __name__ == '__main__':

    start_time = time.time()
    parser = get_parser()
    args = parser.parse_args()

    # display the arguments passed
    for arg in vars(args):
        logging.info(f"{arg:15} : {getattr(args, arg)}")

    compute_and_save_regression(args)
    elapsed_time = time.time() - start_time
    logging.info(f"It took {elapsed_time/60:.1f} min. to run.")