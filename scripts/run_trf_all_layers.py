"""
This script computes and saves regression
results for layers of DNN models and neural areas.
It uses the TRF class to compute the regression
and saves the results in a CSV file.
It also allows for the option to save the
parameters of the regression model.

Args:
    dataset_name: str ['ucsf', 'ucdavis'], -d
    model_name: str, -m
    layers: list of int, default=None, -l
    bin_widths: list of int, -b
    identifier: str, default='', -i
    mVocs: bool, default=False, -v
    shuffled: bool, default=False, -s
    test_trial: int, default=None, -t
    LPF: bool, default=False, -L
    start_ind: int, default=0, --start
    end_ind: int, default=41, --end
    save_param: bool, default=False, --save_param


Example usage:
    python run_trf_all_layers.py -d ucsf -m whisper_tiny -b 50 -l 0 -i plos_test -v -s 
"""
# ------------------  set up logging ----------------------
import logging
from auditory_cortex.utils import set_up_logging
set_up_logging()

import os
import pandas as pd
import numpy as np
import time
import argparse
import gc

# local
from auditory_cortex import saved_corr_dir
# from auditory_cortex import config
import auditory_cortex.utils as utils
# import auditory_cortex.models as models
import auditory_cortex.io_utils.io as io
from auditory_cortex.io_utils.io import write_lmbdas
from auditory_cortex import valid_model_names
# from auditory_cortex.neural_data import NeuralMetaData




from auditory_cortex.neural_data import create_neural_dataset, create_neural_metadata
from auditory_cortex.dnn_feature_extractor import create_feature_extractor
from auditory_cortex.data_assembler import STRFDataAssembler, DNNDataAssembler, DNNAllLayerAssembler
from auditory_cortex.encoding import TRF



def compute_and_save_regression(args):

    START = time.time()
    # bin_widths = config['bin_widths']
    bin_widths = args.bin_widths
    dataset_name = args.dataset_name
    model_name = args.model_name
    layer_ids = args.layer_ids
    shuffled = args.shuffled
    test_trial = args.test_trial
    identifier = args.identifier
    mVocs = args.mVocs
    LPF = args.LPF
    save_param = args.save_param
    # fixed parameters..
    
    tmin=0
    delay=0
    N_sents=500
    num_workers=16
    num_folds=3
    LPF_analysis_bw = 20   
    
    lags=[args.lag]
    
    full_id = utils.get_run_id(
        dataset_name, bin_widths[0], identifier, mVocs=mVocs, shuffled=shuffled, lag=lags[0],
    )
    csv_file_name = model_name+'_'+full_id+'_'+'corr_results.csv'

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
    # metadata = NeuralMetaData()
    sessions = metadata.get_all_available_sessions()
    # ################################################################
    # # list of significant sessions only...
    # sig_sessions = np.array([180613., 180627., 180719., 180720., 180728., 180730., 180731.,
    # 					180807., 180808., 180814., 190606., 191113., 191121., 191125.,
    # 					191206., 191210., 200205., 200206., 200207., 200213., 200219.])

    # sessions = sig_sessions.astype(int)
    ###############################################################
    sessions = np.sort(sessions)
    sessions = sessions[args.start_ind:args.end_ind]
    # sessions = sessions[:20]
    # sessions = sessions[20:]
    current_time = time.time()
    elapsed_time = current_time - START

    for bin_width in bin_widths:
        # sessions = np.array(['200206'])
        # Session in data_dir that we do not have results for...
        if file_exists:
            sessions_done = data[
                    (data['delay']==delay) & \
                    (data['bin_width']==bin_width) 
                ]['session'].unique()

            subjects = sessions[np.isin(sessions,sessions_done.astype(int).astype(str), invert=True)]
        else:
            subjects = sessions

        for session in subjects:
            if mVocs:
                excluded_sessions = ['190726', '200213']
                if session in excluded_sessions:
                    print(f"Excluding session: {session}")
                    continue
            logging.info(f"Working with '{session}'")

            dataset_obj = create_neural_dataset(dataset_name, session)

            # dataset = DNNDataAssembler(
            #     dataset_obj, feature_extractor, layer_ID, bin_width=bin_width, mVocs=mVocs,
            #     LPF=LPF, LPF_analysis_bw=LPF_analysis_bw
            #     )
            dataset = DNNAllLayerAssembler(
            dataset_obj, feature_extractor, layer_ids=layer_ids, bin_width=bin_width, mVocs=mVocs,
            )
            

            trf_obj = TRF(model_name, dataset)
            
            corr, opt_lag, opt_lmbda, trf_model = trf_obj.grid_search_CV(
                    lags=lags, tmin=tmin,
                    num_folds=num_folds,
                    test_trial=test_trial
                )
            weights, biases = trf_model.coef_
            # if save_param:
            #     io.write_trf_parameters(
            #         model_name, session, weights, bin_width=bin_width, 
            #         shuffled=shuffled, layer_ID=layer_ID, LPF=LPF, mVocs=mVocs
            #         )
            #     io.write_trf_parameters(
            #         model_name, session, biases, bin_width=bin_width, 
            #         shuffled=shuffled, layer_ID=layer_ID, LPF=LPF, mVocs=mVocs,
            #         bias=True
            #         )
                
            if mVocs:
                mVocs_corr = corr
                timit_corr = np.zeros_like(corr)
            else:
                mVocs_corr = np.zeros_like(corr)
                timit_corr = corr


            corr_dict = {
                'test_cc_raw': timit_corr[None,...],
                'mVocs_test_cc_raw': mVocs_corr[None,...],
                'win': bin_width,
                'delay': delay, 
                'session': session,
                'model': model_name,
                'N_sents': N_sents,
                'layer_ids': [0],
                'opt_lag': opt_lag,
                'opt_lmbda': np.log10(opt_lmbda),
                'poiss_entropy': np.zeros_like(corr[None,...]),
                'uncertainty_per_spike': np.zeros_like(corr[None,...]),
                'bits_per_spike_NLB': np.zeros_like(corr[None,...]),
                }

            df = utils.write_to_disk(corr_dict, file_path, normalizer=None)

            # make sure to delete the objects to free up memory
            del dataset
            del trf_obj
            del weights
            del biases
            gc.collect()

    END = time.time()
    logging.info(f"Took {(END-START)/60:.2f} min., for bin_widths: '{bin_widths}'.")



# ------------------  get parser ----------------------#

def get_parser():
    # create an instance of argument parser
    parser = argparse.ArgumentParser(
        description='This is to compute and save regression results for for layers '+
            'of DNN models and neural areas',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

    # add arguments to read from command line
    parser.add_argument(
        '-m', '--model_name', dest='model_name', action='store',
        choices=valid_model_names,
        required=True,
        help='model to be used for Regression analysis.'
    )
    parser.add_argument(
        '-d','--dataset_name', dest='dataset_name', type= str, action='store',
        choices=['ucsf', 'ucdavis'],
        help = "Name of neural data to be used."
    )
    parser.add_argument(
        '--lag', dest='lag', type=int, action='store', 
        default=300,
        help="Specify the maximum lag used for STRF."
    )
    parser.add_argument(
        '-b','--bin_widths', dest='bin_widths', nargs='+', type= int, action='store',
        required=True,
        help="Specify list of bin_widths."
    )
    parser.add_argument(
        '-l','--layers', dest='layer_ids', nargs='+', type= int, action='store',
        default=None,
        help="Specify list of layer_ids."
    )
    parser.add_argument(
        '-i','--identifier', dest='identifier', type= str, action='store',
        default='',
        help="Specify identifier for saved results."
    )
    parser.add_argument(
        '-s','--shuffle', dest='shuffled', action='store_true', default=False,
        help="Specify if shuffled network to be used."
    )
    parser.add_argument(
        '-v','--mVocs', dest='mVocs', action='store_true', default=False,
        help="Specify if spikes for mVocs are to be used."
    )
    parser.add_argument(
        '-L','--LPF', dest='LPF', action='store_true', default=False,
        help="Specify if features are to be low pass filtered."
    )
    parser.add_argument(
        '-t','--test_trial', dest='test_trial', type= int, action='store',
        default=None,
        help="trial to test on."
    )
    parser.add_argument(
        '--start', dest='start_ind', type=int, action='store', 
        default=0,
        help="Choose sessions starting index to compute results at."
    )
    parser.add_argument(
        '--end', dest='end_ind', type=int, action='store', 
        default=41,
        # choices=[],
        help="Choose sessions ending index to compute results at."
    )
    parser.add_argument(
        '--save_param', dest='save_param', action='store_true', default=False,
        help="Specify if parameters to be saved."
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