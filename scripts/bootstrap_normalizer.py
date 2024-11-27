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
import auditory_cortex.models as models
import auditory_cortex.io_utils.io as io
from auditory_cortex.io_utils.io import write_lmbdas
from auditory_cortex import valid_model_names
from auditory_cortex.neural_data import NeuralMetaData
from auditory_cortex.datasets import BaselineDataset, DNNDataset
from auditory_cortex.computational_models.encoding import TRF
from auditory_cortex.neural_data.normalizer import Normalizer

from auditory_cortex.neural_data.normalizer_calculator import NormalizerCalculator


def save_normalizer_bootstrap_dist(args):

    # bin_widths = config['bin_widths']
    session_index = args.session_index
    n = args.num_samples
    dataset_name = args.dataset_name
    mVocs = False
    bin_width = 50
    percent_durations = [11, 22, 33, 44, 55, 66, 77, 88, 100]
    iterations = np.arange(1, 81)

    if dataset_name == 'ucsf':
        sessions = np.array([
            '200205', '191121', '191210', # non-primary sessions 27/35 channels...
            '200206', '191113', '180814', '200213', '191206', '191125', '180731',
            '200207', '180807',      # primary sessions 195/227 channels...
            ])
        # norm_obj = Normalizer()
        # sessions = norm_obj.metadata.get_all_available_sessions()
        # sessions = np.sort(sessions)
        sess_id = sessions[session_index]
    else:
        sess_id = session_index
    

    norm_obj = NormalizerCalculator(dataset_name, sess_id)
    
    norm_obj.save_bootstrapped_normalizer(
        percent_durations, iterations=iterations, bin_width=bin_width,
        n=n, mVocs=mVocs
        )




# ------------------  get parser ----------------------#

def get_parser():
    # create an instance of argument parser
    parser = argparse.ArgumentParser(
        description='This is to compute and save regression results for for layers '+
            'of DNN models and neural areas',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument(
		'-d','--dataset_name', dest='dataset_name', type= str, action='store',
		choices=['ucsf', 'ucdavis'],
		help = "Name of neural data to be used."
	)
    parser.add_argument(
        '-s', '--session', dest='session_index', type=int, action='store', 
        default=0,
        help="Choose sessions index."
    )
    parser.add_argument(
        '-n', '--num_samples', dest='num_samples', type=int, action='store', 
        default=10000,
        help="Choose number of samples."
    )

    return parser




# ------------------  main function ----------------------#

if __name__ == '__main__':

    start_time = time.time()
    parser = get_parser()
    args = parser.parse_args()

    # display the arguments passed
    for arg in vars(args):
        print(f"{arg:15} : {getattr(args, arg)}")

    save_normalizer_bootstrap_dist(args)
    elapsed_time = time.time() - start_time
    print(f"It took {elapsed_time/60:.1f} min. to run.")