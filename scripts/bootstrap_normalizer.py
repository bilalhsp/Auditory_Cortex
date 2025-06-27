"""
Computes and saves bootstrapped distributions of normalizers
(both True & Null distributions) for neural datasets.

Args:
    dataset_name: str ['ucsf', 'ucdavis'], -d
    bin_width: list of int, -b
    mVocs: bool, default=False, -v
    session_index: int, default=0, -s
    num_itr: int, default=100000, -n
    
Example usage:
    python bootstrap_normalizer.py -d ucsf -b 50 -s 3 -v -n 1000
"""

import time
import argparse
import numpy as np


# # local
# from auditory_cortex import saved_corr_dir
# from auditory_cortex import config
# import auditory_cortex.utils as utils
# import auditory_cortex.deprecated.models as models
# import auditory_cortex.io_utils.io as io
# from auditory_cortex.io_utils.io import write_lmbdas
# from auditory_cortex import valid_model_names


from auditory_cortex import NEURAL_DATASETS
from auditory_cortex.neural_data.normalizer_calculator import NormalizerCalculator
# ------------------  set up logging ----------------------
import logging
from auditory_cortex.utils import set_up_logging
set_up_logging()

def save_normalizer_bootstrap_dist(args):

    session_index = args.session_index
    num_itr = args.num_itr
    dataset_name = args.dataset_name
    mVocs = args.mVocs
    bin_width = args.bin_width
    percent_durations = [20, 40, 60, 80, 100]
    epoch_ids = np.arange(100)

    if dataset_name == 'ucsf':
        sessions = np.array([
            '200205', '191121', '191210', # non-primary sessions 27/35 channels...
            '200206', '191113', '180814', '200213', '191206', '191125', '180731',
            '200207', '180807',      # primary sessions 195/227 channels...
            ])
        # norm_obj = Normalizer()
        # sessions = norm_obj.metadata.get_all_available_sessions()
        # sessions = np.sort(sessions)
        session = sessions[session_index]
    else:
        session = session_index

    norm_obj = NormalizerCalculator(dataset_name)
    norm_obj.save_bootstrapped_distributions(
        session, percent_durations, epoch_ids=epoch_ids, 
        bin_width=bin_width, num_itr=num_itr, mVocs=mVocs
        )




# ------------------  get parser ----------------------#

def get_parser():
    # create an instance of argument parser
    parser = argparse.ArgumentParser(
        description='This is to compute and save normalizer distributions (both True & Null)'+
            'for different setting of percent_durations and number of repeats. '+
            'At each setting, distribution is computed by number of iterations times.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument(
		'-d','--dataset_name', dest='dataset_name', type= str, action='store',
		choices=NEURAL_DATASETS, required=True,
		help = "Name of neural data to be used."
	)
    parser.add_argument(
        '-b', '--bin_width', dest='bin_width', type=int, action='store', 
        default=50,
        help="Choose bin width in ms."
    )
    parser.add_argument(
        '-s', '--session', dest='session_index', type=int, action='store', 
        default=0,
        help="Choose sessions index."
    )
    parser.add_argument(
        '-v', '--mVocs', dest='mVocs', action='store_true', default=False, 
        help="Choose to use mVocs."
    )
    parser.add_argument(
        '-n', '--num_itr', dest='num_itr', type=int, action='store', 
        default=1000,
        help="Number of iterations for each distribution."
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

    save_normalizer_bootstrap_dist(args)
    elapsed_time = time.time() - start_time
    logging.info(f"It took {elapsed_time/60:.1f} min. to run.")