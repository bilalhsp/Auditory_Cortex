"""
Script to compute and save Null distributions for neural datasets
and stimulus type.

Here is the list of arguments:
dataset_name: str ['ucsf', 'ucdavis']
bin_width: list of int
spike_rate: list of int, default=[50]
n_itr: int, default=1000000
force_redo: bool, default=False
mVocs: bool, default=False

Example usage:
python normalizer_null_dist_poisson.py -d ucsf -b 20 50 -s 50 -n 1000000 -v -f 
"""

import time
import argparse

# from auditory_cortex.neural_data.normalizer import Normalizer
from auditory_cortex.neural_data import NormalizerCalculator
from auditory_cortex import NEURAL_DATASETS

# ------------------  set up logging ----------------------
import logging
from auditory_cortex.utils import set_up_logging
set_up_logging()

# ------------------  get parser ----------------------#

def get_parser():
    # create an instance of argument parser
    parser = argparse.ArgumentParser(
        description='This is to compute and save the normalizer '+
            'for the sessions of neural data.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument(
		'-d','--dataset_name', dest='dataset_name', type= str, action='store',
		choices=NEURAL_DATASETS,
		help = "Name of neural data to be used."
	)
    parser.add_argument(
        '-b','--bin_width', dest='bin_widths', nargs='+', type=int, action='store', 
        default=[50],
        help="Choose bin_width for normalizers."
    )
    parser.add_argument(
        '-s','--spike_rate', dest='spike_rates', nargs='+', type=int, action='store', 
        default=[50],
        help="Choose the spike rate for normalizers."
    )
    parser.add_argument(
        '-n','--num_itr', dest='num_itr', type=int, action='store', 
        default=100000,
        help="Number of iterations."
    )
    parser.add_argument(
        '-f','--force_redo', dest='force_redo', action='store_true', default=False,
        help="Specify if force redoing the distribution again.."
    )
    parser.add_argument(
        '-v','--mVocs', dest='mVocs', action='store_true', default=False,
        help="Specify if spikes for mVocs are to be used."
    )
    
    return parser



# ------------------  Normalizer computing function ----------------------#

def compute_and_save_null_dist(args):

    # create an object for the normalizer
    spike_rates = args.spike_rates
    num_itr = args.num_itr
    force_redo = args.force_redo
    mVocs=args.mVocs
    dataset_name = args.dataset_name

    norm_obj = NormalizerCalculator(dataset_name)

    for bin_width in args.bin_widths:
        for spike_rate in spike_rates:
            bin_width = int(bin_width)
            null_dist_poisson = norm_obj.get_normalizer_null_dist_using_poisson(
                bin_width, spike_rate=spike_rate, num_itr=num_itr, force_redo=force_redo,
                mVocs=mVocs
            )


# ------------------  main function ----------------------#

if __name__ == '__main__':

    start_time = time.time()
    parser = get_parser()
    args = parser.parse_args()

    # display the arguments passed
    for arg in vars(args):
        logging.info(f"{arg:15} : {getattr(args, arg)}")

    compute_and_save_null_dist(args)
    elapsed_time = time.time() - start_time
    logging.info(f"It took {elapsed_time/60:.1f} min. to run in total.")