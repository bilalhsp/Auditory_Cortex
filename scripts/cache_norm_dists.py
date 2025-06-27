"""
Script to compute and save inter-trial distributions
(both True & Null distributions) for neural datasets.

Args:
    dataset_name: str ['ucsf', 'ucdavis'], -d
    bin_width: list of int, -b
    mVocs: bool, default=False, -v
    num_itr: int, default=100000, -n
    start_ind: int, default=0, -s
    end_ind: int, default=45, -e
    force_redo: bool, default=False, -f
    
Example usage:
    python cache_norm_dists.py -d ucsf -b 20 50 -v -n 100000 -s 0 -e 45 
"""
# ------------------  imports ----------------------#
import time
import argparse
from auditory_cortex.neural_data import NormalizerCalculator
from auditory_cortex.neural_data import create_neural_metadata
from auditory_cortex import NEURAL_DATASETS

# ------------------  set up logging ----------------------
import logging
from auditory_cortex.utils import set_up_logging
set_up_logging()

# ------------------  get parser ----------------------#

def get_parser():
    # create an instance of argument parser
    parser = argparse.ArgumentParser(
        description='This is to compute and save the normalizer distributions '+
            '(both True & Null) for the sessions of neural data.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument(
		'-d','--dataset_name', dest='dataset_name', type= str, action='store',
		choices=NEURAL_DATASETS, required=True,
		help = "Name of neural data to be used."
	)
    parser.add_argument(
        '-b','--bin_width', dest='bin_widths', nargs='+', type=int, action='store', 
        default=[50],
        help="Choose bin_width for normalizers."
    )
    parser.add_argument(
        '-v','--mVocs', dest='mVocs', action='store_true', default=False,
        help="Specify if computing for mVocs"
    )
    parser.add_argument(
        '-n','--num_itr', dest='num_itr', type=int, action='store', 
        default=100000,
        help="Number of iterations."
    )
    parser.add_argument(
        '-s','--start', dest='start_ind', type=int, action='store', 
        default=0,
        help="Index to choose subset of sessions from the list of all sessions."
    )
    parser.add_argument(
        '-e','--end', dest='end_ind', type=int, action='store', 
        default=45,
        help="Index to choose subset of sessions from the list of all sessions."
    )
    parser.add_argument(
        '-f','--force_redo', dest='force_redo', action='store_true', default=False,
        help="Specify if force redoing the distribution again.."
    )

    return parser


# ------------------  Normalizer computing function ----------------------#

def compute_and_save_normalizers(args):
    dataset_name = args.dataset_name
    force_redo = args.force_redo
    mVocs = args.mVocs
    num_itr = args.num_itr
    # create an object for the metadata
    metadata = create_neural_metadata(dataset_name)
    sessions = metadata.get_all_available_sessions()[args.start_ind:args.end_ind]
    logging.info(f"Running for sessions starting at index-{args.start_ind}, ending before index-{args.end_ind}..")
    norm_obj = NormalizerCalculator(dataset_name)
    excluded_sessions = ['190726', '200213']
    for session in sessions:
            
        if dataset_name=='ucsf' and mVocs and (session in excluded_sessions):
            logging.info(f"Excluding session: {session}")
            continue
        else:
            for bin_width in args.bin_widths:
                bin_width = int(bin_width)
                norm_dist, null_dist = norm_obj.get_inter_trial_corr_dists_for_session(
                    session, bin_width=bin_width, mVocs=mVocs, num_itr=num_itr,
                    force_redo=force_redo,
                    )
                
# ------------------  main function ----------------------#

if __name__ == '__main__':

    start_time = time.time()
    parser = get_parser()
    args = parser.parse_args()

    # display the arguments passed
    for arg in vars(args):
        logging.info(f"{arg:15} : {getattr(args, arg)}")

    compute_and_save_normalizers(args)
    elapsed_time = time.time() - start_time
    logging.info(f"It took {elapsed_time/60:.1f} min. to run in total.")


