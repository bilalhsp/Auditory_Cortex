"""
Script to compute and save normalizer distributions for neural datasets.

Here is the list of arguments:
dataset_name: str ['ucsf', 'ucdavis']
bin_width: list of int
start_ind: int, default=0
end_ind: int, default=45
force_redo: bool, default=False
app: bool, default=False
mVocs: bool, default=False

Example usage:
python normalizer_save_results2.py -d ucsf -b 20 50 -v -s 0 -e 45 
"""
# ------------------  imports ----------------------#
import time
import argparse
from auditory_cortex.neural_data import NormalizerCalculator
from auditory_cortex.neural_data import create_neural_metadata
from auditory_cortex import NEURAL_DATASETS

import sys
import logging
# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,  # Capture all logs from DEBUG level and above
    handlers=[
        logging.StreamHandler(sys.stdout),  # Stream to the notebook's output
    ]
)

# ------------------  get parser ----------------------#

def get_parser():
    # create an instance of argument parser
    parser = argparse.ArgumentParser(
        description='This is to compute and save the normalizer '+
            'for the sessions of neural data.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument(
        '-b','--bin_width', dest='bin_widths', nargs='+', type=int, action='store', 
        default=[20],
        help="Choose bin_width for normalizers."
    )
    parser.add_argument(
		'-d','--dataset_name', dest='dataset_name', type= str, action='store',
		choices=NEURAL_DATASETS,
		help = "Name of neural data to be used."
	)
    parser.add_argument(
        '-s','--start', dest='start_ind', type=int, action='store', 
        default=0,
        help="Choose sessions starting index to compute normalizers at."
    )
    parser.add_argument(
        '-e','--end', dest='end_ind', type=int, action='store', 
        default=45,
        help="Choose sessions ending index to compute normalizers at."
    )
    parser.add_argument(
        '-f','--force_redo', dest='force_redo', action='store_true', default=False,
        help="Specify if force redoing the distribution again.."
    )
    parser.add_argument(
        '--app', dest='app', action='store_true', default=False,
        help="Specify if using the method of all possible pairs (app) to compute normalizers."
    )
    parser.add_argument(
        '-v','--mVocs', dest='mVocs', action='store_true', default=False,
        help="Specify if computing for mVoc"
    )

    return parser


# ------------------  Normalizer computing function ----------------------#

def compute_and_save_normalizers(args):



    dataset_name = args.dataset_name
    random_pairs = not args.app
    force_redo = args.force_redo
    mVocs = args.mVocs

    # create an object for the metadata
    metadata = create_neural_metadata(dataset_name)
    sessions = metadata.get_all_available_sessions()[args.start_ind:args.end_ind]
    logging.info(f"Running for sessions starting at index-{args.start_ind}, ending before index-{args.end_ind}..")
    
    excluded_sessions = ['190726', '200213']
    for session in sessions:
            
        if mVocs and (session in excluded_sessions):
            logging.info(f"Excluding session: {session}")
            continue
        else:

            norm_obj = NormalizerCalculator(dataset_name, session)
            for bin_width in args.bin_widths:
                bin_width = int(bin_width)
                norm_dist = norm_obj.get_normalizer_for_session(
                    bin_width=bin_width, delay=0, force_redo=force_redo,
                    mVocs=mVocs, random_pairs=random_pairs
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


