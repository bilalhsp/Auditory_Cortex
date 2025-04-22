import os
import time
import argparse
from auditory_cortex.neural_data.deprecated.normalizer import Normalizer


# ------------------  save qualitative figures ----------------------#

def save_sig_sessions_and_channels(args):
    norm_obj = Normalizer()
    bin_width=50
    p_threshold = args.p_value
    mVocs=args.mVocs
    force_redo = args.force_redo
    significant_sessions_and_channels = norm_obj.get_significant_sessions_and_channels_using_poisson_null(
                bin_width=bin_width, p_threshold=p_threshold, mVocs=mVocs, force_redo=force_redo
                )

 

# ------------------  get parser ----------------------#

def get_parser():
    # create an instance of argument parser
    parser = argparse.ArgumentParser(
        description='This is to save the qualitative neural prediction figures to cache dir'+
         'pick the better looking figures for the paper. ',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument(
        '-v','--mVocs', dest='mVocs', action='store_true', default=False,
        help="Specify if spikes for mVocs are to be used."
    )
    parser.add_argument(
        '-f','--force_redo', dest='force_redo', action='store_true', default=False,
        help="Specify force redo."
    )
    parser.add_argument(
        '-p', '--p_value', dest='p_value', type=float, action='store', default=0.05, 
        help="p-value threshold to select significant sessions and channels."
    )
    return parser




# ------------------  main function ----------------------#

if __name__ == '__main__':

    start_time = time.time()
    print("Starting out...")
    parser = get_parser()
    args = parser.parse_args()

    # display the arguments passed
    for arg in vars(args):
        print(f"{arg:15} : {getattr(args, arg)}")

    save_sig_sessions_and_channels(args)
    elapsed_time = time.time() - start_time
    print(f"It took {elapsed_time/60:.1f} min. to run.")




       
