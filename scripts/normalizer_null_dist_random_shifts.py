import time
import argparse

from auditory_cortex.neural_data.normalizer import Normalizer


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
        # choices=[],
        help="Choose bin_width for normalizers."
    )
    parser.add_argument(
        '-n','--n_itr', dest='n_itr', type=int, action='store', 
        default=100000,
        help="Number of iterations."
    )
    parser.add_argument(
        '--min_shift_frac', dest='min_shift_frac', type=float, action='store', 
        default=0.2,
        help="Choose min shift as fraction of total sequence length."
    )
    parser.add_argument(
        '--max_shift_frac', dest='max_shift_frac', type=float, action='store', 
        default=0.8,
        help="Choose max shift as fraction of total sequence length."
    )
    parser.add_argument(
        '-f','--force_redo', dest='force_redo', action='store_true', default=False,
        # choices=[],
        help="Specify if force redoing the distribution again.."
    )
    parser.add_argument(
        '-s','--start', dest='start_ind', type=int, action='store', 
        default=0,
        # choices=[],
        help="Choose sessions starting index to compute normalizers at."
    )
    parser.add_argument(
        '-e','--end', dest='end_ind', type=int, action='store', 
        default=45,
        # choices=[],
        help="Choose sessions ending index to compute normalizers at."
    )

    
    return parser


# ------------------  Normalizer Null dist. computing function ----------------------#

def compute_and_save_normalizer_null_dist(args):

    # create an object for the normalizer

    norm_obj = Normalizer()
    sessions = norm_obj.metadata.get_all_available_sessions()[args.start_ind:args.end_ind]
    print(f"Running for sessions starting at index-{args.start_ind}, ending before index-{args.end_ind}..")

    force_redo = args.force_redo
    n_itr = args.n_itr
    min_shift_frac = args.min_shift_frac
    max_shift_frac = args.max_shift_frac
    for session in sessions:
        # all bin_widths in list
        for bin_width in args.bin_widths:
            bin_width = int(bin_width)

            null_dist_sess = norm_obj.get_normalizer_null_dist_using_random_shifts(
                session, bin_width=bin_width, n_itr=n_itr, 
                min_shift_frac=min_shift_frac, max_shift_frac=max_shift_frac,
                force_redo=force_redo
            )


# ------------------  main function ----------------------#

if __name__ == '__main__':

    start_time = time.time()
    parser = get_parser()
    args = parser.parse_args()

    # display the arguments passed
    for arg in vars(args):
        print(f"{arg:15} : {getattr(args, arg)}")

    compute_and_save_normalizer_null_dist(args)
    elapsed_time = time.time() - start_time
    print(f"It took {elapsed_time/60:.1f} min. to run in total.")