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
        default=[50],
        # choices=[],
        help="Choose bin_width for normalizers."
    )
    parser.add_argument(
        '-s','--spike_rate', dest='spike_rates', nargs='+', type=int, action='store', 
        default=[50],
        # choices=[],
        help="Choose bin_width for normalizers."
    )
    parser.add_argument(
        '-n','--n_itr', dest='n_itr', type=int, action='store', 
        default=1000000,
        help="Number of iterations."
    )
    
    parser.add_argument(
        '-f','--force_redo', dest='force_redo', action='store_true', default=False,
        # choices=[],
        help="Specify if force redoing the distribution again.."
    )
    parser.add_argument(
        '-v','--mVocs', dest='mVocs', action='store_true', default=False,
        help="Specify if spikes for mVocs are to be used."
    )
    # Deprecated.
    # parser.add_argument(
    #     '-i','--identifier', dest='identifier', type= str, action='store',
    #     default='modified_bins_normalizer',
    #     # choices=[],
    #     help="Specify identifier for saved results."
    # )
    
    return parser



# ------------------  Normalizer computing function ----------------------#

def compute_and_save_null_dist(args):

    # create an object for the normalizer
    spike_rates = args.spike_rates
    n_itr = args.n_itr
    force_redo = args.force_redo
    mVocs=args.mVocs
    # normalizer_filename = f"{args.identifier}.csv"
    norm_obj = Normalizer()

    for bin_width in args.bin_widths:
        for spike_rate in spike_rates:
            bin_width = int(bin_width)
            null_dist_poisson = norm_obj.get_normalizer_null_dist_using_poisson(
                bin_width, spike_rate=spike_rate, itr=n_itr, force_redo=force_redo,
                mVocs=mVocs
            )


# ------------------  main function ----------------------#

if __name__ == '__main__':

    start_time = time.time()
    parser = get_parser()
    args = parser.parse_args()

    # display the arguments passed
    for arg in vars(args):
        print(f"{arg:15} : {getattr(args, arg)}")

    compute_and_save_null_dist(args)
    elapsed_time = time.time() - start_time
    print(f"It took {elapsed_time/60:.1f} min. to run in total.")