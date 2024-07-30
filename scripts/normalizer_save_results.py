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
        '-i','--identifier', dest='identifier', type= str, action='store',
        default='modified_bins_normalizer',
        # choices=[],
        help="Specify identifier for saved results."
    )
    parser.add_argument(
        '-d','--delay', dest='delays', nargs='+', type=int, action='store', 
        default=[0],
        # choices=[],
        help="Choose delays to compute normalizers at."
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
    parser.add_argument(
        '-f','--force_redo', dest='force_redo', action='store_true', default=False,
        # choices=[],
        help="Specify if force redoing the distribution again.."
    )
    parser.add_argument(
        '-r','--random_pairs', dest='random_pairs', action='store_true', default=False,
        # choices=[],
        help="Specify if force redoing the distribution again.."
    )
    parser.add_argument(
        '-v','--mVocs', dest='mVocs', action='store_true', default=False,
        # choices=[],
        help="Specify if computing for mVoc"
    )

    
    return parser


# ------------------  Normalizer computing function ----------------------#

def compute_and_save_normalizers(args):

    # create an object for the normalizer

    normalizer_filename = f"{args.identifier}.csv"
    norm_obj = Normalizer(normalizer_filename)
    sessions = norm_obj.metadata.get_all_available_sessions()[args.start_ind:args.end_ind]
    print(f"Running for sessions starting at index-{args.start_ind}, ending before index-{args.end_ind}..")
    random_pairs = args.random_pairs
    force_redo = args.force_redo
    mVocs = args.mVocs
    excluded_sessions = ['190726', '200213']
    for session in sessions:
            
        if mVocs and (session in excluded_sessions):
            print(f"Excluding session: {session}")
            continue
        else:
            # all bin_widths in list
            for bin_width in args.bin_widths:
                bin_width = int(bin_width)
                for delay in args.delays:
                    # select_data = norm_obj.get_normalizer_for_session(
                    #     session, bin_width=bin_width, delay=delay
                    # )
                    if random_pairs:
                        session_norm = norm_obj.get_normalizer_for_session_random_pairs(
                            session, bin_width=bin_width, delay=delay, force_redo=force_redo,
                            mVocs=mVocs
                        )
                    else:
                        session_norm = norm_obj.get_normalizer_for_session_app(
                            session, bin_width=bin_width, delay=delay, force_redo=force_redo
                        )
                # print(f"Saving normalizers for for bin_width-{bin_width} & delay-{delay}...")
                # norm_obj.save_normalizer_for_all_sessions(bin_width=bin_width, delay=delay)


# ------------------  Normalizer computing function ----------------------#

# def compute_and_save_null_dist(args):

#     # create an object for the normalizer

#     normalizer_filename = f"{args.identifier}.csv"
#     norm_obj = Normalizer(normalizer_filename)

#     for bin_width in args.bin_widths:
#         bin_width = int(bin_width)
#         null_dist_poisson = norm_obj.get_normalizer_threshold_using_poisson(
#             bin_width, spike_rate=50
#         )


# ------------------  main function ----------------------#

if __name__ == '__main__':

    start_time = time.time()
    parser = get_parser()
    args = parser.parse_args()

    # display the arguments passed
    for arg in vars(args):
        print(f"{arg:15} : {getattr(args, arg)}")

    compute_and_save_normalizers(args)
    elapsed_time = time.time() - start_time
    print(f"It took {elapsed_time/60:.1f} min. to run in total.")