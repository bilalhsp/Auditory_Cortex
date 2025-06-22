import time
import argparse
from auditory_cortex.deprecated.dataloader import DataLoader



# ------------------  cache spikes function ----------------------#

def cache_spikes(args):
    
    dataloader = DataLoader()
    sessions = dataloader.metadata.get_all_available_sessions()
    for session in sessions:
        print(f"Working with session-{session}...")
        # simply loading neural data for each configuration will cache it on 'cache_dir' 
        for bin_width in args.bin_widths:
            for delay in args.delays:
                _ = dataloader.get_session_spikes(session=session, bin_width=bin_width, delay=delay)

    print(f"All sessions done.")


# ------------------  get parser ----------------------#

def get_parser():
    # create an instance of argument parser
    parser = argparse.ArgumentParser(
        description="This is to load neural spikes and cache the results on "+
        "'cache_dir' on scratch. ",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

    parser.add_argument(
        '-b','--bin_widths', dest='bin_widths', nargs='+', type= int, action='store',
        default=[5, 10, 20, 40],
        # choices=[],
        help="List of bin_widths to cache neural data for."
    )

    parser.add_argument(
        '-d','--delays', dest='delays', nargs='+', type= int, action='store',
        default=[0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100],
        # choices=[],
        help="List of delays to cache neural data for."
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

    cache_spikes(args)
    elapsed_time = time.time() - start_time
    print(f"It took {elapsed_time/60:.1f} min. to run.")
