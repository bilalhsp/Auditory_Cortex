import time
import argparse

from auditory_cortex.dataloader import DataLoader
from auditory_cortex.plotters.plotter_utils import PlotterUtils


# ------------------  cache features function ----------------------

def cache_features(args):

    dataloader = DataLoader()
    i = args.ind
    shuffled = args.shuffled
    model_name = PlotterUtils.model_names[i]
    # model_names = [
    #     'deepspeech2', 'speech2text', 'wave2letter_modified',
    #     'whisper_tiny', 'whisper_base', 'wave2vec2',
    #     ]
    print(f"Model: {model_name}")
    raw_features = dataloader.get_raw_DNN_features(
        model_name, force_reload=True, contextualized=args.contextualized, shuffled=shuffled
        )

    print(f"Done...!")

# ------------------  get parser ----------------------#

def get_parser():
    # create an instance of argument parser
    parser = argparse.ArgumentParser(
        description="This is to load neural spikes and cache the results on "+
        "'cache_dir' on scratch. ",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

    parser.add_argument(
        '-i','--ind', dest='ind', type= int, action='store',
        # default=[5, 10, 20, 40],
        # choices=[],
        help = "Index of the network, that we want to load features for."
    )
    parser.add_argument(
        '-c','--contextualized', dest='contextualized', action='store_true', default=False,
        help="Choose the type of features to extract."
    )
    parser.add_argument(
        '-s','--shuffle', dest='shuffled', action='store_true', default=False,
        # choices=[],
        help="Specify if shuffled network to be used."
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

    cache_features(args)
    elapsed_time = time.time() - start_time
    print(f"It took {elapsed_time/60:.1f} min. to run.")
