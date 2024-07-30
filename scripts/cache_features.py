import time
import argparse

from auditory_cortex import valid_model_names
from auditory_cortex.dataloader import DataLoader
from auditory_cortex.plotters.plotter_utils import PlotterUtils


# ------------------  cache features function ----------------------

def cache_features(args):

    dataloader = DataLoader()
    i = args.ind
    shuffled = args.shuffled
    mVocs = args.mVocs
    
    assert i < len(valid_model_names), f"Specified i={i} out of range."
    model_name = valid_model_names[i]
    # try:
    #     model_name = PlotterUtils.model_names[i]
    # except:
    #     i = i-6
    #     extra_models = ['wav2letter_spect']
    #     model_name = extra_models[i]
    # model_names = [
    #     'deepspeech2', 'speech2text', 'wav2letter_modified',
    #     'whisper_tiny', 'whisper_base', 'wav2vec2',
    #     ]
    print(f"Model: {model_name}")
    if mVocs:
        print(f"Loading features for mVocs")
        raw_features = dataloader.get_raw_DNN_features_for_mVocs(
            model_name, force_reload=True, contextualized=args.contextualized, shuffled=shuffled
            )
    else:
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
    parser.add_argument(
        '-v','--mVocs', dest='mVocs', action='store_true', default=False,
        # choices=[],
        help="Specify if loading for mVocs."
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
