"""
This script is used to cache features for a given DNN model and neural dataset.

Args:
    dataset_name: str ['ucsf', 'ucdavis'], -d
    ind: int, index of the DNN model to be used (read from config), -i
    contextualized: bool, -c
    shuffle: bool, -s
    mVocs: bool, -v
    factor: float, relevant for shuffled -f

Example usage:  
    python cache_features.py -d ucsf -i 3 -s -v
    python cache_features.py -d ucdavis -i 1 -s -v
"""
# ------------------  set up logging ----------------------
import logging
from auditory_cortex.utils import set_up_logging
set_up_logging()

import time
import argparse

from auditory_cortex import valid_model_names
from auditory_cortex.dataloader import DataLoader
from auditory_cortex.neural_data import create_neural_dataset
from auditory_cortex.dnn_feature_extractor import create_feature_extractor

# ------------------  cache features function ----------------------

def cache_features(args):

    
    i = args.ind
    shuffled = args.shuffled
    mVocs = args.mVocs
    factor = args.factor
    dataset_name = args.dataset_name
    model_name = args.model_name
    
    assert model_name is not None or i is not None, \
        f"Either model_name or index (i) must be specified."
    if model_name is None:    
        assert i < len(valid_model_names), f"Specified i={i} out of range."
        model_name = valid_model_names[i]
    logging.info(f"model_name: {model_name}")
    # load the neural dataset
    dataset_obj = create_neural_dataset(dataset_name)
    feature_extractor = create_feature_extractor(model_name, shuffled=shuffled)

    dataloader = DataLoader(dataset_obj, feature_extractor)
    

    # load the features
    features = dataloader.get_raw_DNN_features(
        mVocs=mVocs, force_reload=True, contextualized=False, scale_factor=factor
        )


    # if mVocs:
    # 	logging.info(f"Loading features for mVocs")
    # 	raw_features = dataloader.get_raw_DNN_features_for_mVocs(
    # 		model_name, force_reload=True, contextualized=args.contextualized, shuffled=shuffled
    # 		)
    # else:
    # 	raw_features = dataloader.get_raw_DNN_features(
    # 		model_name, force_reload=True, contextualized=args.contextualized, shuffled=shuffled,
    # 		scale_factor=factor
    # 		)

    logging.info(f"Done...!")

# ------------------  get parser ----------------------#

def get_parser():
    # create an instance of argument parser
    parser = argparse.ArgumentParser(
        description="This is to load neural spikes and cache the results on "+
        "'cache_dir' on scratch. ",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    
    parser.add_argument(
        '-d','--dataset_name', dest='dataset_name', type= str, action='store',
        choices=['ucsf', 'ucdavis'], required=True,
        help = "Name of neural data to be used."
    )
    parser.add_argument(
        '-m', '--model_name', dest='model_name', action='store',
        choices=valid_model_names, default=None,
        help='model to be used for Regression analysis.'
    )
    parser.add_argument(
        '-i','--ind', dest='ind', type= int, action='store', default=None,
        help = "Index of the network, that we want to load features for."
    )
    parser.add_argument(
        '-c','--contextualized', dest='contextualized', action='store_true', default=False,
        help="Choose the type of features to extract."
    )
    parser.add_argument(
        '-s','--shuffle', dest='shuffled', action='store_true', default=False,
        help="Specify if shuffled network to be used."
    )
    parser.add_argument(
        '-v','--mVocs', dest='mVocs', action='store_true', default=False,
        help="Specify if loading for mVocs."
    )
    parser.add_argument(
        '-f','--factor', dest='factor', type=float, action='store', default=1,
        help="Specify the scale factor."
    )

    return parser



# ------------------  main function ----------------------#

if __name__ == '__main__':

    start_time = time.time()
    parser = get_parser()
    args = parser.parse_args()

    # display the arguments passed
    for arg in vars(args):
        logging.info(f"{arg:15} : {getattr(args, arg)}")

    cache_features(args)
    elapsed_time = time.time() - start_time
    logging.info(f"It took {elapsed_time/60:.1f} min. to run.")
