import time
import argparse

from auditory_cortex import valid_model_names
from auditory_cortex.dataloader2 import DataLoader
from auditory_cortex.neural_data import create_neural_dataset
from auditory_cortex.dnn_feature_extractor import create_feature_extractor

import logging

# Configure the logging system
logging.basicConfig(
    level=logging.INFO,  # Logging level for the logger DEBUG, INFO, WARNING, ERROR, CRITICAL
    # format="%(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

# ------------------  cache features function ----------------------

def cache_features(args):

	
	i = args.ind
	shuffled = args.shuffled
	mVocs = args.mVocs
	factor = args.factor
	dataset_name = args.dataset_name
	
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

	logger.info(f"Done...!")

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
		'-i','--ind', dest='ind', type= int, action='store',
		required=True,
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
	parser.add_argument(
		'-f','--factor', dest='factor', type=float, action='store', default=1,
		# choices=[],
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
		logger.info(f"{arg:15} : {getattr(args, arg)}")

	cache_features(args)
	elapsed_time = time.time() - start_time
	logger.info(f"It took {elapsed_time/60:.1f} min. to run.")
