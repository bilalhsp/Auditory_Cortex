import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import argparse
import gc

# local
from auditory_cortex import saved_corr_dir
from auditory_cortex import config
import auditory_cortex.utils as utils
import auditory_cortex.deprecated.models as models
import auditory_cortex.io_utils.io as io
from auditory_cortex.io_utils.io import write_lmbdas
from auditory_cortex import valid_model_names
from auditory_cortex.neural_data import NeuralMetaData
from auditory_cortex.deprecated.datasets import BaselineDataset, DNNDataset
from auditory_cortex.computational_models.encoding import TRF


def compute_and_save_regression(args):

	START = time.time()
	# bin_widths = config['bin_widths']
	bin_widths = args.bin_widths
	model_name = args.model_name
	layer_ID = args.layer_ID
	shuffled = args.shuffled
	test_trial = args.test_trial
	identifier = args.identifier
	mVocs = args.mVocs
	LPF = args.LPF  
	N_sents = args.N_sents
	test_bootstrap = args.test_bootstrap
	N_test_trials = args.N_test_trials
	# itr = args.itr
	# fixed parameters..
	
	tmin=0
	delay=0
	# N_sents=500
	num_workers=16
	num_folds=3
	LPF_analysis_bw = 20
	# LPF_analysis_bw = 10
	
	lags=[300]
	use_nonlinearity=False

	id_details = f'trf_lags{lags[0]}_bw{bin_widths[0]}'
	if identifier != '':
		identifier = id_details + '_boot_' + identifier
	else:
		identifier = id_details

	if mVocs:
		identifier = 'mVocs_' + identifier
	else:
		identifier = 'timit_' + identifier

	csv_file_name = 'corr_results.csv'
	# if identifier != '':
	csv_file_name = identifier + '_' + csv_file_name

	if shuffled:
		csv_file_name = model_name + '_reset_' + csv_file_name
	else:
		csv_file_name = model_name + '_' + csv_file_name
	# CSV file to save the results at
	file_exists = False
	file_path = os.path.join(saved_corr_dir, csv_file_name)
	if os.path.exists(file_path):
		data = pd.read_csv(file_path)
		file_exists = True

	if shuffled:
		print(f"Running TRF for 'Untrained' networks...")
	else:
		print(f"Running TRF for 'Trained' networks...")

	metadata = NeuralMetaData()
	sessions = metadata.get_all_available_sessions()
	# ################################################################
	# # list of significant sessions only...
	# sig_sessions = np.array([180613., 180627., 180719., 180720., 180728., 180730., 180731.,
	# 					180807., 180808., 180814., 190606., 191113., 191121., 191125.,
	# 					191206., 191210., 200205., 200206., 200207., 200213., 200219.])

	# sessions = sig_sessions.astype(int)
	###############################################################
	sessions = np.sort(sessions)
	sessions = sessions[args.start_ind:args.end_ind]
	# sessions = sessions[:20]
	# sessions = sessions[20:]
	current_time = time.time()
	elapsed_time = current_time - START

	for bin_width in bin_widths:
		sessions = np.array([
			'200205', '191121', '191210', # non-primary sessions 27/35 channels...
			'200206', '191113', '180814', '200213', '191206', '191125', '180731',
			'200207', '180807',      # primary sessions 195/227 channels...
			])
		# Session in data_dir that we do not have results for...
		if file_exists:
			sessions_done = data[
					(data['delay']==delay) & \
					(data['bin_width']==bin_width) 
				]['session'].unique()

			subjects = sessions[np.isin(sessions,sessions_done.astype(int).astype(str), invert=True)]
		else:
			subjects = sessions

		for session in subjects:
			if mVocs:
				excluded_sessions = ['190726', '200213']
				if session in excluded_sessions:
					print(f"Excluding session: {session}")
					continue
			print(f"Working with '{session}'")

			dataset = DNNDataset(
				session, bin_width, model_name, layer_ID, mVocs=mVocs,
				shuffled=shuffled,
				LPF=LPF, LPF_analysis_bw=LPF_analysis_bw
				)
			trf_obj = TRF(model_name, dataset)

			if test_bootstrap:
				trf_model = trf_obj.load_saved_model(model_name, session, layer_ID, bin_width, shuffled=shuffled)
				corr = trf_obj.evaluate(trf_model, test_trial=N_test_trials)
				opt_lag = [0]*corr.size
				opt_lmbda = [1]*corr.size
				N_sents = N_test_trials
			else:
				corr, opt_lag, opt_lmbda, trf_model = trf_obj.grid_search_CV(
						lags=lags, tmin=tmin,
						num_workers=num_workers, num_folds=num_folds,
						use_nonlinearity=use_nonlinearity,
						test_trial=test_trial, N_sents=N_sents
					)
				# betas = trf_model.coef_
				# io.write_trf_parameters(
				#     model_name, session, betas, bin_width=bin_width, 
				#     shuffled=shuffled, layer_ID=layer_ID, LPF=LPF,
				#     )
				
			if mVocs:
				mVocs_corr = corr
				timit_corr = np.zeros_like(corr)
			else:
				mVocs_corr = np.zeros_like(corr)
				timit_corr = corr


			corr_dict = {
				'test_cc_raw': timit_corr[None,...],
				'mVocs_test_cc_raw': mVocs_corr[None,...],
				'win': bin_width,
				'delay': delay, 
				'session': session,
				'model': model_name,
				'N_sents': N_sents,
				'layer_ids': [layer_ID],
				'opt_lag': opt_lag,
				'opt_lmbda': np.log10(opt_lmbda),
				'poiss_entropy': np.zeros_like(corr[None,...]),
				'uncertainty_per_spike': np.zeros_like(corr[None,...]),
				'bits_per_spike_NLB': np.zeros_like(corr[None,...]),
				}

			df = utils.write_to_disk(corr_dict, file_path, normalizer=None)

			# make sure to delete the objects to free up memory
			del dataset
			del trf_obj
	
			gc.collect()

	END = time.time()
	print(f"Took {(END-START)/60:.2f} min., for bin_widths: '{bin_widths}'.")



# ------------------  get parser ----------------------#

def get_parser():

	parser = argparse.ArgumentParser(
		description='This is to compute and save regression results for for layers '+
			'of DNN models and neural areas for the purpose of doing bootstrap analysis.'+
			'This script is useful for two types of bootstrapping analysis: '+
			'1. Bootstrapping on the training set, and '+
			'2. Bootstrapping on the test set. '+
			'For the first, specify the proportion of training data to be used to '+
			'fit the betas as N_sents. For the second, specify the number of test trials to be used.',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter
		)

	parser.add_argument(
		'-m', '--model_name', dest='model_name', action='store',
		choices=valid_model_names,
		required=True,
		help='model to be used for Regression analysis.'
	)

	parser.add_argument(
		'-b','--bin_widths', dest='bin_widths', nargs='+', type= int, action='store',
		default=[50],
		help="Specify list of bin_widths."
	)
	parser.add_argument(
		'-l','--layer', dest='layer_ID', type=int, action='store',
		required=True,
		help="Specify the layer ID."
	)
	parser.add_argument(
		'-N','--N_sents', dest='N_sents', type=int, action='store',
		choices=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
		help="Specify the proportion of total stimuli duration to be used."
	)
	parser.add_argument(
		'-i','--identifier', dest='identifier', type= str, action='store',
		default='',
		help="Specify identifier for saved results."
	)
	parser.add_argument(
		'-s','--shuffle', dest='shuffled', action='store_true', default=False,
		help="Specify if shuffled network to be used."
	)
	parser.add_argument(
		'-v','--mVocs', dest='mVocs', action='store_true', default=False,
		help="Specify if spikes for mVocs are to be used."
	)
	parser.add_argument(
		'-L','--LPF', dest='LPF', action='store_true', default=False,
		help="Specify if features are to be low pass filtered."
	)
	parser.add_argument(
		'-t','--test_trial', dest='test_trial', type= int, action='store',
		default=None,
		help="trial to test on."
	)
	parser.add_argument(
		'--start', dest='start_ind', type=int, action='store', 
		default=0,
		# choices=[],
		help="Choose sessions starting index to compute results at."
	)
	parser.add_argument(
		'--end', dest='end_ind', type=int, action='store', 
		default=41,
		# choices=[],
		help="Choose sessions ending index to compute results at."
	)
	parser.add_argument(
		'--test_bootstrap', dest='test_bootstrap', action='store_true', default=False,
		help="Specify if bootstrap on the test set is required."
	)
	parser.add_argument(
		'--N_test_trials', dest='N_test_trials', type=int, action='store',
		choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
		help="Specify the number of test trials to be used."
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

	compute_and_save_regression(args)
	elapsed_time = time.time() - start_time
	print(f"It took {elapsed_time/60:.1f} min. to run.")