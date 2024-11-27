import numpy as np

import auditory_cortex.io_utils.io as io

import logging
logger = logging.getLogger(__name__)

class NormalizerAnalyzer:

	def __init__(self):
		pass


	def get_normalizer_for_session_random_pairs(
		self, session, bin_width=50, mVocs=False, dataset_name='ucsf', random_pairs=True, **kwargs):
		"""Retrieves the distribution of normalizer for all the channels 
		of the specified session, at specified bin_width and delay.
		"""
		session = str(int(float(session)))    # to make sure session is in str format.
		bin_width = int(bin_width)
		logger.info(f"Getting normalizer dist. for sess-{session}, bw-{bin_width}, mVocs={mVocs}")

		if random_pairs:
			method = 'random'
		else:
			method = 'app'
		
		if dataset_name=='ucsf' and mVocs:
			# handling the excluded sessions for mVocs...
			if session == '190726':
				return np.zeros((1000, 60))
			elif session == '200213':
				return np.zeros((1000, 64))
			
		norm_dict_all_sessions = io.read_normalizer_distribution(
				bin_width=bin_width, delay=0, method=method, mVocs=mVocs, dataset_name=dataset_name
				)
		
		if (norm_dict_all_sessions is None) or (session not in norm_dict_all_sessions.keys()):
			raise ValueError(f"Normalizer distribution not found for session-{session}, bw-{bin_width}, mVocs={mVocs}")

		norm_dist_session =  norm_dict_all_sessions[session]
		# Redundant check: We have already penalized correlations for being NAN (zero sequence)
		# check zero valid samples in the distribution
		if norm_dist_session.shape[0] == 0:
			norm_dist_session = np.zeros((1, norm_dist_session.shape[-1]))
		return norm_dist_session


	def get_normalizer_null_dist_using_poisson(
		self, bin_width=50, mVocs=False, dataset_name='ucsf', spike_rate=50, **kwargs
		):
		"""Retrieves null distribution for normalizer using poisson
		sequences. Reads off the saved results from memory, raises an 
		error if not found.
		
		Args:
			bin_width: int = bin width in ms
			mVocs: bool = If True, 
			dataset_name: str = Name of the dataset
		"""

		null_dist_poisson = io.read_normalizer_null_distribution_using_poisson(
			bin_width=bin_width, spike_rate=spike_rate, mVocs=mVocs, dataset_name=dataset_name
			)
		if null_dist_poisson is None:
			raise ValueError(f"Null distribution not found for bw-{bin_width}, mVocs={mVocs}")
		return null_dist_poisson
	
	def get_normalizer_bootstrap_distributions(
		self, session, iterations:list, percent_durs:list, num_trials: list,
		bin_width=50, dataset_name='ucsf', mVocs=False
		):
		"""Retrieves distributions of normalizer for bootstrap analysis,
		for the specified session.

		Args:
			session: str = session id
			iterations: list = iteration ids of bootstrap analysis
			percent_durs: list = list of percent durations
			num_trials: list = list of number of trials for each percent duration
			bin_width: int = bin width in ms
			dataset_name: str = Name of the dataset
			mVocs: bool = If True, 

		Returns:
			dict of dict: {num_trials: {percent_dur: np.array(num_ch, num_samples)}}
		"""
		# std_devs = np.zeros((len(num_trials), len(percent_durs)))
		std_devs = {}
		for i, num_tr in enumerate(num_trials):
			std_devs_dur = {}
			for j, pd in enumerate(percent_durs):
				dist_all_itr = []
				for itr in iterations:
	
					bootstrap_dist = io.read_bootstrap_normalizer_dist(
						session, itr, pd, num_tr, bin_width, dataset_name=dataset_name, mVocs=mVocs
					)
					dist_all_itr.append(np.median(bootstrap_dist, axis=0))				
				std_devs_dur[pd] = np.array(dist_all_itr).transpose() #	np.std(dist_all_itr, axis=0)
			std_devs[num_tr] = std_devs_dur
		return std_devs

	
	


