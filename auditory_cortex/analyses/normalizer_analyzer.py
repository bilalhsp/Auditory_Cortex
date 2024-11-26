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
	


