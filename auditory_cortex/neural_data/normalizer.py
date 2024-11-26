import os
import time
import numpy as np
import pandas as pd
import scipy

from .dataset import NeuralData
from .neural_meta_data import NeuralMetaData
from auditory_cortex import results_dir
import auditory_cortex.io_utils.io as io


class Normalizer:
	"""Computes the inter-trial correlations for stimuli
	having multilple trials. These will be used as normalizer
	for regression correlations and for identifying channels
	(and sessions) with good enough SNR."""

	def __init__(self, normalizer_filename = None):
		if normalizer_filename is None:
			# print(f"Using default normalizer file...")
			normalizer_filename = "modified_bins_normalizer.csv"
			# "corr_normalizer.csv" original with full length sequnces. 
		
		# print(f"Creating normalizer object from: {normalizer_filename}")
		# self.dataframe, self.filepath = Normalizer.load_data(normalizer_filename)
		self.metadata = NeuralMetaData()
		self.loaded_datasets = {}
		# self.test_sent_IDs = [12,13,32,43,56,163,212,218,287,308]

	def get_significant_sessions(self, bin_width=20, delay=0, threshold=0.068):
		"""Returns a list of session having at least one significant channels"""
		select_data = self._get_normalizers_for_bin_width_and_delay(bin_width, delay)
		select_data = select_data[select_data['normalizer'] >= threshold]
		return select_data['session'].unique()
	
	def get_good_channels(self, session, bin_width=20, delay=0, threshold=0.068):
		"""Returns a list of sig. channels for the session, bw and delay"""
		select_data = self.get_normalizer_for_session(session, bin_width=bin_width, delay=delay)
		select_data = select_data[select_data['normalizer'] >= threshold]
		return select_data['channel'].unique()
		
	def _get_normalizers_for_bin_width_and_delay(self, bin_width=20, delay=0):
		"""Retrieves section of dataframe for specified bin_width and delay"""
		select_data =  self.dataframe[
			(self.dataframe['bin_width']==bin_width) &\
			(self.dataframe['delay']==delay)
		]
		return select_data

	def get_normalizer_for_session(self, session, bin_width=20, delay=0):
		"""Retrives normalizer for the arguments."""
		session = float(session)
		select_data = self._get_normalizers_for_bin_width_and_delay(bin_width, delay)
		select_data = select_data[(select_data['session']==session)]
		if select_data.shape[0] == 0:
			print(f"Results NOT available for session-{session} at bw-{bin_width} & delay-{delay}, computing now...", end='')
			self.save_normalizer_for_session(session, bin_width=bin_width, delay=delay)
			print(f"Done.")
			# recursive call after computing and saving resutlts
			self.get_normalizer_for_session(session, bin_width, delay)
			# raise ValueError(f"Results NOT available for session-{session} at bw-{bin_width} & delay-{delay},\
			#     use 'save_normalizer_for_session(...)' or 'save_normalizer_for_all_sessions()'")
		else:
			return select_data

	
	def save_normalizer_for_all_sessions(self, bin_width=20, delay=0):
		sessions = self.metadata.get_all_available_sessions()
		select_data = self._get_normalizers_for_bin_width_and_delay(bin_width, delay)
		sessions_done = select_data['session'].unique()

		sessions_remaining = sessions[np.isin(sessions, sessions_done, invert=True)]

		for session in sessions_remaining:
			print(f"Normalizer not available for {session}" +
				f" at bw-{bin_width}, delay-{delay}, computing now...")
			strt_time = time.time()
			self.save_normalizer_for_session(session, bin_width=bin_width, delay=delay)
			end_time = time.time()
			print(f"It took {end_time-strt_time}s for session-{session}.!")


	def save_normalizer_for_session(self, session, bin_width=20, delay=0):
		"""Computes and saves the normalizer result, for the given configuration."""
		norm_dist = self._compute_normalizer(session, bin_width=bin_width,
									   delay=delay)
		session = float(session)
		num_channels = norm_dist.size
		data = np.stack([
			session*np.ones(num_channels),
			np.arange(num_channels),
			bin_width*np.ones(num_channels),
			delay*np.ones(num_channels),
			norm_dist
		], axis=1
		)
		df = pd.DataFrame(
			data= data,
			columns=self.dataframe.columns,
						)
		# writing back...
		self.dataframe = Normalizer.write_data(df, self.filepath)

		# self.dataframe = pd.concat([self.dataframe, df], axis=0, ignore_index=True)
		# self.dataframe.to_csv(self.filepath, index=False)
	





#########   computes normalizer for the session ##############    
   

	# def _compute_normalizer(self, session, bin_width=20, delay=0, n=100000):
	#     """Compute dist. of normalizer for correlations (repeatability of neural
	#     spikes), and return median."""
	#     # session = str(int(session))
	#     # dataset = NeuralData(session)
	#     dataset = self._get_dataset_obj(session)

	#     sent_wise_repeated_spikes = {}
	#     for sent in self.metadata.test_sent_IDs:
	#         sent_wise_repeated_spikes[sent] = dataset.get_repeated_trials(
	#             sents=[sent], bin_width=bin_width, delay=delay
	#             )
			
	#     normalizer_all = Normalizer.inter_trial_corr_using_random_pairing(
	#         sent_wise_repeated_spikes, n=n
	#         )
	#     normalizer_all_med = np.median(normalizer_all, axis=0)
	#     return normalizer_all_med, normalizer_all
	
 



	def _get_dataset_obj(self, session):
		"""Retrieves dataset object"""
		session = str(int(session))
		if session not in self.loaded_datasets.keys():
			self.loaded_datasets.clear()
			self.loaded_datasets[session] = NeuralData(session)
		return self.loaded_datasets[session]
	
	def compute_normalizer_threshold(self, bin_width, p_value=5, itr=10000):
		"""Computes significance threshold for normalizer at bin_width."""

		total_samples_test_set = 0
		if bin_width == 1000:
			# special case...
			total_samples_test_set = self.metadata.test_sent_IDs.size
		else:
			for sent in self.metadata.test_sent_IDs:
				total_samples_test_set += self.metadata.stim_samples(sent, bin_width=bin_width)

		print(f"Computing null distribution for bin_width: {bin_width}, num_samples: {total_samples_test_set}...")
		null_dist = []
		for i in range(itr):
			gaussian_sample_of_same_length = np.random.randn(2, total_samples_test_set)
			null_dist.append(np.corrcoef(gaussian_sample_of_same_length)[0,1])

		q = 100 - p_value
		return np.percentile(null_dist, q), null_dist
	
	### -------------------------------------------------------------------------- ###
	###       Computing the distribution of Normalizers (repeated trials)          ###
	### -------------------------------------------------------------------------- ###


	# --------   Method 1: computes normalizer separately for each channel   ------- #
	# --------                  using random pairs of trials.                ------- #

	def _compute_normalizer_using_random_pairs(
			self, session, bin_width=20, delay=0, n=100000, mVocs=False):
		"""Compute dist. of normalizer for correlations (repeatability of neural
		spikes), using 100k runs with random concatenation of trials at each iteration.."""
		# session = str(int(session))
		# dataset = NeuralData(session)
		dataset = self._get_dataset_obj(session)
		
		if mVocs:
			stim_wise_repeated_spikes = {}
			mVoc_ids = self.metadata.mVoc_test_stimIds
			for mVoc in mVoc_ids:
				stim_wise_repeated_spikes[mVoc] = dataset.get_repeated_mVoc_trials(
					[mVoc], bin_width=bin_width, delay=delay
					)
		else: 

			stim_wise_repeated_spikes = {}
			for sent in self.metadata.test_sent_IDs:
				stim_wise_repeated_spikes[sent] = dataset.get_repeated_trials(
					sents=[sent], bin_width=bin_width, delay=delay
					)
			
		normalizer_all = Normalizer.inter_trial_corr_using_random_pairing(
			stim_wise_repeated_spikes, n=n, mVocs=mVocs
			)
		# remove the nan entries from the distribution..
		normalizer_all = np.delete(normalizer_all, np.where(np.isnan(normalizer_all))[0], axis=0)
		return normalizer_all
	

	def get_normalizer_for_session_random_pairs(
			self, session, bin_width=20, delay=0, force_redo=False, mVocs=False,
			verbose=False):
		"""Retrieves the distribution of normalizer for all the channels 
		of the specified session, at specified bin_width and delay.
		"""
		session = session = str(int(float(session)))    # to make sure session is in str format.
		bin_width = int(bin_width)
		if verbose:
			print(f"Getting normalizer dist. for sess-{session}, bw-{bin_width}, mVocs={mVocs}")
		
		if mVocs:
			# handling the excluded sessions for mVocs...
			if session == '190726':
				return np.zeros((1000, 60))
			elif session == '200213':
				return np.zeros((1000, 64))
		
		if not force_redo:
			norm_dict_all_sessions = io.read_normalizer_distribution(
				bin_width=bin_width, delay=delay, method='random', mVocs=mVocs
				)
		if force_redo or (norm_dict_all_sessions is None) or (session not in norm_dict_all_sessions.keys()):
			norm_dist_session = self._compute_normalizer_using_random_pairs(
				session, bin_width, delay=delay, mVocs=mVocs,
				)
			# save normalizer dist to disk..s
			io.write_normalizer_distribution(
				session, bin_width, delay, norm_dist_session, method='random', mVocs=mVocs
			)
			# return  norm_dist_session
		else:
			norm_dist_session =  norm_dict_all_sessions[session]

		# Redundant check: We have already penalized correlations for being NAN (zero sequence)
		# check zero valid samples in the distribution
		if norm_dist_session.shape[0] == 0:
			norm_dist_session = np.zeros((1, norm_dist_session.shape[-1]))
		return norm_dist_session


	def save_bootstrapped_normalizer(self, session, iterations=None, bin_width=50, n=1000):
		
		session = int(session)
		dataset = self._get_dataset_obj(session)

		stim_wise_repeated_spikes = {}
		for sent in self.metadata.test_sent_IDs:
			stim_wise_repeated_spikes[sent] = dataset.get_repeated_trials(
				sents=[sent], bin_width=bin_width, delay=0
				)
			
		if iterations is None:
			iterations = [1]
		percent_durations = [25, 50, 75, 100]
		num_trials_list = np.arange(3, 12)
		for itr in iterations:
			for percent_dur in percent_durations:
				for num_trials in num_trials_list:
					stim_ids = self.get_test_set_ids(percent_dur)

					normalizer_all = Normalizer.inter_trial_corr_for_bootstrap_analysis(
						stim_wise_repeated_spikes, stim_ids, n=n, num_trials=num_trials)

					normalizer_all = np.delete(normalizer_all, np.where(np.isnan(normalizer_all))[0], axis=0)
					io.write_bootstrap_normalizer_dist(
						normalizer_all, session, itr, percent_dur, num_trials, bin_width
					)


	# --------   Method 2: computes normalizer separately for each channel   -------- #
	# --------                  using all possible pairs                     -------- #

	def _compute_normalizer_all_possible_pairs(self, session, bin_width, delay=0):
		"""Computes correlations separately for each sentence and using 
		all possible trial pairs. Returns the distribution comprising of 
		data points for all channels.    
		Args:
			session: int = session ID to compute the normalizer for.

		Returns:
			corr_dist: ndarray = shape = (all_possible_pairs*num_sents, ch) 
		"""
		dataset = self._get_dataset_obj(session)
		sent_ids = self.metadata.test_sent_IDs
		corr_dist_combined = []
		for sent_id in sent_ids:
			repeated_trials_sent = dataset.get_repeated_trials(
				sents=[sent_id], bin_width=bin_width,
				delay=delay
				)
			corr_dist_sent = Normalizer.inter_trial_corr_all_possible_pairs(repeated_trials_sent)
			corr_dist_combined.append(corr_dist_sent)
		corr_dist_combined = np.concatenate(corr_dist_combined, axis=0)
		corr_dist_combined = np.nan_to_num(corr_dist_combined)
		return corr_dist_combined


	def get_normalizer_for_session_app(self, session, bin_width=20, delay=0, force_redo=False):
		"""Retrieves the distribution of normalizer for all the channels 
		of the specified session, at specified bin_width and delay.
		"""
		session = session = str(int(float(session)))    # to make sure session is in str format.
		bin_width = int(bin_width)
		norm_dict_all_sessions = io.read_normalizer_distribution(
			bin_width=bin_width, delay=delay, method='app',
			)
		if (norm_dict_all_sessions is None) or (session not in norm_dict_all_sessions.keys()) or force_redo:
			norm_dist_session = self._compute_normalizer_all_possible_pairs(
				session, bin_width, delay=delay
				)
			# save normalizer dist to disk..s
			io.write_normalizer_distribution(
				session, bin_width, delay, norm_dist_session, method='app',
			)
			return  norm_dist_session
		else:
			return norm_dict_all_sessions[session]
	

	### --------------------------------------------------------------------------###
	###         Computing the Null distribution of Normalizers                    ###
	### --------------------------------------------------------------------------###

	# --------  Method 1 Null distribution: Using random poisson sequences   ------ #

	def _compute_normalizer_null_dist_using_poisson(
			self, bin_width, spike_rate=50, p_value=5, itr=10000, mVocs=False
		):
		"""Normalizer based on assumption that spikes are generated by a poisson process, 
		and are uniformly distributed in time."""
		print(f"Poisson Process: Null distribution for bin_width: {bin_width}, spike_rate: {spike_rate}...")
		test_duration = self.metadata.get_total_test_duration(mVocs)
		total_spikes = int(spike_rate * test_duration)
		num_bins = int(np.ceil(round(1000*test_duration/bin_width, 3)))
		null_dist = []
		for i in range(itr):
			spike_times_1 = np.random.uniform(0, test_duration, int(total_spikes))
			counts_1,_ = np.histogram(spike_times_1, num_bins)

			spike_times_2 = np.random.uniform(0, test_duration, int(total_spikes))
			counts_2, _ = np.histogram(spike_times_2, num_bins)

			null_dist.append(np.corrcoef(counts_1, counts_2)[0,1])
	   
		q = 100 - p_value
		return np.percentile(null_dist, q), np.array(null_dist)
	

	def get_normalizer_null_dist_using_poisson(
			self, bin_width, spike_rate=50, itr=10000, force_redo=False,
			mVocs=False
		):
		"""Retrieves null distribution for normalizer using poisson
		sequences. Reads off the disk, if already available, or recomputes
		otherwise..
		
		Args:
			bin_width: int = bin width in ms
			spike_rate: int = spikes per second (Hz)
			itr: int = number of iterations.
			mVocs: bool = If True, 
		"""
		if not force_redo:
			null_dist_poisson = io.read_normalizer_null_distribution_using_poisson(
				bin_width=bin_width, spike_rate=spike_rate, mVocs=mVocs
				)
		if force_redo or null_dist_poisson is None:
			threshold, null_dist_poisson = self._compute_normalizer_null_dist_using_poisson(
				bin_width, spike_rate, itr=itr, mVocs=mVocs
			)
			io.write_normalizer_null_distribution_using_poisson(
				bin_width, spike_rate, null_dist_poisson, mVocs=mVocs
			)
		return null_dist_poisson

	def _compute_significant_sessions_and_channels_using_poisson_null(
			self, bin_width = 50, spike_rate = 50, p_threshold = 0.01, mVocs=False
		):
		"""Computes significant channels for all the sessions, using
		two-sample Welch's t-test (assumes unqual variances).
		It test the null hypothesis that two distributions, distribution of repeatability
		correlations (normalizer dist.) and distribution of correlation between
		random poisson sequences (null dist.) have equal means.

		Args:
			bin_width: int = bin width in ms.
			spike_rate: int = spikes/second for poisson sequences in Null dist.
			p_threshold: float = threshold used to decide if null hypothesis if False.

		Returns:
			dict: {session: np.array([ch, ch])}
		"""
		total_sig_channels = 0
		sessions = self.metadata.get_all_available_sessions()
		significant_sessions_and_channels = {}
		null_dist = self.get_normalizer_null_dist_using_poisson(
			bin_width=bin_width, spike_rate=spike_rate, mVocs=mVocs
			)
		null_dist = np.array(null_dist)
		for session in sessions:
			norm_dist = self.get_normalizer_for_session_random_pairs(
				session=session, bin_width=bin_width, mVocs=mVocs 
			)
			num_channels = self.metadata.get_num_channels(session)
			for ch in range(num_channels):
				# pvalue = scipy.stats.ttest_ind(
				#     norm_dist[:,ch], null_dist, equal_var=False, alternative='greater'
				#     ).pvalue
				pvalue = scipy.stats.mannwhitneyu(
					norm_dist[:,ch], null_dist,alternative='greater'
					).pvalue

				# significance condition..
				if pvalue < p_threshold:    
					total_sig_channels += 1
					if session in significant_sessions_and_channels.keys():
						significant_sessions_and_channels[session].append(ch)
					else:
						significant_sessions_and_channels[session] = [ch]
		print(f"Total significant neurons at {bin_width}ms bin width = {total_sig_channels}")
		return significant_sessions_and_channels

	def get_significant_sessions_and_channels_using_poisson_null(
			self, bin_width = 50, spike_rate=50, p_threshold=0.05, force_redo=False,
			mVocs=False
		):
		"""Retrieves significant channels for all the sessions, using
		two-sample Welch's t-test (assumes unqual variances).
		Reads off the disk and recomputes if not found.

		Args:
			bin_width: int = bin width in ms.
			
		Returns:
			dict: {session: np.array([ch, ch])}
		"""
		significant_sessions_and_channels = io.read_significant_sessions_and_channels(
			bin_width=bin_width, p_threshold=p_threshold, mVocs=mVocs
			)
		
		if significant_sessions_and_channels is None or force_redo:
			significant_sessions_and_channels = self._compute_significant_sessions_and_channels_using_poisson_null(
				bin_width=bin_width, spike_rate=spike_rate, p_threshold=p_threshold, mVocs=mVocs
			)

			io.write_significant_sessions_and_channels(
				bin_width=bin_width,
				p_threshold=p_threshold,
				significant_sessions_and_channels=significant_sessions_and_channels,
				mVocs=mVocs
			)
		return significant_sessions_and_channels



	
	# ----------  Method 2 Null distribution: Randomly shifted seq.    ---------- #

	def _compute_normalizer_null_dist_using_random_shifts(
			self, session, bin_width=50, n_itr=100000, min_shift_frac=0.2, max_shift_frac=0.8
			):
		"""Computes Null distribution by correlating randomly shifted spike sequence for 
		randomly choosen trial against the original spike sequence (not shifted).
		At each iteration shift is randomly choosen from [min_shift_samples, max_shift_samples],
		where shift_sample = seq_length * shift_fraction
		
		Args:
			session: int = session ID
			bin_width: int = bin width in ms.
			n_itr: int = number of iterations of correlation computation
			min_shift_frac: float = fraction of sequence length for min_shift_samples
			max_shift_frac: float = fraction of sequence length for max_shift_samples
		"""

		dataset = self._get_dataset_obj(session)
		print(f"Loading spikes for all trial presentations...")
		all_trial_spikes = dataset.get_repeated_trials(
			sents=self.metadata.test_sent_IDs, bin_width=bin_width, delay=0
			)

		num_trials, seq_len, num_channels = all_trial_spikes.shape
		null_dist = np.zeros((n_itr, num_channels))
		min_shift = int(seq_len*min_shift_frac)
		max_shift = int(seq_len*max_shift_frac)
		for t in range(n_itr):
			
			tr_id1, tr_id2 = np.random.choice(
					np.arange(0, num_trials),
					size=2,
					replace=True
				)
			spk_seq1 = all_trial_spikes[tr_id1].squeeze()
			spk_seq2 = all_trial_spikes[tr_id2].squeeze()
			# randomly roll the sequence to get the second sequence
			rand_shift = np.random.randint(min_shift, max_shift)
			spk_seq2 = np.roll(spk_seq2, rand_shift, axis=0)

			for ch in range(num_channels):
				null_dist[t, ch] = np.corrcoef(
					spk_seq1[...,ch],
					spk_seq2[...,ch]
					)[0,1]
		# delete the entire iteration if there is nan for entry channel...
		null_dist = np.delete(null_dist, np.where(np.isnan(null_dist))[0], axis=0)
		return null_dist
	

	def get_normalizer_null_dist_using_random_shifts(
			self, session, bin_width=50, n_itr=100000, min_shift_frac=0.2, max_shift_frac=0.8,
			force_redo=False
		):
		"""Retrieves null distribution for normalizer using randomly 
		shifted spike sequence of a trial vs non-shifted sequence.
		At each iteration, trial is randomly choosen out of all available 
		trials.
		Reads off the disk, if already available, or recomputes
		otherwise..
		
		Args:
			bin_width: int = bin width in ms
			spike_rate: int = spikes per second (Hz)
			itr: int = number of iterations.
		"""
		session = str(int(float(session)))  # enforcing exact format of session
		null_dist_shifts = io.read_normalizer_null_distribution_random_shifts(
			bin_width=bin_width, min_shift_frac=min_shift_frac,
			max_shift_frac=max_shift_frac
			)
		if null_dist_shifts is None or session not in null_dist_shifts.keys() or force_redo:
			null_dist_sess = self._compute_normalizer_null_dist_using_random_shifts(
				session=session, bin_width=bin_width, n_itr=n_itr,
				min_shift_frac=min_shift_frac, max_shift_frac=max_shift_frac
			)
			io.write_normalizer_null_distribution_using_random_shifts(
				session, bin_width=bin_width, min_shift_frac=min_shift_frac,
				max_shift_frac=max_shift_frac, null_dist_sess=null_dist_sess,
			)
			return null_dist_sess
		else:
			null_dist = null_dist_shifts[session]
			null_dist = np.delete(null_dist, np.where(np.isnan(null_dist))[0], axis=0)
			return null_dist
		
		


	def _compute_significant_sessions_and_channels_using_shifts_null(
			self, bin_width = 50, p_threshold = 0.05, min_shift_frac=0.2, max_shift_frac=0.8,
		):
		"""Computes significant channels for all the sessions, using
		two-sample Welch's t-test (assumes unqual variances).
		It test the null hypothesis that two distributions, distribution of repeatability
		correlations (normalizer dist.) and distribution of correlation between
		spikes sequence and its randomly shifted version (null dist.) have equal means.

		Args:
			bin_width: int = bin width in ms.
			spike_rate: int = spikes/second for poisson sequences in Null dist.
			p_threshold: float = threshold used to decide if null hypothesis if False.

		Returns:
			dict: {session: np.array([ch, ch])}
		"""
		total_sig_channels = 0
		sessions = self.metadata.get_all_available_sessions()
		significant_sessions_and_channels = {}
		# null_dist = self.get_normalizer_null_dist_using_poisson(
		#     bin_width=bin_width, spike_rate=spike_rate
		#     )
		# null_dist = np.array(null_dist)
		for session in sessions:
			null_dist = self.get_normalizer_null_dist_using_random_shifts(
				session=session, bin_width=bin_width,
				min_shift_frac=min_shift_frac, max_shift_frac=max_shift_frac
				)

			norm_dist = self.get_normalizer_for_session_app(
				session=session, bin_width=bin_width 
			)
			num_channels = self.metadata.get_num_channels(session)
			for ch in range(num_channels):
				pvalue = scipy.stats.ttest_ind(
					norm_dist[:,ch], null_dist[:,ch], equal_var=False, alternative='greater'
					).pvalue
				# significance condition..
				if pvalue < p_threshold:    
					total_sig_channels += 1
					if session in significant_sessions_and_channels.keys():
						significant_sessions_and_channels[session].append(ch)
					else:
						significant_sessions_and_channels[session] = [ch]
		print(f"Total significant neurons at {bin_width}ms bin width = {total_sig_channels}")
		return significant_sessions_and_channels
	


	def get_significant_sessions_and_channels_using_shifts_null(
			self, bin_width = 50, p_threshold=0.05, force_redo=False
		):
		"""Retrieves significant channels for all the sessions, using
		two-sample Welch's t-test (assumes unqual variances).
		Reads off the disk and recomputes if not found.

		Args:
			bin_width: int = bin width in ms.
			
		Returns:
			dict: {session: np.array([ch, ch])}
		"""
		significant_sessions_and_channels = io.read_significant_sessions_and_channels(
			bin_width=bin_width, p_threshold=p_threshold, use_poisson_null=False
			)
		
		if significant_sessions_and_channels is None or force_redo:
			significant_sessions_and_channels = self._compute_significant_sessions_and_channels_using_shifts_null(
				bin_width=bin_width, p_threshold=p_threshold,
				min_shift_frac=0.2, max_shift_frac=0.8, # default values..
			)

			io.write_significant_sessions_and_channels(
				bin_width=bin_width,
				p_threshold=p_threshold,
				significant_sessions_and_channels=significant_sessions_and_channels,
				use_poisson_null=False
			)
		return significant_sessions_and_channels


	def get_test_set_ids(self, percent_duration=None):
		"""Returns a smaller mapping set of given size."""
		stim_ids = self.metadata.test_sent_IDs
		np.random.shuffle(stim_ids)
		if percent_duration is None or percent_duration >= 100:	
			return stim_ids
		else:
			required_duration = percent_duration*16/100
			print(f"Stim ids for duration={required_duration:.2f} sec")
			stimili_duration=0
			for n in range(len(stim_ids)):
				stimili_duration += self.metadata.stim_duration(stim_ids[n])
				if stimili_duration >= required_duration:
					break
		print(f"Total duration={stimili_duration:.2f} sec")
		return stim_ids[:n+1]

		

	
#########################################################
######      static methods
########################################################

	@staticmethod
	def write_data(df, filepath):
		if os.path.exists(filepath):
			existing_data = pd.read_csv(filepath)
		else:
			existing_data = pd.DataFrame(columns=df.columns)
		updated_data = pd.concat([existing_data, df], axis=0, ignore_index=True)
		updated_data.to_csv(filepath, index=False)
		print(f"Result updated...!")
		return updated_data
		


	@staticmethod
	def load_data(filename):
		"""Loads/creates dataframe containing normalizers"""
		filepath = os.path.join(
			results_dir, 'neural_repeatibitliy', filename
			)
		if not os.path.exists(os.path.dirname(filepath)):
			print(f"Creating directory structure for normalizer...!")
			os.makedirs(os.path.dirname(filepath))

		if os.path.exists(filepath):
			print(f"Reading existing dataframe.")
			dataframe = pd.read_csv(filepath)
		else:
			print(f"Creating new dataframe.")
			columns = ['session', 'channel','bin_width',
					   'delay', 'normalizer']
			dataframe = pd.DataFrame(columns=columns)
		return dataframe, filepath
	
	@staticmethod
	def inter_trial_corr_all_possible_pairs(spikes_all_trials, num_trials=None):
		"""Computes correlations between all possible trial pairs,
		and returns the distribution. For example for input of shape (11,:, 64),
		there are 11 trials, so number of possible pairs are 11C2 = 11*10/2 = 55.
		
		Args:
			spikes_all_trials: ndarray = (num_trials, seq_length, channels)

		Returns:
			sent_corr_dist: ndarray = shape = (all_possible_pairs, ch) 
		"""
		if num_trials is None:
			num_trials = spikes_all_trials.shape[0]
		sent_corr_dist = []
		num_channels = spikes_all_trials.shape[-1]
		# imagine upper half of the matrix, with diagonal entries included..
		possible_pairs = int((num_trials**2 - num_trials)/2)
		sent_corr_dist = np.zeros((possible_pairs, num_channels))
		index = 0
		for i in range(num_trials):
			for j in range(i+1, num_trials):
				for ch in range(num_channels):
					trial1 = spikes_all_trials[i,:,ch]
					trial2 = spikes_all_trials[j,:,ch]
					sent_corr_dist[index, ch] = np.corrcoef(
						trial1, trial2
						)[0,1]
				index += 1
		return sent_corr_dist

	@staticmethod
	def inter_trial_corr(spikes, n=100000):
		"""Compute distribution of inter-trials correlations.

		Args: 
			spikes (ndarray): (repeats, samples/time, channels)

		Returns:
			trials_corr (ndarray): (n, channels) distribution of inter-trial correlations
		"""
		num_channels = spikes.shape[-1]
		trials_corr = np.zeros((n, num_channels))
		for t in range(n):
			trials = np.random.choice(np.arange(0,spikes.shape[0]), size=2, replace=False)
			for ch in range(num_channels):
				trials_corr[t, ch] = np.corrcoef(
					spikes[trials[0],:,ch].squeeze(), spikes[trials[1],:,ch].squeeze()
					)[0,1]
		return trials_corr

	@staticmethod
	def inter_trial_corr_using_random_pairing(sent_wise_repeated_spikes, n=100000, mVocs=False):
		"""Compute distribution of inter-trials correlations, using bootstrapping.
		At each iteration randomly selects trial pair for each sentence. Assigns one
		trial to first long sequence and second trial to second long sequence. 

		Args: 
			spikes (ndarray): (repeats, samples/time, channels)

		Returns:
			trials_corr (ndarray): (n, channels) distribution of inter-trial correlations
		"""
		num_channels = next(iter(sent_wise_repeated_spikes.values())).shape[-1]
		trials_corr = np.zeros((n, num_channels))
		if mVocs:
			trial_ids = np.arange(15)
		else:
			trial_ids = np.arange(11)
		for t in range(n):
			long_seq1 = []
			long_seq2 = []

			for sent, spikes in sent_wise_repeated_spikes.items():
				tr1, tr2 = np.random.choice(trial_ids, size=2, replace=False)

				long_seq1.append(spikes[tr1])
				long_seq2.append(spikes[tr2])
			long_seq1 = np.concatenate(long_seq1, axis=0)
			long_seq2 = np.concatenate(long_seq2, axis=0)

			for ch in range(num_channels):
				corr_ch = np.corrcoef(
					long_seq1[...,ch].squeeze(), long_seq2[...,ch].squeeze()
					)[0,1]
				# nan might result from entires sequence being zero, 
				# penalize that by setting corr equal to zero.
				if np.isnan(corr_ch):
					corr_ch = 0
				trials_corr[t, ch] = corr_ch
		return trials_corr
	
	@staticmethod
	def inter_trial_corr_for_bootstrap_analysis(
		sent_wise_repeated_spikes, stim_ids=None, n=10000, mVocs=False, num_trials=3, 
		):

		if stim_ids is None:
			stim_ids = list(sent_wise_repeated_spikes.keys())
		num_channels = next(iter(sent_wise_repeated_spikes.values())).shape[-1]
		trials_corr = np.zeros((n, num_channels))
		if mVocs:
			max_num_trials = 15
			
		else:
			max_num_trials = 11
		assert num_trials <= max_num_trials and num_trials > 2, "num_trials must be between 2 and {}".format(max_num_trials)
		trial_ids = np.arange(max_num_trials)
		trial_ids = np.random.choice(trial_ids, size=num_trials, replace=False)

		for t in range(n):
			long_seq1 = []
			long_seq2 = []

			for stim_id in stim_ids:
				spikes = sent_wise_repeated_spikes[stim_id]
				tr1, tr2 = np.random.choice(trial_ids, size=2, replace=False)

				long_seq1.append(spikes[tr1])
				long_seq2.append(spikes[tr2])
			long_seq1 = np.concatenate(long_seq1, axis=0)
			long_seq2 = np.concatenate(long_seq2, axis=0)

			for ch in range(num_channels):
				corr_ch = np.corrcoef(
					long_seq1[...,ch].squeeze(), long_seq2[...,ch].squeeze()
					)[0,1]
				# nan might result from entires sequence being zero, 
				# penalize that by setting corr equal to zero.
				if np.isnan(corr_ch):
					corr_ch = 0
				trials_corr[t, ch] = corr_ch
		return trials_corr


