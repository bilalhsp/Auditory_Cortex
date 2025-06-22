# import math
import numpy as np
from scipy.signal import resample
from abc import ABC, abstractmethod
import gc
import naplib as nl
# from auditory_cortex import config
# from auditory_cortex.neural_data import dataset
from auditory_cortex.deprecated.dataloader import DataLoader
# from auditory_cortex import utils
# from sklearn.linear_model import RidgeCV, ElasticNet, Ridge, PoissonRegressor
from transformers import Speech2TextProcessor
# from auditory_cortex.utils import SyntheticInputUtils
	
   
class BaseDataset(ABC):
	"""Base class for all the datasets."""
	def __init__(
			self, session, bin_width,
			mVocs=False,
			LPF=False,
			LPF_analysis_bw=20
			):
		"""
		Args:
			model_name:
			session: int = session ID
			bin_width: int = bin width in ms
		"""        
		self.session = str(int(session))
		self.bin_width = bin_width
		self.mVocs=mVocs
		self.dataloader = DataLoader()
		self.LPF = LPF
		self.LPF_analysis_bw = LPF_analysis_bw
		if self.LPF:
			print(f"creating Dataset for LPF features, to predict at {self.LPF_analysis_bw}ms.")

		if self.mVocs:
			print(f"{self.session}: creating DNNDataset for mVocs data..")
		else:
			print(f"{self.session}: creating DNNDataset for timit data..")
		self.fs = self.dataloader.metadata.get_sampling_rate(mVocs=mVocs)

		self.training_sent_ids, self.all_trial_ids = self.get_training_stim_ids(mVocs=self.mVocs)
		# self.data_cache, self.num_channels = self.load_sent_wise_features_and_spikes()

	@abstractmethod
	def load_features(self):
		"""loads the features for the given session."""
		...
	
	def get_bin_width(self):
		"""Returns the bin width (sampling rate) for the dataset."""
		if self.LPF:
			return self.LPF_analysis_bw
		else:   
			return self.bin_width

	def get_stimuli_duration(self, stim_ids, mVocs=False):
		"""Returns the total duration of the stimuli in seconds."""
		return self.dataloader.metadata.get_total_stimuli_duration(stim_ids, mVocs=mVocs)

	def get_training_stim_ids(self, mVocs=False):
		"""Returns the stim ids for training set.
		
		Args:
			mVocs: bool = If True, returns ids for mVocs,
				otherwise for timit stimuli.
		"""
		if mVocs:
			stim_ids = self.dataloader.metadata.mVocTrialIds
			# exclude the missing trial IDs from list of Ids
			missing_trial_ids = self.dataloader.get_dataset_object(session=self.session).missing_trial_ids
			stim_ids = stim_ids[np.isin(stim_ids, missing_trial_ids, invert=True)]
			test_ids = self.dataloader.metadata.mVoc_test_trIds
			training_stim_ids = stim_ids[np.isin(stim_ids, test_ids, invert=True)]
			all_trial_ids = stim_ids
		else:
			sent_IDs = self.dataloader.sent_IDs
			testing_sent_ids = self.dataloader.test_sent_IDs
			training_stim_ids = sent_IDs[np.isin(sent_IDs, testing_sent_ids, invert=True)]
			all_trial_ids = sent_IDs
		return training_stim_ids, all_trial_ids

	def get_testing_stim_ids(self, mVocs=False):
		"""Returns the stim ids for testing set.
		
		Args:
			session: int = session ID, needed ONLY if mVocs=True.
			mVocs: bool = If True, returns ids for mVocs,
				otherwise for timit stimuli.
		"""
		if mVocs:
			testing_stim_ids = self.dataloader.metadata.mVoc_test_stimIds
		else:
			testing_stim_ids = self.dataloader.test_sent_IDs
		return testing_stim_ids


	def load_sent_wise_features_and_spikes(self):
		"""Reads data for all the sents (having single presentation)
		return the features (spectrogram) and spike pairs.
		"""
		raw_spikes = self.load_neural_spikes()
		layer_features = self.load_features()
		
		data_cache = {
			'features': layer_features,
			'spikes': raw_spikes,
		}
		num_channels = self.dataloader.get_num_channels(self.session)
		return data_cache, num_channels
	
	def load_neural_spikes(self):
		"""load neural spikes for the given session."""
		if self.LPF:
			bin_width = self.LPF_analysis_bw
		else:
			bin_width = self.bin_width
		print(f"DNNDataset: Loading data for session-{self.session} at bin_width-{bin_width}ms.")
		raw_spikes = self.dataloader.get_session_spikes(
			session=self.session,
			bin_width=bin_width,
			delay=0,
			mVocs=self.mVocs
			)
		return raw_spikes

	def get_data(self, stim_ids=None):
		"""Returns spectral-features, spikes pair for the 
		given sent IDs.
		"""
		if stim_ids is None:
			stim_ids = self.training_sent_ids
		
		features = self.data_cache['features']
		spikes = self.data_cache['spikes']
		features_list = []
		spikes_list = []

		for stim in stim_ids:
			features_list.append(features[stim])
			spikes_list.append(spikes[stim])
		
		return features_list, spikes_list

	
	def get_test_data(self):
		"""Returns spectral-features, spikes (all trials)
		for the test sent IDs, given session.
			
		Returns:
			features_list: list = [(time, channels)] each entry of list is a feature for on stim_id. 
				for a sent audio.
			repeated_spikes_list: ndarray = (num_repeats, time, channels) all trials concatenated along time axis.
		"""
		features_list = []
		# repeated_spikes_list = {i: [] for i in range(11)}
		all_sent_spikes = []
		testing_stim_ids = self.get_testing_stim_ids(mVocs=self.mVocs)
		features = self.data_cache['features']
		if self.LPF:
			bin_width = self.LPF_analysis_bw
		else:
			bin_width = self.bin_width
		for stim in testing_stim_ids:
			# make sure to drop the partial sample at the end, 
			# this has already been done for repeated trials..
			if self.mVocs:
				tr_id = self.dataloader.metadata.get_mVoc_tr_id(stim)[0]
				features_list.append(features[tr_id])
			else:
				features_list.append(features[stim])
			repeated_spikes = self.dataloader.get_neural_data_for_repeated_trials(
				session=self.session,
				bin_width=bin_width,
				delay=0,
				stim_ids=[stim],
				mVocs=self.mVocs
				)
			all_sent_spikes.append(repeated_spikes)
		all_sent_spikes = np.concatenate(all_sent_spikes, axis=1)
		return features_list, all_sent_spikes


class BaselineDataset(BaseDataset):
	def __init__(
			self, session, bin_width, mVocs=False,
			mel_spectrogram=False,
			num_freqs=80):
		super().__init__(session, bin_width, mVocs=mVocs)
		self.mel_spectrogram = mel_spectrogram
		if self.mel_spectrogram:
			self.processor = Speech2TextProcessor.from_pretrained("facebook/s2t-large-librispeech-asr")
			print(F"Using mel-spectrogram for STRF.")
		else:
			print(F"Using wavelet-spectrogram for STRF.")
		self.num_freqs = num_freqs # num_freqs in the spectrogram
		
		self.data_cache, self.num_channels = self.load_sent_wise_features_and_spikes()
		
	def load_features(self):
		"""Loads spectrogram features for the given session."""
		sampling_rate = self.dataloader.metadata.get_sampling_rate(mVocs=self.mVocs)
		spect_features = {}

		for stim in self.all_trial_ids:

			aud = self.dataloader.get_stim_aud(stim, mVocs=self.mVocs)
			spect = self.get_spectrogram(aud, sampling_rate)
			num_bins = self.dataloader.get_num_bins(stim, bin_width=self.bin_width, mVocs=self.mVocs)
			spect = resample(spect, num_bins, axis=0)
			spect = resample(spect, self.num_freqs, axis=1)

			spect_features[stim] = spect

		return spect_features

	def get_spectrogram(self, aud, sampling_rate):
		"""Transforms the given audio into the spectrogram"""
		# Getting the spectrogram at 10 ms and then resample to match the bin_width
		if sampling_rate != 16000:
			n_new = int(aud.size*16000/sampling_rate)
			aud = resample(aud, n_new)
		if self.mel_spectrogram:
			spect = self.processor(aud, padding=True, sampling_rate=16000).input_features[0]
		else:
			spect = nl.features.auditory_spectrogram(aud, 16000, frame_len=10)

		return spect
	




class DNNDataset(BaseDataset):
	def __init__(
			self, session, bin_width, model_name, layer_ID,
			mVocs=False,
			shuffled=False,
			force_reload=False,
			LPF=False,
			LPF_analysis_bw=20
			
			):
		"""
		Args:
			model_name:
			session: int = session ID
			bin_width: int = bin width in ms
		"""
		super().__init__(
			session, bin_width, mVocs=mVocs,
			LPF=LPF, LPF_analysis_bw=LPF_analysis_bw
			)       

		self.model_name = model_name
		self.shuffled = shuffled
		self.force_reload = force_reload
		self.layer_ID = layer_ID

		self.data_cache, self.num_channels = self.load_sent_wise_features_and_spikes()
		
		# free up memory
		del self.dataloader.DNN_feature_dict
		del self.dataloader.DNN_shuffled_feature_dict
		del self.dataloader.neural_spikes
		gc.collect()  # Force garbage collection

	def load_features(self):
		"""loads the DNN features for the given session."""
		all_layer_features = self.dataloader.get_resampled_DNN_features(
			self.model_name, bin_width=self.bin_width, force_reload=self.force_reload,
			shuffled=self.shuffled, mVocs=self.mVocs, 
			LPF=self.LPF, LPF_analysis_bw=self.LPF_analysis_bw
			)
		layer_features = all_layer_features[self.layer_ID]
		return layer_features
