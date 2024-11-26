# import math
import os
import numpy as np
from scipy.signal import resample
from abc import ABC, abstractmethod
import gc
import naplib as nl

from auditory_cortex.dataloader2 import DataLoader
from transformers import Speech2TextProcessor
from auditory_cortex import CACHE_DIR

import logging
logger = logging.getLogger(__name__)

HF_CACHE_DIR = os.path.join(CACHE_DIR, 'hf_cache')
   
class BaseDataAssembler(ABC):
	"""Base class for all the datasets."""
	def __init__(
			self, 
			bin_width,
			dataset_obj, 
			feature_extractor=None,
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
		self.dataloader = DataLoader(dataset_obj, feature_extractor)

		self.bin_width = bin_width
		self.mVocs=mVocs
		# self.dataloader = DataLoader()
		self.LPF = LPF
		self.LPF_analysis_bw = LPF_analysis_bw
		if self.LPF:
			logger.info(f"creating Dataset for LPF features, to predict at {self.LPF_analysis_bw}ms.")

		if self.mVocs:
			logger.info(f"creating Dataset for mVocs data.")
		else:
			logger.info(f"creating Dataset for timit data.")
		self.fs = self.dataloader.get_sampling_rate(mVocs=mVocs)
		self.training_stim_ids = self.dataloader.get_training_stim_ids(mVocs=self.mVocs)
		self.testing_stim_ids = self.dataloader.get_testing_stim_ids(mVocs=self.mVocs)

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


	def load_sent_wise_features_and_spikes(self):
		"""Reads data for all the sents (having single presentation)
		return the features (spectrogram) and spike pairs.
		"""
		training_spikes, testing_spikes = self.load_neural_spikes()
		layer_features = self.load_features()
		
		data_cache = {
			'features': layer_features,
			'training_spikes': training_spikes,
			'testing_spikes': testing_spikes
		}
		# get the number of channels
		stim_ids = list(training_spikes.keys())
		channel_ids = list(training_spikes[stim_ids[0]].keys())
		# num_channels = self.dataloader.get_num_channels(self.mVocs)
		return data_cache, channel_ids
	
	def load_neural_spikes(self):
		"""load neural spikes for the given session."""
		if self.LPF:
			bin_width = self.LPF_analysis_bw
		else:
			bin_width = self.bin_width
		logger.info(f"Loading data for session at bin_width-{bin_width}ms.")
		training_spikes = self.dataloader.get_session_spikes(
			bin_width=bin_width,
			delay=0,
			repeated=False,
			mVocs=self.mVocs
			)
		testing_spikes = self.dataloader.get_session_spikes(
			bin_width=bin_width,
			delay=0,
			repeated=True,
			mVocs=self.mVocs
			)
		return training_spikes, testing_spikes


	def get_training_data(self, stim_ids=None):
		"""Returns spectral-features, spikes (all trials) for the test sent IDs.
			
		Returns:
			features_list: list = [(time, num_dnn_units)] each entry of list is a
				feature for stim_id.
			spikes_list: list = [(time, channels)] all trials concatenated along time axis.
		"""
		if stim_ids is None:
			stim_ids = self.training_stim_ids
		
		features = self.data_cache['features']
		training_spikes = self.data_cache['training_spikes']
		features_list = []
		spikes_list = []

		for stim in stim_ids:
			features_list.append(features[stim])
			# each ch_spikes has shape (n_trial, time), for unique stimuli n_trial=1
			# np.stack([spikes for spikes in training_spikes[1].values()], axis=-1)
			stim_spikes = np.stack([ch_spikes for ch_spikes in training_spikes[stim].values()], axis=-1).squeeze()
			spikes_list.append(stim_spikes)
		
		return features_list, spikes_list
	
	def get_testing_data(self, stim_ids=None):
		"""Returns spectral-features, spikes (all trials) for the test sent IDs.
			
		Returns:
			features_list: list = [(time, channels)] each entry of list is a
				feature for stim_id.
			repeated_spikes_list: ndarray = (num_repeats, time, channels) all trials concatenated along time axis.
		"""
		if stim_ids is None:
			stim_ids = self.testing_stim_ids
		
		features = self.data_cache['features']
		testing_spikes = self.data_cache['testing_spikes']
		features_list = []
		spikes_list = []

		for stim in stim_ids:
			features_list.append(features[stim])
			# each ch_spikes has shape (n_trial, time), for unique stimuli n_trial=num_repeats
			stim_spikes = np.stack([ch_spikes for ch_spikes in testing_spikes[stim].values()], axis=-1).squeeze()
			spikes_list.append(stim_spikes)
		return features_list, spikes_list



class STRFDataAssembler(BaseDataAssembler):
	def __init__(
			self, dataset_obj, bin_width, mVocs=False,
			mel_spectrogram=False,
			num_freqs=80):
		super().__init__(dataset_obj, bin_width, mVocs=mVocs)
		self.mel_spectrogram = mel_spectrogram
		if self.mel_spectrogram:
			self.processor = Speech2TextProcessor.from_pretrained(
				"facebook/s2t-large-librispeech-asr", cache_dir=HF_CACHE_DIR
				)
			logger.info(f"Using mel-spectrogram for STRF.")
		else:
			logger.info(f"Using wavelet-spectrogram for STRF.")
		self.num_freqs = num_freqs # num_freqs in the spectrogram
		
		self.data_cache, self.channel_ids = self.load_sent_wise_features_and_spikes()
		self.num_channels = len(self.channel_ids)
	def load_features(self):
		"""Loads spectrogram features for the given session."""
		sampling_rate = self.dataloader.get_sampling_rate(mVocs=self.mVocs)
		
		all_stim_ids = np.concatenate([self.training_stim_ids, self.testing_stim_ids])
		spect_features = {}

		for stim_id in all_stim_ids:

			aud = self.dataloader.get_stim_audio(stim_id, mVocs=self.mVocs)
			num_bins = self.dataloader.get_num_bins(stim_id, bin_width=self.bin_width, mVocs=self.mVocs)
			spect = self.get_spectrogram(aud, sampling_rate)
			spect = resample(spect, num_bins, axis=0)
			spect = resample(spect, self.num_freqs, axis=1)

			spect_features[stim_id] = spect

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
	

class DNNDataAssembler(BaseDataAssembler):
	def __init__(
			self, dataset_obj, feature_extractor, 
			layer_id, bin_width, 
			mVocs=False,
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
			bin_width, dataset_obj, feature_extractor, mVocs=mVocs,
			LPF=LPF, LPF_analysis_bw=LPF_analysis_bw
			)       

		self.model_name = self.dataloader.feature_extractor.model_name
		self.force_reload = force_reload
		self.layer_id = layer_id

		self.data_cache, self.channel_ids = self.load_sent_wise_features_and_spikes()
		self.num_channels = len(self.channel_ids)
		# free up memory
		del self.dataloader.DNN_feature_dict
		del self.dataloader.DNN_shuffled_feature_dict
		del self.dataloader.neural_spikes
		gc.collect()  # Force garbage collection

	def load_features(self):
		"""loads the DNN features for the given session."""
		all_layer_features = self.dataloader.get_resampled_DNN_features(
			bin_width=self.bin_width, mVocs=self.mVocs, 
			LPF=self.LPF, LPF_analysis_bw=self.LPF_analysis_bw,
			force_reload=self.force_reload,
			)
		layer_features = all_layer_features[self.layer_id]
		return layer_features
