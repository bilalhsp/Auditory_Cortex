import os
import struct
import scipy.io 
import numpy as np

from auditory_cortex import neural_data_dir, config, NEURAL_DATASETS

DATASET_NAME = NEURAL_DATASETS[1]
DATA_DIR = os.path.join(neural_data_dir, DATASET_NAME)

class UCDavisMetaData:
	def __init__(self):
		self.data_dir = DATA_DIR

		self.sampling_rate = 48000
		# reading timit metadata
		self.timit_meta_file = os.path.join(self.data_dir, 'recanzone_timit_details.mat')
		self.timit_meta = UCDavisMetaData.read_stim_meta(self.timit_meta_file)

		timit_mask = self.timit_meta['timit'].use.astype(bool)
		timit_durs = self.timit_meta['timit'].durSec[timit_mask]
		self.timit_ids = self.timit_meta['timit'].wfmName[timit_mask]
		self.timit_dur_dict = {stim_id: dur for stim_id, dur in zip(self.timit_ids, timit_durs)}
		# self.timit_audios = self.read_stim_audios(self.timit_ids, mVocs=False)
		self.timit_audios = {}	# no need to read in advance

		# reading mVocs metadata
		self.mVocs_meta_file = os.path.join(self.data_dir, 'MSL.mat')
		self.mVocs_meta = UCDavisMetaData.read_stim_meta(self.mVocs_meta_file)
		mVocs_mask = self.mVocs_meta['MSL'].useThisSound.astype(bool)
		mVocs_durs = self.mVocs_meta['MSL'].actualDur#[mVocs_mask]   #mVocs_mask is not right
		self.mVocs_ids = self.mVocs_meta['MSL'].WFMname#[mVocs_mask]
		self.mVocs_dur_dict = {stim_id: dur for stim_id, dur in zip(self.mVocs_ids, mVocs_durs)}
		# self.mVocs_audios = self.read_stim_audios(self.mVocs_ids, mVocs=True)
		self.mVocs_audios = {}	# no need to read in advance

	def get_stim_duration(self, stim_id, mVocs=False):
		"""Returns duration of the stimulus in seconds"""
		if mVocs:
			dur_dict = self.mVocs_dur_dict
		else:
			dur_dict = self.timit_dur_dict
		return dur_dict[stim_id]

	def get_stim_audio(self, stim_id, mVocs=False):
		"""Reads stim audio for the given stimulus id"""
		if mVocs:
			stim_audios = self.mVocs_audios
		else:
			stim_audios = self.timit_audios

		if stim_id not in stim_audios:
			if mVocs:
				stim_dir = os.path.join(self.data_dir, 'NIMH_Mvoc_WFM')
			else:
				stim_dir = os.path.join(self.data_dir, 'TIMIT_48000')
			stim_filepath = os.path.join(stim_dir, stim_id)
			stim_audio = UCDavisMetaData.read_wfm(stim_filepath)
			stim_audios[stim_id] = stim_audio
		return stim_audios[stim_id]

	def get_sampling_rate(self, mVocs=False):
		return self.sampling_rate


	def read_stim_audios(self, stim_ids, mVocs=False):
		"""Read the audio files for the given stimulus ids
		
		Args:
			stim_ids (list): List of stimulus ids to read
			mVocs (bool): If True, read the mVocs audio files, otherwise read the BMT3 audio files
		
		Returns:
			dict: Dictionary of stimulus ids and their corresponding audio waveforms
		"""
		if mVocs:
			stim_dir = os.path.join(self.data_dir, 'NIMH_Mvoc_WFM')
		else:
			stim_dir = os.path.join(self.data_dir, 'TIMIT_48000')
		stim_audios = {}
		for stim_id in stim_ids:
			stim_filepath = os.path.join(stim_dir, stim_id)
			stim_audio = self.read_wfm(stim_filepath)
			stim_audios[stim_id] = stim_audio
		return stim_audios

	@staticmethod
	def read_wfm(filename):
		"""
		Reads a WFM file containing 16-bit integer waveform data stored as binary,
		and returns the waveform as a NumPy array of floating-point samples in the
		range [-1, 1].
		
		Args:
			filename (str): Path to the WFM file.
			
		Returns:
			np.ndarray: Array of audio samples.
		"""
		with open(filename, "rb") as file:
			# Read the entire binary data
			binary_data = file.read()
			
			# Unpack the binary data as 16-bit integers ('h' = short, little-endian '<')
			num_samples = len(binary_data) // 2  # 2 bytes per 16-bit sample
			samples = struct.unpack(f"<{num_samples}h", binary_data)  # Little-endian 16-bit integers
		return np.array(samples, dtype=np.float32)/2**15	# Normalize to [-1, 1]

	@staticmethod
	def read_stim_meta(filepath):
		"""Read the stimulus metadata file"""
		stim_meta = scipy.io.loadmat(filepath, squeeze_me=True, struct_as_record=False)
		return stim_meta
