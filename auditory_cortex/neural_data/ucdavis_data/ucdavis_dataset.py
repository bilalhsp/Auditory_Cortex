
import os
import scipy.io 
import numpy as np

from ..base_dataset import BaseDataset
from .recording_config import RecordingConfig
from .ucdavis_metadata import UCDavisMetaData
from auditory_cortex import neural_data_dir, NEURAL_DATASETS

DATASET_NAME = NEURAL_DATASETS[1]

class UCDavisDataset(BaseDataset):
	def __init__(self, sess_id=0, data_dir=None):
		self.dataset_name = DATASET_NAME
		if data_dir is None:
			data_dir = os.path.join(neural_data_dir, DATASET_NAME)
		self.data_dir = data_dir
		self.metadata = UCDavisMetaData(data_dir)
		self.rec_dir = os.path.join(self.data_dir, 'Data')
		self.sess_rec_files = os.listdir(self.rec_dir)
		self.cfg = RecordingConfig()
		self.session_id = sess_id
		self.rec_filename = self.cfg.sessions_dict[sess_id] 
		self.data, self.exp_tr_data, self.exp_stim_ids = self.read_sess_dataset(self.rec_filename)

		# read stim audios for both TIMIT
		# stim_ids = dataset.get_stim_ids(mVocs=False)
		# all_stim_ids = np.concatenate([stim_ids['unique'], stim_ids['repeated']])
		# self.timit_audios = dataset.read_stim_audios(all_stim_ids, mVocs=False)

		# read stim audios for both TIMIT
		# stim_ids = dataset.get_stim_ids(mVocs=True)
		# all_stim_ids = np.concatenate([stim_ids['unique'], stim_ids['repeated']])
		# self.mVocs_audios = dataset.read_stim_audios(all_stim_ids, mVocs=True)

	def exp_name(self, mVocs=False):
		if mVocs:
			return 'BMM3'
		else:
			return 'BMT3'

	def get_stim_ids(self, mVocs=False):
		"""Get the stimulus ids (both unique and repeated) for the experiment"""
		exp_name = self.exp_name(mVocs)
		return self.exp_stim_ids[exp_name]
	
	def get_training_stim_ids(self, mVocs=False):
		"""Returns the set of training stimulus ids"""
		stim_ids = self.get_stim_ids(mVocs)
		return stim_ids['unique']
	
	def get_testing_stim_ids(self, mVocs=False):
		"""Returns the set of testing stimulus ids"""
		stim_ids = self.get_stim_ids(mVocs)
		return stim_ids['repeated']

	def get_stim_audio(self, stim_id, mVocs=False):
		"""Reads stim audio for the given stimulus id"""
		return self.metadata.get_stim_audio(stim_id, mVocs)

	def get_stim_duration(self, stim_id, mVocs=False):
		"""Returns duration of the stimulus in seconds"""
		return self.metadata.get_stim_duration(stim_id, mVocs)

	def get_sampling_rate(self, *args, **kwargs):
		return self.metadata.get_sampling_rate()
	
	def get_num_bins(self, stim_id, bin_width, mVocs=False):
		"""Returns the number of bins for the given stimulus id"""
		duration = self.get_stim_duration(stim_id, mVocs)
		return BaseDataset.calculate_num_bins(duration, bin_width/1000)
		# bin_width = bin_width/1000
		# return int((duration + bin_width/2)/bin_width)

	def total_stimuli_duration(self, mVocs=False):
		"""Returns the total duration of all the stimuli in the experiment,
		separately for unique and repeated stimuli"""
		stim_ids = self.get_stim_ids(mVocs)
		stim_duration = {}
		for stim_type, stim_ids in stim_ids.items():
			stim_duration[stim_type] = sum([self.get_stim_duration(stim_id, mVocs) for stim_id in stim_ids])
		return stim_duration

	def extract_spikes(self, bin_width=50, delay=0, repeated=False, mVocs=False):
		"""Returns the binned spike counts for all the stimuli
		
		Args:
			bin_width: int = miliseconds specifying the time duration of each bin
			delay: int = miliseconds specifying the time delay
			repeated: bool = If True, extract spikes for repeated stimuli, otherwise for unique stimuli
			mVocs: bool = If True, extract spikes for mVocs experiment, otherwise for TIMIT experiment

		Returns:
			spikes: dict of dict = {stim_id: {channel: spike_counts}}
		"""
		stim_group = 'repeated' if repeated else 'unique'
		stim_ids = self.get_stim_ids(mVocs)[stim_group]
		spikes = {}
		for stim_id in stim_ids:
			spikes[stim_id] = self.stim_spike_counts(stim_id, mVocs, bin_width, delay)
		return spikes

	def stim_spike_times(self, stim_id, mVocs=False):
		"""Returns the spike times for the given channel, spike
		times are returned relative to the stimulus onset"""
		exp_name = self.exp_name(mVocs)
		exp_data = self.exp_tr_data[exp_name]

		exp_stim_ids = np.array([exp_stim_id.split('\\')[-1] for exp_stim_id in exp_data.StimulusName])
		tr_id = np.where(exp_stim_ids==stim_id)[0]
		stim_onset = exp_data.StimulusTimeOn[tr_id]
		stim_dur = self.get_stim_duration(stim_id, mVocs)
		tetrodes = ['WM_1', 'WM_2', 'WM_3', 'WM_4']
		spike_times = {}
		for tet in tetrodes:
			spk_times_all_trials = []
			for onset in stim_onset:
				spikes_mask = (self.data[tet].times >= onset) & (self.data[tet].times <= onset + stim_dur)
				spk_times_trial = self.data[tet].times[spikes_mask] - onset	# relative to stimulus onset
				spk_times_all_trials.append(spk_times_trial)
			spike_times[tet] = spk_times_all_trials
		return spike_times
		
	def stim_spike_counts(self, stim_id, mVocs=False, bin_width=50, delay=0):
		"""Returns the binned spike counts for the given stimulus id"""
		spike_times = self.stim_spike_times(stim_id, mVocs)
		stim_dur = self.get_stim_duration(stim_id, mVocs)
		return BaseDataset.bin_spike_times(spike_times, stim_dur, bin_width, delay)

	def read_sess_dataset(self, rec_filename: 'str'):
		"""Specify the session number to read the recording data"""
		# rec_filename = self.sess_rec_files[sess_id]
		rec_filepath = os.path.join(self.rec_dir, rec_filename)
		data =  scipy.io.loadmat(rec_filepath, squeeze_me=True, struct_as_record=False)
		
		# experiment wise trial data...
		experiments = data['TrialStimData']
		exp_tr_data = {np.unique(getattr(exp, 'StimulusType'))[0]: exp for exp in experiments}

		# get filenames of unique and repeated stimuli for each experiment
		exp_stim_ids = {}
		for exp_name, exp_data in exp_tr_data.items():
			all_stim_names = [name.split('\\')[-1] for name in getattr(exp_data, 'StimulusName')]
			num_repeats = int(exp_name[3:])
			unique_stim_names = np.array([name for name in all_stim_names if all_stim_names.count(name) == 1])
			repeated_stim_names = np.unique([name for name in all_stim_names if all_stim_names.count(name) != 1])

			stim_names = {'unique': unique_stim_names, 'repeated': repeated_stim_names}
			exp_stim_ids[exp_name] = stim_names

		return data , exp_tr_data, exp_stim_ids



	

	# @staticmethod
	# def bin_spike_times(s_times, duration, bin_width=50, delay=0):
	# 	"""Given the spike time, returns bins containing number of
	# 	spikes in the 'bin_width' durations following the stimulus onset.
		
	# 	Args:
	# 		s_times: dict = spike times for all the channels.
	# 		duration: float = duration of trial presentation in seconds.
	# 		bin_width: int = miliseconds specifying the time duration of each bin, Default=50.
	# 		delay: int = miliseconds specifying the time delay, Default=0.
	
	# 	Returns:
	# 		counts: dict= Binned spike counts for all channels
	# 	"""
	# 	# converting to seconds
	# 	bin_width = bin_width/1000
	# 	delay=delay/1000
	# 	counts = {}
	# 	# adding bin_width//2 makes sure that last bin is created
	# 	# if there is a duration of at least half bin_width at the end
	# 	# i.e. partial bin at the end should be at least half of bin_width
	# 	# to be included in the last bin
	# 	bins = np.arange(delay, delay + duration + bin_width/2, bin_width)
	# 	for ch, times in s_times.items():
	# 		# for repeated stimuli, spike times will be a list
	# 		if isinstance(times, list):
	# 			counts_all_trials = []
	# 			for tr_times in times:
	# 				counts_all_trials.append(np.histogram(tr_times, bins)[0])
	# 			counts[ch] = np.array(counts_all_trials)
	# 		else:
	# 			counts[ch], _ = np.histogram(times, bins)
	# 	return counts
		


