from scipy import io
import numpy as np
import json
import os
import matplotlib.pyplot as plt

import auditory_cortex.utils as utils
from auditory_cortex import neural_data_dir
from auditory_cortex.neural_data.deprecated.neural_meta_data import NeuralMetaData


class NeuralData:
	"""Neural_dataset class loads neural data, from the directory specified at creation & 
	provides functions to retrieve 'relative/absolute' spike times, or spike counts in the durations
	whose width is specified by 'win'
	'dir_path': (String) Path to the directory containing data files and the json_file.
	'json_file': (String) Default: 'Neural_data_files.json' specifies data files to be loaded.
	"""
	def __init__(self, sub, dir=None, mat_file = 'out_sentence_details_timit_all_loudness.mat', verbose=False):
		print(f"NeuralData:  Creating object for session: {sub} ... ", end='')
		self.sub = str(int(sub))
		if dir is None:
			dir = neural_data_dir
		self.dir = os.path.join(dir, 'ucsf')
		self.sentences = io.loadmat(os.path.join(self.dir, mat_file), struct_as_record = False, squeeze_me = True, )
		self.features = self.sentences['features']
		self.phn_names = self.sentences['phnnames']
		self.sentdet = self.sentences['sentdet']
		self.fs = self.sentdet[0].soundf   #since fs is the same for all sentences, using fs for the first sentence
		self.names = os.listdir(os.path.join(self.dir, self.sub)) 
		# print(self.names)
		self.spikes, self.trials = self.load_data(verbose=verbose)
		self.num_channels = len(self.spikes.keys())
		self.sents = np.arange(1,500)   # 499 sentences in total (1 - 499)
		self.sent_sections = {}
		
		self.ordered_sent_IDs, self.inter_stimulus_intervals, self.ordered_trial_IDs = self.get_ordered_sent_IDs_and_trial_IDs()
		
		# mVocs 
		mVocs_excluded_sessions = ['190726', '200213']
		self.metadata = NeuralMetaData()
		if self.sub not in mVocs_excluded_sessions:
			self.mVocs_first_tr, self.missing_trial_ids = self.get_mVocs_trial_details()

		print("Done.")
		# print(f"Data from {self.num_channels} channels loaded...!")

	def load_data(self, verbose):
		""" Loads data from __MUspk.mat files and returns a tuple of dictionaries. 
		Takes in the path of directory having the __MUspk.mat files and json file 
		with filenames to load. 
		'dir: (string) address of location with __MUspk.mat files
		'j_file': (string) json file with names of __Muspk.mat files to load
		Returns:
		(spikes, trials): 1st carries dictionary of spike structs read from __MUspk files
		and second one carries dictionary of trial structs.
		"""
		path = os.path.join(self.dir, self.sub)
		spikes = {}
		trials = {}
		data = {}
		self.names.sort()
		for i, name in enumerate(self.names):
			if verbose:
				print(name)
			if 'MUspk' in name:
				# print(name)
				data[i] = io.loadmat(os.path.join(path,name), squeeze_me = True, struct_as_record = False)
				spikes[i] = data[i]['spike']
				trials[i] = data[i]['trial']
		
		return spikes, trials

	def phoneme(self, sent=1):
		# subtracting 1 because timitStimcodes range [1,500) and sent indices range [0,499)
		# MATLAB ID to PYTHON ID
		sent -= 1
		
		#indices where phoneme exists
		phoneme_present = np.amax(self.sentdet[sent].phnmat, axis=0)
		# one-hot-encoding to indices (these may carry 0 where no phoneme is present)
		indices = np.argmax(self.sentdet[sent].phnmat, axis = 0)
		#eliminate 0's for no phonemes
		indices = indices[np.where(phoneme_present>0)]
		return self.phn_names[indices]
  
	def audio(self, sent=1):
		# subtracting 1 because timitStimcodes range [1,500) and sent indices range [0,499)
		sent -= 1
		fs = self.sentdet[sent].soundf
		bef, aft = 0.5, 0.5
		#bef = self.sentdet[sent].befaft[0]
		#aft = self.sentdet[sent].befaft[1]
		sound = self.sentdet[sent].sound
		sound = sound[int(bef*fs):-int(aft*fs)]
		
		return sound

	def duration(self, sent=1):
		# subtracting 1 because sent_id =1 is stored at index 0 in sentdet and so on...
		sent -= 1
		bef, aft = 0.5, 0.5
		#bef = self.sentdet[sent].befaft[0]
		#aft = self.sentdet[sent].befaft[1]
		duration = self.sentdet[sent].duration - (bef + aft)
		return duration

	def audio_phoneme_data(self):
		audio = {}
		phnm = {}
		for i in range(499):
			phnm[i] = self.phoneme(i)
			audio[i] = self.audio(i)
		return audio, phnm
  
	def get_trials(self, sent):
		#get all trials for sentence 'sent'
		#Trials are repeated for these sentences only
		#sents = [12,13,32,43,56,163,212,218,287,308]

		#trials = np.unique(obj.dataset.spikes[1].trial[obj.dataset.spikes[1].timitStimcode==sent])
		# Using channel 0 trials dict to get list of trials for 'sent', but trials for all the channels are the same 
		#only the outcome 'spikes' can vary, so that is the reason of not using spikes
		
		try: 
			# np.where() returns Python index (0 onwards), add 1 to get MATLAB index (1 onwards)
			trials = (np.where(self.trials[0].timitStimcode == sent)[0]) + 1          # adding 1 to match the indexes
			return trials
		except:
			raise ModuleNotFoundError("No trial data found...!")
	

	def retrieve_spike_counts_for_all_trials(self, sent, win=50, delay=0):
		
		trials = self.get_trials(sent)
		spk = self.retrieve_spike_counts(trial=trials[0], win = win, delay=delay)
		spikes = {i:np.zeros((len(trials), spk[0].shape[0])) for i in range(self.num_channels)}
		for i in range(self.num_channels):
			spikes[i][0] = spk[i]
		for x, tr in enumerate(trials[1:]):
			spikes_tr = self.retrieve_spike_counts(trial=tr, win = win, delay=delay)
			for i in range(self.num_channels):
				spikes[i][x+1] = spikes_tr[i]
		return spikes

	def retrieve_spike_times(self, sent=212, trial = 0 , timing_type = 'relative'):
		"""Returns times of spikes, relative to stimulus onset or absolute time
		'sent' (int) index of stimulus sentencce 
		'trial' (int) specific trial # for the above sentence, some sentences may have 
		neural spikes for more than 1 trials of the given sentence. By default result of 
		foremost trial will be returned.
		'timing_type' (string:) 'relative' (default) returns spike times relative to StimOnset time, 
						'absolute' returns spike times in seconds.
		
		"""
		s_times = {}
		if trial ==0:
			# if no trial # is provided, use the first trial for the given sentence
			# tr carries the trial # to index through spike data
			#tr = self.spikes[1].trial[self.spikes[1].timitStimcode==sent][trial] 
			# Extracting 'trial #' using the first channel...!
			# tr = (np.where(self.trials[0].timitStimcode == sent)[0][0]) + 1      # adding 1 to match the indexes
			tr = self.get_trials(sent)[0]           # Using 1st trial for stimuli with multiple trials
		else:
			#print('Please provide Trial # corresponding to the provided sentence using of spike.trial')
			tr = trial
			# ADDED one to i to make it match indices
		for i in range(self.num_channels):
			#spike_indices to index through spike fields
			# print("here: ",i, tr)
			# j = i+1
			j = i
			spike_indices = np.where(self.spikes[j].trial == tr)
			#spike times relative to the stimuls On time (Stimon)
			if timing_type == 'relative':
				s_times[i] = self.spikes[j].stimlock[spike_indices]   
			elif timing_type == 'absolute':
				s_times[i] = self.spikes[j].spktimes[spike_indices]   
		
		return s_times

	def create_bins(self, s_times, sent=212, trial=0, win=50, delay = 0):
		"""Returns bins containing number of spikes in the 'win' durations
		following the stimulus onset.
		
		Args:
			sent (int): ID of the audio stimulus (sentence)
			trial (int): trial ID to create bins for. Can be used to get binned spikes
				directly using the trial number, instead of sent ID.
			win (int): miliseconds specifing the width of time slots for bins.
			delay (ms): Delaying the features versus spikes. (we can also think of this
				as advancing the spikes, that would mean delaying the spikes w.r.t. to spikes.)

		"""
		if trial != 0:
			# MATLAB ID to PYTHON ID
			trial -= 1
			sent = self.trials[0].timitStimcode[trial]
		win = win/1000
		delay = delay/1000
		bins = {}             #miliseconds
		# round the 3rd decimal digit, before ceiling...!
		n = int(np.ceil(round(self.duration(sent)/win, 3)))
		
		# store boundaries of sent thirds...
		one_third = int(n/3)
		two_third = int(2*n/3)
		self.sent_sections[sent] = [0, one_third, two_third, n] 


		for i in range(self.num_channels):
			tmp = np.zeros(n, dtype=np.int32) 
			j = 0
			st = delay
			en = st+win                  #End time for ongoing search window
			for val in s_times[i]:
				if val < (tmp.size * win + delay):
					if (val<= en and val>st):
						tmp[j] += 1
					
					elif val>en:    
						while(val > en):
							j += 1
							st += win
							en += win
						if j<n:
							tmp[j] += 1
					
			bins[i] = tmp
		
		return bins

	def retrieve_spike_counts(self, sent=212, trial = 0, win = 50, delay=0):
		"""Returns number of spikes in every 'win' miliseconds duration following the 
		stimulus onset time.
		'sent' (int) index of stimulus sentencce 
		'trial' (int) specific trial # for the above sentence, some sentences may have 
		neural spikes for more than 1 trials of the given sentence. By default result of 
		foremost trial will be returned.
		'win' (int: 50) miliseconds specifying the time duration of each bin
		"""
		#get 'relative' spike times for the given sentence and trial
		s_times = self.retrieve_spike_times(sent=sent, trial=trial)
		#return spikes count in each bin
		output = self.create_bins(s_times, sent=sent, trial=trial, win = win,delay=delay)
		
		return output

	def spike_counts(self, sent=212, trial=0, win=50, delay=0):
		## Spike count using np.histogram function, this is in addition to
		#  my own binning implementation in Retrieve_spikes_count()
		# and they both give the same output
		s_times = self.retrieve_spike_times(sent=sent, trial=trial)
		win = win/1000
		counts = {}
	
		duration = round(self.duration(sent),3)  #round off to 3 decimals...
		bins = np.arange(delay, delay + duration, win)
		for i in range(self.num_channels):
			counts[i], _ = np.histogram(s_times[i], bins)
		return counts

	def extract_spikes(self, bin_width=20, delay=0, sents = None):
		"""Return neural spikes for given sents
		
		Args:
			bin_width: int = in ms
			delay: int = in ms
		"""
		print(f"NeuralData: Reading neural spikes for {self.sub}. ")
		if sents is None:
			sents = self.sents
		raw_spikes = {}
		for x,i in enumerate(sents):
			spikes = self.retrieve_spike_counts(sent=i,win=bin_width,delay=delay)
			raw_spikes[i] = np.stack([spikes[ch] for ch in range(self.num_channels)], axis=1)
		# self.raw_spikes =  raw_spikes
		return raw_spikes
  
	def get_stim_onset(self, tr):
		"""Retreives stimulus onset time for the given trial ID"""
		# safe to look at the 0th channel
		# These trial IDs floating around are MATLAB ID's i.e. starting from 1
		# whenever we directly index any array, make sure to convert them to 
		# Python indexes.
		# MATLAB ID to PYTHON ID
		tr -= 1
		return self.trials[0].stimon[tr] 


	def get_ordered_sent_IDs_and_trial_IDs(self):
		"""Returns sent ID in the ordered of stimulus presentation 
		to the subject, and corresponding trial ID.

		Returns:
			ordered_sent_IDs (ndarray): sent ID's in the order they were presented
			inter_stimulus_dead_intervals: dead intervals after each stimulus,
			ith index of this list reads dead interval AFTER corresponding sent 
			at ith index.
			list_all_timit_trials: ordered list of timit trials.
		"""   
		# getting the list of trials 
		sent_to_trial = {}
		list_all_timit_trials = []
		for id in self.sents:
			trials = self.get_trials(id)
			sent_to_trial[id] = trials
			if trials.size > 1:
				list_all_timit_trials.extend(trials)
			else:
				list_all_timit_trials.append(trials[0])
		list_all_timit_trials = np.array(list_all_timit_trials)
		list_all_timit_trials.sort()

		# getting the ordered list of sent IDs
		ordered_sent_IDs = []
		for tr in list_all_timit_trials:
			for id in sent_to_trial.keys():
				if tr in sent_to_trial[id]:
					ordered_sent_IDs.append(id)
		ordered_sent_IDs = np.array(ordered_sent_IDs)

		inter_stimulus_dead_intervals = []
		# corresponding dead intervals within stimuli...
		for i, (tr, sent_ID) in enumerate(zip(list_all_timit_trials, ordered_sent_IDs)):
			if i < ordered_sent_IDs.size - 1:
				try:
					inter_stimulus_dead_intervals.append(
					# Stim_onset(next_trial) - Stim_onset(current_trial) - duration(current_trial)
					self.get_stim_onset(tr+1) - self.get_stim_onset(tr) - self.duration(sent_ID)
					)
				except: raise IndexError(f"Sent: ID = {sent_ID}, i={i}")
			else:
				# in case we are at the last trial, just add zero dead time...
				inter_stimulus_dead_intervals.append(0)

		inter_stimulus_dead_intervals = np.array(inter_stimulus_dead_intervals)

		return ordered_sent_IDs, inter_stimulus_dead_intervals, list_all_timit_trials

  

	def retrieve_contextualized_spikes(self, bin_width=20, delay=0):
		"""Retrieves spikes for all sent IDs [0, 498] in order of
		presentations, with spikes for inter-trial intervals as well.
		
		Args:
			bin_width: float= in miliseconds.
			delay: (float) = in miliseconds

		Returns:
			ndarray (bins, num_channels)
		"""

		# convert to seconds
		bin_width /= 1000 # in seconds
		delay /= 1000 # in seconds

		first_sent = self.ordered_sent_IDs[0]
		last_sent = self.ordered_sent_IDs[498]

		first_tr = self.get_trials(first_sent)[0]   # first sent presentation..
		last_tr = self.get_trials(last_sent)[0]   # first sent presentation..

		# while directly indexing array, convert MATLAB index to Python index
		ch = 0  # extracting trial data using ch=0, it would be same for all channels
		session_strt_time = self.trials[ch].stimon[first_tr-1] 
		session_end_time = self.trials[ch].stimon[last_tr-1] + self.duration(last_sent) 
		total_session_duration = session_end_time - session_strt_time



		all_channel_spikes = []
		for ch in range(self.num_channels):
			spk_times = []
			# np.where gives us Python index so no need to correction...
			first_spk_ind = np.where(self.spikes[ch].trial==first_tr)[0][0]

			last_spike_seq = np.where(self.spikes[ch].trial==last_tr)[0]
			# At least one session has ZERO spikes for last trial, 
			# so we look trial before last, until we get a trial with spikes..
			while last_spike_seq.size < 1:
				last_tr -= 1
				last_spike_seq = np.where(self.spikes[ch].trial==last_tr)[0]
			last_spk_ind = last_spike_seq[-1]
			for spk_ind in range(first_spk_ind, last_spk_ind+1):
				spk_times.append(self.spikes[ch].spktimes[spk_ind])

			spk_times = np.array(spk_times) - session_strt_time + delay
			time_steps = np.arange(0, total_session_duration, bin_width)
			spk_counts, edges = np.histogram(spk_times, time_steps)

			all_channel_spikes.append(spk_counts)

		return np.stack(all_channel_spikes, axis=1), total_session_duration
	

	def get_repeated_trials(self, sents=None, bin_width=20, delay=0):
		"""Get repeated trials for given sents as 'ndarray'. """
		# print(f"Getting spikes for repeated trials of timit..")
		if sents is None:
			sents = [12,13,32,43,56,163,212,218,287,308]
		spikes_dict = {}
		min_repeats = 500   #repetition of trials (mostly it is 11)
		sum_spikes = False
		if bin_width == 1000:
			# treat this as a special case, where all samples are summed across time.
			bin_width = 20
			sum_spikes = True

		for s in sents:
			spikes_sentence = self.retrieve_spike_counts_for_all_trials(sent=s, win = bin_width, delay=delay)
			spikes_dict[s] = np.stack([spikes_sentence[ch] for ch in range(self.num_channels)], axis=-1)
			if spikes_dict[s].shape[0] < min_repeats:
				min_repeats = spikes_dict[s].shape[0] 
			if sum_spikes:
				spikes_dict[s] = np.sum(spikes_dict[s], axis=1)[:,None,:]
		if sum_spikes:
			all_repeated_trials = np.concatenate([spikes_dict[s][:min_repeats,:,:] for s in sents], axis=1)
		else:
			# stopped trimming last partial bin otherwise do [[:min_repeats,:-1,:]]
			all_repeated_trials = np.concatenate([spikes_dict[s][:min_repeats,:,:] for s in sents], axis=1)
	  
		return all_repeated_trials
  
####################################
##### mVocs methods..
################################

	def get_mVocs_trials(self, mVocs):
		"""Returns trial IDs for the mVocs ID"""
		
		try: 
			# np.where() returns Python index (0 onwards), add 1 to get MATLAB index (1 onwards)
			trials = (np.where(self.trials[0].mVocStimcode == mVocs)[0]) + 1          # adding 1 to match the indexes
			return trials
		except:
			raise ModuleNotFoundError("No trial data found...!")
		
	def get_mVocs_trial_details(self):
		"""Returns mVocs trial presentation details like first mVocs trial ID
		missing trial ids. 
		Normally, we would expect trial ids ranging from first mVocs trial ID
		and first mVocs trial id + 780 to be all the trials, but some of these 
		trials might be missing for some sessions. So we need to keep track of
		missing trial ids.

		Returns:
			mVocs_first_tr: int = trial id of first mVocs presentation, rest of the trial
				ids will be right after this.
			missing_trial_ids: list = some of the session might have missing trial 
				presentations.

		"""
		mVocstimCodes = np.unique(self.spikes[0].mVocStimcode)
		# mVocstimCodes
		trial_ids = []
		# excluding trial id=0
		for mVocs in mVocstimCodes[1:]:
			trials = self.get_mVocs_trials(mVocs)
			trial_ids.extend(list(trials))
		if len(trial_ids)==0:
			raise ModuleNotFoundError(f"No mVocs trial presentations for session: {self.sub}...!")
		trial_ids.sort()
		mVocs_first_tr = np.min(trial_ids)

		trial_ids = trial_ids - mVocs_first_tr
		all_trial_ids = np.arange(780)
		missing_trial_ids = all_trial_ids[np.isin(all_trial_ids, trial_ids,invert=True)]
		return mVocs_first_tr, missing_trial_ids

	
	def extract_mVocs_spikes(self, bin_width=50, delay=0, trials = None):
		"""Return neural spikes for given trials
		
		Args:
			bin_width: int = in ms
			delay: int = in ms
			trials: list = trial Ids of mVocs in [0, 779]
		"""
		print(f"NeuralData: Reading mVocs spikes for {self.sub}. ")
		if trials is None:
			all_trials = self.metadata.mVocTrialIds
			trials = all_trials[np.isin(all_trials, self.missing_trial_ids, invert=True)]
		raw_spikes = {}
		for tr in trials:
			spikes = self.retrieve_mVocs_spike_counts(trial=tr, win=bin_width, delay=delay)
			raw_spikes[tr] = np.stack([spks for spks in spikes.values()], axis=1)
		# self.raw_spikes =  raw_spikes
		return raw_spikes
	
	
	def retrieve_mVocs_spike_counts(self, trial=0, win=50, delay=0):
		"""Returns number of spikes in every 'win' miliseconds duration following the 
		stimulus onset time.

		Args:
			trial: (int) = specific trial # for mVocs from [0, 779].
			win: int = miliseconds specifying the time duration of each bin, Default=50.
			delay: int = miliseconds specifying the time delay, Default=0.
		"""
		if trial in self.missing_trial_ids:
			raise ModuleNotFoundError("No trial data found...!")
		duration = self.metadata.get_mVoc_dur(trial)

		# map trial Id [0, 779] to session specific trial Id.. 
		sess_trial = self.mVocs_first_tr + trial
		#get 'relative' spike times for the given trial
		s_times = self.retrieve_spike_times(trial=sess_trial)
		#return spikes count in each bin
		output = NeuralData.bin_spike_times(s_times, duration, win=win, delay=delay)
		return output
	

	def get_repeated_mVoc_trials(
			self, mVocs_ids, bin_width=50, delay=0):
		"""
		Retrieves all trials of neural spikes for the given mVocs.

		Args:
			mVocs_ids: list = mVocs ids 
			bin_width: int = bin width in ms.
			delay: int = neural delay in ms.

		Returns:
			return ndarray =(num_trials, samples, n_channels)
		"""
		mVocs_spikes = {}

		for mVocs in mVocs_ids:
			tr_ids = self.metadata.get_mVoc_tr_id(mVocs)
			all_trial_spikes = []
			for tr in tr_ids:
				spikes = self.retrieve_mVocs_spike_counts(tr, bin_width, delay)
				spikes = np.stack([spk for spk in spikes.values()], axis=1)

				all_trial_spikes.append(spikes)
			mVocs_spikes[mVocs] = np.stack(all_trial_spikes, axis=0)
		repeated_trials = np.concatenate([spk for spk in mVocs_spikes.values()], axis=1).astype(np.float32)
		return repeated_trials
	

	
	@staticmethod
	def bin_spike_times(s_times, duration, win=50, delay=0):
		"""Given the spike time, returns bins containing number of
		spikes in the 'win' durations following the stimulus onset.
		
		Args:
			s_times: dict = spike times for all the channels.
			duration: float = duration of trial presentation in seconds.
			win: int = miliseconds specifying the time duration of each bin, Default=50.
			delay: int = miliseconds specifying the time delay, Default=0.
	
		Returns:
			counts: dict= Binned spike counts for all channels
		"""
		# converting to seconds
		win = win/1000
		delay=delay/1000
		counts = {}
		n = int(np.ceil(round(duration/win,3)))
		# in order to get duration equal to integer multiple of win
		# using n+1 because stop entry is not included..
		dur = round((n+1)*win, 3) - 0.001	# to prevent incrementally small decimal values to create extra bin
		bins = np.arange(delay, delay + dur, win)
		for ch, times in s_times.items():
			counts[ch], _ = np.histogram(times, bins)
		return counts




		# convert to seconds..
		# win = win/1000

		# counts = {}

		# n = int(np.ceil(round(duration/win, 3)))
		# # print(f"Duration={duration} sec, num_samples={n}")
		# # duration = round(duration,3)  #round off to 3 decimals...
		# # bins = np.arange(delay, delay + duration, win)
		
		# # delay is in ms
		# # converting duration to ms
		# # np.histogram returns one less bin, so need number of bins = n+1
		# bins = np.linspace(delay, delay + (duration*1000), n+1)/1000
		# for ch, times in s_times.items():
		# 	counts[ch], _ = np.histogram(times, bins)
		# return counts





####################################
##### MOVED to Regresion..
################################
  # def get_normalizer(self, bin_width=20, delay=0, n=1000):
  #     """Compute dist. of normalizer and return median."""
  #     # using sents with repeats
  #     sents = [12,13,32,43,56,163,212,218,287,308]
  #     all_repeated_trials = self.get_repeated_trials(sents=sents, bin_width=bin_width, delay=delay)
  #     normalizer_all = utils.inter_trial_corr(all_repeated_trials, n=n)
  #     normalizer_all_med = np.median(normalizer_all, axis=0)
  #     return normalizer_all_med




	
	# Neural Data Plotting Functions....
	
	def rastor_plot(self ,sent=12, ch=9):
		# Rastor plot for all the trials of given 'sent' and channel 'ch'

		#Repeated trials for following timitStimcodes only
		#sents = [12,13,32,43,56,163,212,218,287,308]
		spikes = {}
		max_time = 0
		#fig = plt.figure(figsize=(12,6))
		trials = self.get_trials(sent=sent)
		for i, tr in enumerate(trials):
			spikes[i] = self.retrieve_spike_times(sent=sent, trial=tr)[ch]
			mx = np.amax(spikes[i], axis=0)
			if mx > max_time:
				max_time = mx 
			#print(spikes[i].shape)
			plt.eventplot(spikes[i], lineoffsets=i+1, linelengths=0.3, linestyles='-', linewidths=8)
		plt.xlim(0,self.duration(sent))
		plt.xlabel('Time (s)', fontsize=14)
		plt.ylabel('Trials', fontsize=14)
		#plt.title(f"Rastor Plot for session: {self.sub}, sentence: {sent}, ch: '{self.names[ch]}'", fontsize=14, fontweight='bold')
	
	def psth(self, sent=12, ch=9, win = 40):
		trials = self.get_trials(sent=sent)
		spikes = {}
		#fig = plt.figure(figsize=(12,6))
		for i, tr in enumerate(trials):
			spikes[i] = self.retrieve_spike_counts(sent=12, trial=tr, win=win)[ch]
			if i==0:
				psth = np.zeros(spikes[i].shape[0])
			psth += spikes[i]
			#print(spikes[i].shape)
		#print(psth.shape)
	
		psth /= trials.size
		edges = np.float64(np.arange(0, psth.shape[0]))*win/1000

		return edges, psth
	# plt.bar(edges,psth, width=(0.8*win/1000))
	# plt.xlim(0,self.duration(sent))
	# #plt.xlabel('Time (s)', fontsize=14)
	# plt.ylabel('Spike Counts', fontsize=14)
	#plt.title(f"PSTH session: {self.sub}, sentence: {sent}, ch: '{self.names[ch]}', bin: {win}", fontsize=14, fontweight='bold')

	
	def signal_power(self, win, ch, sents = [12,13,32,43,56,163,212,218,287,308]):
		
		sp = 0
		for s in sents:
			r = self.retrieve_spike_counts_for_all_trials(sent=s, w=win)[ch]
			N = r.shape[0]
			trail_sum = np.sum(r, axis=0)
			n1 = np.var(trail_sum, axis=0)
			n2 = 0
			for i in range(r.shape[0]):
				n2 += np.var(r[i])
			sp += (n1 - n2)/(N*(N-1))
		sp /= len(sents)
		return sp