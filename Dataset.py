from scipy import io
import numpy as np
import json
import os


class Neural_Data:
  """Neural_dataset class loads neural data, from the directory specified at creation & 
  provides functions to retrieve 'relative/absolute' spike times, or spike counts in the durations
  whose width is specified by 'win'
  'dir': (String) Path to the directory containing data files and the json_file.
  'json_file': (String) Default: 'Neural_data_files.json' specifies data files to be loaded.
  """
  def __init__(self, dir, json_file="Neural_data_files.json", mat_file = 'out_sentence_details_timit_all_loudness.mat'):
    self.dir = dir
    self.sentences = io.loadmat(os.path.join(self.dir + mat_file), struct_as_record = False, squeeze_me = True, )
    self.features = self.sentences['features']
    self.phn_names = self.sentences['phnnames']
    self.sentdet = self.sentences['sentdet']
    self.fs = self.sentdet[0].soundf   #since fs is the same for all sentences, using fs for the first sentence
    self.json_file = json_file
    self.spikes, self.trials = self.load_data(self.dir, self.json_file)
    self.num_channels = len(self.spikes.keys())
    print(f"Data from {self.num_channels} channels loaded...!")

  def load_data(self,dir, j_file):
    """ Loads data from __MSspk.mat files and returns a tuple of dictionaries. 
    Takes in the path of directory having the __MUspk.mat files and json file 
    with filenames to load. 
    'dir: (string) address of location with __MUspk.mat files
    'j_file': (string) json file with names of __Muspk.mat files to load
    Returns:
    (spikes, trials): 1st carries dictionary of spike structs read from __MUspk files
    and second one carries dictionary of trial structs.
    """
    data = {}
    spikes = {}
    trials = {}

    with open(os.path.join(dir,j_file), 'r') as file:
      f_names = json.load(file)
    
    # Load all data files, as specified by the json file.
    for i in range(len(f_names.keys())):
      data[i] = io.loadmat(os.path.join(dir,f_names[str(i)]), squeeze_me = True, struct_as_record = False)
      spikes[i] = data[i]['spike']
      trials[i] = data[i]['trial']
    
    return spikes, trials

  def phoneme(self, sent=0):
    #indices where phoneme exists
    phoneme_present = np.amax(self.sentdet[sent].phnmat, axis=0)
    # one-hot-encoding to indices (these may carry 0 where no phoneme is present)
    indices = np.argmax(self.sentdet[sent].phnmat, axis = 0)
    #eliminate 0's for no phonemes
    indices = indices[np.where(phoneme_present>0)]
    return self.phn_names[indices]
  
  def audio(self, sent=0):
    sound = self.sentdet[sent].sound
    return sound

  def audio_phoneme_data(self):
    audio = {}
    phnm = {}
    for i in range(499):
      phnm[i] = self.phoneme(i)
      audio[i] = self.audio(i)
    return audio, phnm
  
  def retrieve_spike_times(self, sent=212, trial = 0 , timing_type = 'relative', early_spikes = True):
    """Returns times of spikes, relative to stimulus onset or absolute time
    'sent' (int) index of stimulus sentencce 
    'trial' (int) specific trial # for the above sentence, some sentences may have 
    neural spikes for more than 1 trials of the given sentence. By default result of 
    foremost trial will be returned.
    'timing_type' (string:) 'relative' (default) returns spike times relative to StimOnset time, 
                    'absolute' returns spike times in seconds.
    
    'early_spikes' (bool: True) Neural data has some cases of spikes just before the 
    stimulus onset time, this allows user to select or reject such spikes.
    """
    s_times = {}
    if trial ==0:
      # if no trial # is provided, use the first trial for the given sentence
      # tr carries the trial # to index through spike data
      tr = self.spikes[0].trial[self.spikes[0].timitStimcode==sent][trial] 
    else:
      print('Please provide Trial # corresponding to the provided sentence using of spike.trial')
      tr = trial
      
    for i in range(self.num_channels):
      #spike_indices to index through spike fields
      spike_indices = np.where(self.spikes[i].trial == tr)
      #spike times relative to the stimuls On time (Stimon)
      if timing_type == 'relative':
        s_times[i] = self.spikes[i].stimlock[spike_indices]   
      elif timing_type == 'absolute':
        s_times[i] = self.spikes[i].spktimes[spike_indices]   
    
    return s_times

  def create_bins(self, s_times, sent=212, win=50, early_spikes = True):
    """Returns bins containing number of spikes in the 'win' durations
      following the stimulus onset.
      'win' (int) miliseconds specifing the width of time slots for binds
      'early_spikes' (bool: default = True) to include or discard early spikes 
      that start a little before stimulus onset time.
    """
    win = win/1000
    bins = {}             #miliseconds
    for i in range(self.num_channels):
      tmp = np.zeros(int(np.floor(self.sentdet[sent].duration/win + 0.4)))  # Trying to exactly match number of frames given by transformer (rounding precision)
      if s_times[i][-1] > 0:
        j = 0
        en = win                  #End time for ongoing search window

        for val in s_times[i]:
          if val < (tmp.size * win):
            if (val< en):
              if early_spikes:
                tmp[j] += 1
              else:      
                if val >= 0:
                  tmp[j] += 1
            else:    
              while(val > en):
                j += 1
                # tmp[j] += 1
                en += win
              tmp[j] += 1
      bins[i] = tmp
    
    return bins

  def retrieve_spikes_count(self, sent=212, trial = 0, win = 50, early_spikes = True):
    """Returns number of spikes in every 'win' miliseconds duration following the 
    stimulus onset time.
    'sent' (int) index of stimulus sentencce 
    'trial' (int) specific trial # for the above sentence, some sentences may have 
    neural spikes for more than 1 trials of the given sentence. By default result of 
    foremost trial will be returned.
    'win' (int: 50) miliseconds specifying the time duration of each bin
    'early_spikes' (bool: True) Neural data has some cases of spikes just before the 
    stimulus onset time, this allows user to select or reject such spikes.
    """
    #get 'relative' spike times for the given sentence and trial
    s_times = self.retrieve_spike_times(sent=sent, trial=trial, early_spikes = early_spikes)
    #return spikes count in each bin
    output = self.create_bins(s_times, sent = sent, win = win, early_spikes = early_spikes)
    
    return output
