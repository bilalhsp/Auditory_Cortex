"""
This module contains functionality to read neural spikes
for each recording session in 'ucdavis' dataset.

By organization there is a separate .mat file for each session.
Within the main directory for ucdavis data these files are located in ./Data/*.mat.

Each .mat file contains the following fields:
- TrialStimData: contains data structure for each experiment with details about
    the trials. For this dataset there are two experiments (TIMIT and mVocs)
    So it contains two datastructures. Sub-fields of each datastructre are presented later.
- WM_1: contains spike info (times, codes) for the 1st tetrode
- WM_2: contains spike info (times, codes) for the 2nd tetrode
- WM_3: contains spike info (times, codes) for the 3rd tetrode
- WM_4: contains spike info (times, codes) for the 4th tetrode


TrialStimData has the following sub-fields:
- StimulusName: complete filepath of stimulus audio, we extract filename only and use this as stimulus id
- StimulusTimeOn: time (seconds) of stimulus presentation, use with stimulus duration to get trial duration. 
- StimulusType: name of experiment each trial was part of. For TIMIT and mVocs this would be 
    ['BMT#', 'BMM#'] where # is the number of repeats for that experiment.
    e.g. for 12-repeat TIMIT experiment it would be 'BMT12'.

Tetrode is a group of 4 electrodes that are used to record neural spikes. Layout of 
these small electrodes helps in identifying the spikes from different neurons (spike sorting).
Each WM_# field contains the following sub-fields:

- times: precise time of occurance of spike
- codes: assigned code for single unit or multi-unit activity (SUA or MUA) to each spike.
    SUA is assigned a 3-digit code 
    MUA is assigned a 4-digit code

"""


import os
import scipy.io 
import numpy as np

from ..base_dataset import BaseDataset, register_dataset
from .ucdavis_metadata import UCDavisMetaData
from auditory_cortex import neural_data_dir, NEURAL_DATASETS

DATASET_NAME = NEURAL_DATASETS[1]

@register_dataset(DATASET_NAME)
class UCDavisDataset(BaseDataset):
    def __init__(self, sess_id=3, data_dir=None):
        """Initialize the UCDavisDataset with session id and data directory."""
        sess_id = int(sess_id)
        self.session_id = sess_id
        self.dataset_name = DATASET_NAME
        if data_dir is None:
            data_dir = os.path.join(neural_data_dir, self.dataset_name)
        self.data_dir = data_dir
        self.metadata = UCDavisMetaData()
        self.rec_dir = os.path.join(self.data_dir, 'Data')
        
        self.num_repeats = int(self.metadata.num_repeats_for_sess(self.session_id))
        self.rec_filename = self.metadata.full_session_name(sess_id)
        self.data, self.exp_wise_trial, self.exp_stim_ids = self.read_sess_dataset(self.rec_filename)

        self.tetrodes = ['WM_1', 'WM_2', 'WM_3', 'WM_4' ]  # example tetrode
        self.assigned_units = np.concatenate([np.unique(self.data[tet].codes) for tet in self.tetrodes])
        # all assigned unit ids (SUA and MUA) across all tetrodes

        # self.data is the entire data structre: contains
        #   TrialStimData, WM_1, WM_2, WM_3, WM_4
        # self.exp_wise_trial: contains trial data for BMT# and BMM#
        #   {BMT#: struct, BMM#: struct}
        # self.exp_stim_ids
        #   {
        # BMT#: {'unique': [...], 'repeated': [...]},
        # BMM#: {'unique': [...], 'repeated': [...]}
        #}

    def exp_name(self, mVocs=False):
        """Returns the experiment name for stim type and num of repeats"""
        if mVocs:
            return f'BMM{self.num_repeats}'
        else:
            return f'BMT{self.num_repeats}'

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
        times are returned relative to the stimulus onset.
        
        """
        exp_name = self.exp_name(mVocs)
        trial_data = self.exp_wise_trial[exp_name]

        exp_stim_ids = np.array([
            exp_stim_id.split('\\')[-1] for exp_stim_id in self.get_value(trial_data, 'StimulusName')
            ])
        tr_id = np.where(exp_stim_ids==stim_id)[0]
        stim_onset = self.get_value(trial_data, 'StimulusTimeOn')[tr_id]
        stim_dur = self.get_stim_duration(stim_id, mVocs)
        tetrodes = ['WM_1', 'WM_2', 'WM_3', 'WM_4']
        spike_times = {code: [] for code in self.assigned_units}  # dict to hold spike times for each assigned unit
        for tet in tetrodes:
            all_tet_codes = np.unique(self.data[tet].codes)        # all unique codes for the tetrode
            # spk_times_all_trials = []
            for onset in stim_onset:
                spikes_mask = (self.data[tet].times >= onset) & (self.data[tet].times <= onset + stim_dur)
                relative_spk_times = self.data[tet].times[spikes_mask] - onset	# relative to stimulus onset
                tr_codes = self.data[tet].codes[spikes_mask]
                for code in all_tet_codes:
                    code_mask = tr_codes == code
                    spike_times[code].append(relative_spk_times[code_mask])  
                    # append relative spike times for each code
            #     spk_times_all_trials.append(relative_spk_times)
            # spike_times[tet] = spk_times_all_trials
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
        # keys in data: ['TrialStimData', 'WM_1', 'WM_2', 'WM_3', 'WM_4']
        
        experiments = self.get_value(data, 'TrialStimData')     # returns two structs [BMT#, BMM#]
        # experiment wise trial data...
        exp_wise_trial = {
            np.unique(self.get_value(exp, 'StimulusType'))[0]: exp for exp in experiments
            }                                                   # {BMT#: struct, BMM#: struct}

        # get filenames of unique and repeated stimuli for each experiment
        exp_stim_ids = {}
        for exp_name, trial_data in exp_wise_trial.items():
            all_stim_names = [name.split('\\')[-1] for name in self.get_value(trial_data, 'StimulusName')]
            # num_repeats = int(exp_name[3:])
            unique_stim_names = np.array([name for name in all_stim_names if all_stim_names.count(name) == 1])
            repeated_stim_names = np.unique([name for name in all_stim_names if all_stim_names.count(name) > 1])

            stim_names = {'unique': unique_stim_names, 'repeated': repeated_stim_names}
            exp_stim_ids[exp_name] = stim_names

        return data, exp_wise_trial, exp_stim_ids