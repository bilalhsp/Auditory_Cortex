"""
This module contains functionality to read metadata files for 
'ucdavis' dataset, which includes TIMIT and mVocs experiments.
By organization this dataset has two files for metadata:
- recanzone_timit_details.mat: Metadata for TIMIT experiment
- MSL.mat: Metadata for mVocs experiment

Notes about Metadata files:
- recanzone_timit_details.mat: 
Contains metadata for TIMIT experiment, including stimulus names, durations, and usage flags.
    Following fields are relevant and useful:
    + 'wfmName': names of the waveforms
    + 'durSec': durations of the waveforms in seconds
    + 'use': boolean array indicating whether the waveform is used in the experiment (not sure if this is useful)
    + indUsed: matches 12 repeat experiment...looks good for all experiments.
    
    Following fields are confusing...The are supposed to contain stim ids (indices of wfmName)
        used in the experiment, but they do not match the wfmName names from neural recording files.
        For example, for 12 repeat experiments, there are 46 timit stimuli that are repeated but 
        indUsed3 gives 49 and indUsed12 gives 21. 
    06-23-25: I have founds that indUsed matches stimulus ids for both 3-repeat and 12-repeat experiments.
        i.e. if I get stim ids using metadata and dataset object, they come out to be the same.
        Make sure to verify this by running the actual experiment.!!
    + indUsed3: do not match...
    + indUsed12: do not match...
    
- MSL.mat:
Contains metadata for mVocs experiment, including waveform names, durations, and usage flags.
    Following fields are relevant and useful:
    + 'WFMname': names of the waveforms
    + 'actualDur': durations of the waveforms in seconds
    + indUsed: matches 12 repeat experiment...looks good for all experiments.
"""

import os
import glob
from pathlib import Path
import struct
import scipy.io 
import numpy as np

from .recording_config import RecordingConfig
from ..base_metadata import BaseMetaData, register_metadata
from auditory_cortex import neural_data_dir, NEURAL_DATASETS

import logging
logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s'
    )

DATASET_NAME = NEURAL_DATASETS[1]
DATA_DIR = os.path.join(neural_data_dir, DATASET_NAME)

@register_metadata(DATASET_NAME)
class UCDavisMetaData(BaseMetaData):
    def __init__(self):
        self.data_dir = DATA_DIR
    
        filepaths = sorted(Path(self.data_dir, 'Data').glob("*.mat"), key=lambda p: p.name)
        self.sessions_dict = {i: p.name for i, p in enumerate(filepaths)}

        self.cfg = RecordingConfig()
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

        self.timit_stim_ids, self.mVocs_stim_ids = self.read_stim_ids()

    def num_repeats_for_sess(self, sess_id, mVocs=False):
        """Returns the number of repeats (of test data) for the given session id
        
        Args:
            sess_id (int): Session ID to get the number of repeats for
            mVocs (bool): Number of repeats for mVocs and TIMIT are the same,
                so this argument is not used, but kept for consistency.

        Returns:
            int: Number of repeats for the given session id
        """
        full_session_name = self.full_session_name(sess_id)
        return self.cfg.sess_wise_num_repeats[full_session_name]

    def get_all_available_sessions(self, num_repeats=None):
        """Returns sessions IDs of all available sessions (with neural data available)
        
        Args: 
            num_repeats (int): Number of repeats of test data, default is 12
        """
        if num_repeats is None:
            session_ids = list(self.sessions_dict.keys())
        else:
            if num_repeats not in self.cfg.sess_wise_num_repeats.values():
                raise ValueError(f"Number of repeats {num_repeats} is not valid. Try one of {self.cfg.sess_wise_num_repeats.values()}.")
            else:
                all_session_ids = list(self.sessions_dict.keys())
                session_ids = [sess_id for sess_id in all_session_ids if self.num_repeats_for_sess(sess_id) == num_repeats]
        return session_ids
    
    def full_session_name(self, sess_id):
        """Returns the full name of the session (with date and monkey name)"""
        sess_id = int(sess_id)  # Ensure sess_id is a string
        if sess_id not in self.sessions_dict:
            raise ValueError(f"Session ID {sess_id} not found in sessions dictionary.")
        return self.sessions_dict[sess_id]
    
    def get_stim_ids(self, mVocs=False):
        """Returns the set of stimulus ids for both unique and repeated stimuli
        Returns:
            {'unique': (n,), 'repeated': (m,)}
        """
        if mVocs:
            return self.mVocs_stim_ids
        else:
            return self.timit_stim_ids

    
    def get_training_stim_ids(self, mVocs=False):
        """Returns the set of training stimulus ids (stimuli with unique presentations)
        
        Returns:    
            (n,) - array of training stimulus ids
        """
        return self.get_stim_ids(mVocs)['unique']

    def get_testing_stim_ids(self, mVocs=False):
        """Returns the set of testing stimulus ids (stimuli with repeated presentations)
        
        Returns:    
            (n,) - array of testing stimulus ids
        """
        return self.get_stim_ids(mVocs)['repeated']

    def total_stimuli_duration(self, mVocs=False):
        """Returns the total duration of all the stimuli in the experiment,
        separately for unique and repeated stimuli
        
        Returns:
            {'unique': float, 'repeated': float}
        """
        stim_durations = {}
        for stim_presentation in ['unique', 'repeated']:
            stim_ids = self.get_stim_ids(mVocs)[stim_presentation]
            if len(stim_ids) > 0:
                total_duration = sum(self.get_stim_duration(stim_id, mVocs) for stim_id in stim_ids)
                stim_durations[stim_presentation] = total_duration
        return stim_durations
    
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
    

    def wfm_id_to_stim_id(self, wfm_id, mVocs=False):
        """Converts waveform idx to stimulus id
        
        Args:
            wfm_id (str): Waveform id to convert, (MATLAB 1-based indices)
            mVocs (bool): If True, convert for mVocs, otherwise for TIMIT
        
        Returns:
            str: Corresponding stimulus id
        """
        if mVocs:
            return self.mVocs_meta['MSL'].WFMname[wfm_id-1]    # make sure to adjust for 1-based indexing
        else:
            return self.timit_meta['timit'].wfmName[wfm_id-1]
        

    def read_stim_ids(self):
        """Reads stimulus ids (unique vs repeated) from the metadata files
        
        Args:
            mVocs (bool): If True, read mVocs stimulus ids, otherwise read TIMIT stimulus ids
        
        Returns:
            dict, dict: dictionary of unique and repeated stimulus ids for both timit and mVocs
        """
        timit_ids = {}
        mVocs_ids = {}

        # these are indices of waveforms names in the list...
        timit_ids['repeated'] = np.sort(np.concatenate([
                self.timit_meta['timit'].indUsed.maleRepeat,
                self.timit_meta['timit'].indUsed.femaleRepeat
            ]))

        timit_ids['unique'] = np.sort(np.concatenate([
            self.timit_meta['timit'].indUsed.maleUnique,
            self.timit_meta['timit'].indUsed.femaleUnique
        ]))

        mVocs_ids['repeated'] = np.sort(np.concatenate([
            self.mVocs_meta['MSL'].indUsed.gruntRepeat,
            self.mVocs_meta['MSL'].indUsed.screamRepeat,
            self.mVocs_meta['MSL'].indUsed.cooRepeat
        ]))

        mVocs_ids['unique'] = np.sort(np.concatenate([
            self.mVocs_meta['MSL'].indUsed.gruntUnique,
            self.mVocs_meta['MSL'].indUsed.screamUnique,
            self.mVocs_meta['MSL'].indUsed.cooUnique
        ]))


        # converting indices to stimulus ids
        for stim_type, wfm_ids in timit_ids.items():
            timit_ids[stim_type] = np.array([
                self.wfm_id_to_stim_id(wfm_id, mVocs=False) for wfm_id in wfm_ids
                ])

        for stim_type, wfm_ids in mVocs_ids.items():
            mVocs_ids[stim_type] = np.array([
                self.wfm_id_to_stim_id(wfm_id, mVocs=True) for wfm_id in wfm_ids
                ])

        return timit_ids, mVocs_ids

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
    


