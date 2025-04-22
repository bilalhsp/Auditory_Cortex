from typing import Union
import numpy as np
from abc import ABC, abstractmethod

class BaseDataset(ABC):
    
    @abstractmethod
    def total_stimuli_duration(self, mVocs=False):
        """Returns the total duration of all the stimuli in the experiment,
        separately for unique and repeated stimuli
        
        Returns:
            {'unique': float, 'repeated': float}
        """
        return {'unique': 0, 'repeated': 0}

    @abstractmethod
    def get_stim_audio(self, stim_id, mVocs=False):
        """Reads stim audio for the given stimulus id"""
        pass

    @abstractmethod
    def get_stim_duration(self, stim_id, mVocs=False):
        """Returns duration of the stimulus in seconds"""
        pass    
    
    @abstractmethod
    def get_num_bins(self, stim_id, bin_width, mVocs=False):
        """Returns the number of bins for the given stimulus id"""
        pass

    @abstractmethod
    def get_sampling_rate(self, mVocs=False):
        """Returns the sampling rate of the neural data"""
        pass

    @abstractmethod
    def get_stim_ids(self, mVocs=False):
        """Returns the set of stimulus ids for both unique and repeated stimuli
        Returns:
            {'unique': (n,), 'repeated': (m,)}
        """
        pass
    
    @abstractmethod
    def get_training_stim_ids(self, mVocs=False):
        """Returns the set of training stimulus ids"""
        pass

    @abstractmethod
    def get_testing_stim_ids(self, mVocs=False):
        """Returns the set of testing stimulus ids"""
        pass

    @abstractmethod
    def extract_spikes(
        self, bin_width: int=50, delay: int=0, repeated: bool=False, mVocs:bool=False
        ):
        """Returns the binned spike counts for all the stimuli
        
        Args:
            bin_width: int = miliseconds specifying the time duration of each bin
            delay: int = miliseconds specifying the time delay
            repeated: bool = If True, extract spikes for repeated stimuli, otherwise for unique stimuli
            mVocs: bool = If True, extract spikes for mVocs experiment, otherwise for TIMIT experiment

        Returns:
            spikes: dict of dict = {stim_id: {channel: spike_counts}}
        """
        pass


    @staticmethod
    def bin_spike_times(s_times, duration, bin_width=50, delay=0):
        """Given the spike time, returns bins containing number of
        spikes in the 'bin_width' durations following the stimulus onset.
        
        Args:
            s_times: dict = spike times for all the channels.
            duration: float = duration of trial presentation in seconds.
            bin_width: int = miliseconds specifying the time duration of each bin, Default=50.
            delay: int = miliseconds specifying the time delay, Default=0.
    
        Returns:
            counts: dict= Binned spike counts for all channels
        """
        # converting to seconds
        bin_width = bin_width/1000
        delay=delay/1000
        counts = {}
        # adding bin_width//2 makes sure that last bin is created
        # if there is a duration of at least half bin_width at the end
        # i.e. partial bin at the end should be at least half of bin_width
        # to be included in the last bin
        duration += 1e-6
        bins = np.arange(delay, delay + duration + bin_width/2, bin_width)
        for ch, times in s_times.items():
            # for repeated stimuli, spike times will be a list
            if isinstance(times, list):
                counts_all_trials = []
                for tr_times in times:
                    counts_all_trials.append(np.histogram(tr_times, bins)[0])
                counts[ch] = np.array(counts_all_trials)
            else:
                counts[ch], _ = np.histogram(times, bins)
        return counts
    
    @staticmethod
    def calculate_num_bins(duration, bin_width):
        """Calculates the number of bins for the given duration and bin_width
        Args:
            duration: float = duration of trial presentation in seconds.
            bin_width: int = time duration of each bin in seconds.
        """
        # adding epsilon to make sure that the last bin is included when on threshold
        duration += 1e-6
        return int((duration + bin_width/2)/bin_width)
