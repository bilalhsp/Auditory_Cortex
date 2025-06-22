from typing import Union
import numpy as np
from abc import ABC, abstractmethod

class BaseMetaData(ABC):

    @abstractmethod
    def total_stimuli_duration(self, mVocs=False):
        """Returns the total duration of all the stimuli in the experiment,
        separately for unique and repeated stimuli
        
        Returns:
            {'unique': float, 'repeated': float}
        """
        return {'unique': 0, 'repeated': 0}
    
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
    def get_all_available_sessions(self):
        """Returns all the available sessions in the metadata"""
        pass

    @abstractmethod
    def get_sampling_rate(self, mVocs=False):
        """Returns the sampling rate of the neural data"""  
        pass

    @abstractmethod
    def get_stim_audio(self, stim_id, mVocs=False):
        """Reads stim audio for the given stimulus id"""
        pass

    @abstractmethod
    def get_stim_duration(self, stim_id, mVocs=False):
        """Returns duration of the stimulus in seconds"""
        pass
