from typing import Union
import numpy as np
from abc import ABC, abstractmethod


NEURAL_METADATA_REGISTRY  = {}

def register_metadata(name: str):
    """
    Decorator to register a neural dataset class.
    
    Args:
        name (str): name of the dataset to be used.
    
    Returns:
        function: returns the decorated class.
    """
    def decorator(cls):
        if name in NEURAL_METADATA_REGISTRY :
            raise ValueError(f"Dataset '{name}' is already defined!")
        NEURAL_METADATA_REGISTRY[name] = cls
        return cls
    return decorator

def create_neural_metadata(dataset_name, *args, **kwargs):
    if dataset_name not in NEURAL_METADATA_REGISTRY :
        raise ValueError(f"Dataset '{dataset_name}' is not defined!")
    return NEURAL_METADATA_REGISTRY[dataset_name](*args, **kwargs)


class BaseMetaData(ABC):

    @abstractmethod
    def num_repeats_for_sess(self, sess_id, mVocs=False):
        """Returns the number of repeats (of test data) for the given session id
        
        Args:
            sess_id (int): Session ID to get the number of repeats for
            mVocs (bool): Number of repeats for mVocs and TIMIT are the same,
                so this argument is not used, but kept for consistency.

        Returns:
            int: Number of repeats for the given session id
        """
        return 0

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
        """Returns the set of training stimulus ids (stimuli with unique presentations)
        
        Returns:    
            (n,) - array of training stimulus ids
        """
        pass

    @abstractmethod
    def get_testing_stim_ids(self, mVocs=False):
        """Returns the set of testing stimulus ids (stimuli with repeated presentations)
        
        Returns:    
            (n,) - array of testing stimulus ids
        """
        pass
    
    @abstractmethod
    def get_all_available_sessions(self):
        """Returns sessions IDs of all available sessions 
        (with neural data available)
        """
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

    def sample_stim_ids_by_duration(self, percent_duration=None, repeated=False, mVocs=False):
        """Returns random choice of stimulus ids, for the desired fraction of total 
        duration of test set as specified by percent_duration.
        
        Args:
            percent_duration: float = Fraction of total duration to consider.
                If None or >= 100, returns all stimulus ids.
            repeated: bool = if True, returns spikes for repeated trials, else for unique trials.
            mVocs: bool = If True, mVocs trials are considered otherwise TIMIT
        
        Returns:
            list: stimulus subset for the fraction of duration.
        """
        stim_durations = self.total_stimuli_duration(mVocs)
        if repeated:
            all_stim_ids = self.get_testing_stim_ids(mVocs)
            total_duration = stim_durations['repeated']
        else:
            all_stim_ids = self.get_training_stim_ids(mVocs)
            total_duration = stim_durations['unique']
        np.random.shuffle(all_stim_ids)
        if percent_duration is None: 
            return all_stim_ids, total_duration
        else:
            required_duration = percent_duration*total_duration/100
            stim_duration=0
            choosen_stim_ids = []

            while stim_duration < required_duration:
                stim_id = np.random.choice(all_stim_ids)
                stim_duration += self.get_stim_duration(stim_id, mVocs=mVocs)
                choosen_stim_ids.append(stim_id)
            return np.array(choosen_stim_ids), stim_duration
    
