import os
import time
import numpy as np
import pandas as pd

from .dataset import NeuralData
from .neural_meta_data import NeuralMetaData
from auditory_cortex import results_dir


class Normalizer:
    """Computes the inter-trial correlations for stimuli
    having multilple trials. These will be used as normalizer
    for regression correlations and for identifying channels
    (and sessions) with good enough SNR."""

    def __init__(self, normalizer_filename = None):
        if normalizer_filename is None:
            print(f"Using default normalizer file...")
            normalizer_filename = "modified_bins_normalizer.csv"
            # "corr_normalizer.csv" original with full length sequnces. 
        
        print(f"Creating normalizer object from: {normalizer_filename}")
        self.dataframe, self.filepath = Normalizer.load_data(normalizer_filename)
        self.metadata = NeuralMetaData()
        self.loaded_datasets = {}
        # self.test_sent_IDs = [12,13,32,43,56,163,212,218,287,308]

    def get_significant_sessions(self, bin_width=20, delay=0, threshold=0.068):
        """Returns a list of session having at least one significant channels"""
        select_data = self._get_normalizers_for_bin_width_and_delay(bin_width, delay)
        select_data = select_data[select_data['normalizer'] >= threshold]
        return select_data['session'].unique()
    
    def get_good_channels(self, session, bin_width=20, delay=0, threshold=0.068):
        """Returns a list of sig. channels for the session, bw and delay"""
        select_data = self.get_normalizer_for_session(session, bin_width=bin_width, delay=delay)
        select_data = select_data[select_data['normalizer'] >= threshold]
        return select_data['channel'].unique()
        
    def _get_normalizers_for_bin_width_and_delay(self, bin_width=20, delay=0):
        """Retrieves section of dataframe for specified bin_width and delay"""
        select_data =  self.dataframe[
            (self.dataframe['bin_width']==bin_width) &\
            (self.dataframe['delay']==delay)
        ]
        return select_data

    def get_normalizer_for_session(self, session, bin_width=20, delay=0):
        """Retrives normalizer for the arguments."""
        session = float(session)
        select_data = self._get_normalizers_for_bin_width_and_delay(bin_width, delay)
        select_data = select_data[(select_data['session']==session)]
        if select_data.shape[0] == 0:
            print(f"Results NOT available for session-{session} at bw-{bin_width} & delay-{delay}, computing now...", end='')
            self.save_normalizer_for_session(session, bin_width=bin_width, delay=delay)
            print(f"Done.")
            # recursive call after computing and saving resutlts
            self.get_normalizer_for_session(session, bin_width, delay)
            # raise ValueError(f"Results NOT available for session-{session} at bw-{bin_width} & delay-{delay},\
            #     use 'save_normalizer_for_session(...)' or 'save_normalizer_for_all_sessions()'")
        else:
            return select_data

    
    def save_normalizer_for_all_sessions(self, bin_width=20, delay=0):
        sessions = self.metadata.get_all_available_sessions()
        select_data = self._get_normalizers_for_bin_width_and_delay(bin_width, delay)
        sessions_done = select_data['session'].unique()

        sessions_remaining = sessions[np.isin(sessions, sessions_done, invert=True)]

        for session in sessions_remaining:
            print(f"Normalizer not available for {session}" +
                f" at bw-{bin_width}, delay-{delay}, computing now...")
            strt_time = time.time()
            self.save_normalizer_for_session(session, bin_width=bin_width, delay=delay)
            end_time = time.time()
            print(f"It took {end_time-strt_time}s for session-{session}.!")


    def save_normalizer_for_session(self, session, bin_width=20, delay=0):
        """Computes and saves the normalizer result, for the given configuration."""
        norm_dist = self._compute_normalizer(session, bin_width=bin_width,
                                       delay=delay)
        session = float(session)
        num_channels = norm_dist.size
        data = np.stack([
            session*np.ones(num_channels),
            np.arange(num_channels),
            bin_width*np.ones(num_channels),
            delay*np.ones(num_channels),
            norm_dist
        ], axis=1
        )
        df = pd.DataFrame(
            data= data,
            columns=self.dataframe.columns,
                        )
        # writing back...
        self.dataframe = Normalizer.write_data(df, self.filepath)

        # self.dataframe = pd.concat([self.dataframe, df], axis=0, ignore_index=True)
        # self.dataframe.to_csv(self.filepath, index=False)

    
                

    def _compute_normalizer(self, session, bin_width=20, delay=0, n=100000):
        """Compute dist. of normalizer for correlations (repeatability of neural
        spikes), and return median."""
        # session = str(int(session))
        # dataset = NeuralData(session)
        dataset = self._get_dataset_obj(session)
        all_repeated_trials = dataset.get_repeated_trials(
            sents=self.metadata.test_sent_IDs, bin_width=bin_width,
            delay=delay
            )

        normalizer_all = Normalizer.inter_trial_corr(all_repeated_trials, n=n)
        normalizer_all_med = np.median(normalizer_all, axis=0)
        return normalizer_all_med
    
    def _get_dataset_obj(self, session):
        """Retrieves dataset object"""
        session = str(int(session))
        if session not in self.loaded_datasets.keys():
            self.loaded_datasets.clear()
            self.loaded_datasets[session] = NeuralData(session)
        return self.loaded_datasets[session]
    
    def compute_normalizer_threshold(self, bin_width, p_value=5, itr=10000):
        """Computes significance threshold for normalizer at bin_width."""

        total_samples_test_set = 0
        if bin_width == 1000:
            # special case...
            total_samples_test_set = self.metadata.test_sent_IDs.size
        else:
            for sent in self.metadata.test_sent_IDs:
                total_samples_test_set += self.metadata.stim_samples(sent, bin_width=bin_width)

        print(f"Computing null distribution for bin_width: {bin_width}, num_samples: {total_samples_test_set}...")
        null_dist = []
        for i in range(itr):
            gaussian_sample_of_same_length = np.random.randn(2, total_samples_test_set)
            null_dist.append(np.corrcoef(gaussian_sample_of_same_length)[0,1])

        q = 100 - p_value
        return np.percentile(null_dist, q), null_dist
    

    def compute_normalizer_threshold_using_poisson(
            self, bin_width, spike_rate=50, p_value=5, itr=10000
        ):
        """Normalizer based on assumption that spikes are generated by a poisson process, 
        and are uniformly distributed in time."""
        print(f"Poisson Process: Null distribution for bin_width: {bin_width}, spike_rate: {spike_rate}...")
        test_duration = self.metadata.get_total_test_duration()
        total_spikes = int(spike_rate * test_duration)
        num_bins = int(np.ceil(round(1000*test_duration/bin_width, 3)))
        null_dist = []
        for i in range(itr):
            spike_times_1 = np.random.uniform(0, test_duration, int(total_spikes))
            counts_1,_ = np.histogram(spike_times_1, num_bins)

            spike_times_2 = np.random.uniform(0, test_duration, int(total_spikes))
            counts_2, _ = np.histogram(spike_times_2, num_bins)

            null_dist.append(np.corrcoef(counts_1, counts_2)[0,1])
       
        q = 100 - p_value
        return np.percentile(null_dist, q), null_dist

    
#########################################################
######      static methods
########################################################

    @staticmethod
    def write_data(df, filepath):
        if os.path.exists(filepath):
            existing_data = pd.read_csv(filepath)
        else:
            existing_data = pd.DataFrame(columns=df.columns)
        updated_data = pd.concat([existing_data, df], axis=0, ignore_index=True)
        updated_data.to_csv(filepath, index=False)
        print(f"Result updated...!")
        return updated_data
        


    @staticmethod
    def load_data(filename):
        """Loads/creates dataframe containing normalizers"""
        filepath = os.path.join(
            results_dir, 'neural_repeatibitliy', filename
            )
        if not os.path.exists(os.path.dirname(filepath)):
            print(f"Creating directory structure for normalizer...!")
            os.makedirs(os.path.dirname(filepath))

        if os.path.exists(filepath):
            print(f"Reading existing dataframe.")
            dataframe = pd.read_csv(filepath)
        else:
            print(f"Creating new dataframe.")
            columns = ['session', 'channel','bin_width',
                       'delay', 'normalizer']
            dataframe = pd.DataFrame(columns=columns)
        return dataframe, filepath
    

    @staticmethod
    def inter_trial_corr(spikes, n=100000):
        """Compute distribution of inter-trials correlations.

        Args: 
            spikes (ndarray): (repeats, samples/time, channels)

        Returns:
            trials_corr (ndarray): (n, channels) distribution of inter-trial correlations
        """
        trials_corr = np.zeros((n, spikes.shape[2]))
        num_channels = spikes.shape[-1]
        for t in range(n):
            trials = np.random.choice(np.arange(0,spikes.shape[0]), size=2, replace=False)
            for ch in range(num_channels):
                trials_corr[t, ch] = np.corrcoef(
                    spikes[trials[0],:,ch].squeeze(), spikes[trials[1],:,ch].squeeze()
                    )[0,1]
        return trials_corr
    







