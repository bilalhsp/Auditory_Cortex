import math
import numpy as np
from scipy.signal import resample

import naplib as nl
from auditory_cortex import config
# from auditory_cortex.neural_data import dataset
from auditory_cortex.dataloader import DataLoader
from auditory_cortex import utils
from sklearn.linear_model import RidgeCV, ElasticNet, Ridge, PoissonRegressor






class BaselineDataset:
    def __init__(self, session, bin_width, num_freqs=80):
        """
        Args:
            session: int = session ID
            bin_width: int = bin width in ms
            num_freqs (int): Number of frequency channels on spectrogram
        """       
        # self.model_name = 'strf_model'
        # data_dir = config['neural_data_dir']
        # self.dataset = NeuralData(data_dir, session)
        # self.dataset.extract_spikes(bin_width=self.bin_width, delay=0)
        # self.fs = self.dataset.fs
        self.session = str(int(session))
        self.bin_width = bin_width
        self.dataloader = DataLoader()

        self.fs = self.dataloader.metadata.get_sampling_rate()
        self.num_freqs = num_freqs # num_freqs in the spectrogram

        sent_IDs = self.dataloader.sent_IDs
        self.testing_sent_ids = self.dataloader.test_sent_IDs
        self.training_sent_ids = sent_IDs[np.isin(sent_IDs, self.testing_sent_ids, invert=True)]

        # self.session_bw_cache = {}
        self.data_cache, self.num_channels = self.load_sent_wise_features_and_spikes()



    def load_sent_wise_features_and_spikes(self):
        """Reads data for all the sents (having single presentation)
        return the features (spectrogram) and spike pairs.
        """
        print(f"BaselineDataset: Loading data for session-{self.session} at bin_width-{self.bin_width}ms.")
        raw_spikes = self.dataloader.get_session_spikes(
            session=self.session,
            bin_width=self.bin_width,
            delay=0
            )

        sent_wise_spects = {}
        sent_wise_spikes = {} 
        for sent in self.training_sent_ids:
            aud = self.dataloader.metadata.stim_audio(sent)
            # Getting the spectrogram at 10 ms and then resample to match the bin_width
            spect = nl.features.auditory_spectrogram(aud, self.fs, frame_len=10)
            
            spikes = raw_spikes[sent]
            num_bins = spikes.shape[0]
            spect = resample(spect, num_bins, axis=0)
            spect = resample(spect, self.num_freqs, axis=1)

            sent_wise_spects[sent] = spect
            sent_wise_spikes[sent] = spikes

        data_cache = {
            'spects': sent_wise_spects,
            'spikes': sent_wise_spikes,
        }
        num_channels = self.dataloader.get_num_channels(self.session)
        return data_cache, num_channels


    def get_data(self, sent_IDs=None):
        """Returns spectral-features, spikes pair for the 
        given sent IDs.
        """
        if sent_IDs is None:
            sent_IDs = self.training_sent_ids
        
        spects = self.data_cache['spects']
        spikes = self.data_cache['spikes']
        spikes_list = []
        spects_list = []

        for sent in sent_IDs:
            spects_list.append(spects[sent])
            spikes_list.append(spikes[sent])
        
        return spects_list, spikes_list

    
    def get_test_data(self):
        """Returns spectral-features, spikes (all trials)
        for the test sent IDs, given session.
            
        Returns:
            spect_list: list = each entry of list is a spect 
                for a sent audio.
            repeated_spikes_list: list of lists = 11 lists 
                corresponding to 11 trials and each one having
                entries equal to number of sentences in test set.

        """
        spect_list = []
        # repeated_spikes_list = {i: [] for i in range(11)}
        all_sent_spikes = []

        for sent in self.testing_sent_ids:
            aud = self.dataloader.metadata.stim_audio(sent)
            # Getting the spectrogram at 10 ms and then resample to match the bin_width
            spect = nl.features.auditory_spectrogram(aud, self.fs, frame_len=10)

            repeated_spikes = self.dataloader.get_neural_data_for_repeated_trials(
                session=self.session,
                bin_width=self.bin_width,
                delay=0,
                sent_IDs=[sent]
                )

            num_bins = repeated_spikes.shape[1]
            spect = resample(spect, num_bins, axis=0)
            spect = resample(spect, self.num_freqs, axis=1)
            spect_list.append(spect)
            all_sent_spikes.append(repeated_spikes)
        all_sent_spikes = np.concatenate(all_sent_spikes, axis=1)
        return spect_list, all_sent_spikes
    



class DNNDataset:
    def __init__(
            self, session, bin_width, model_name, layer_ID,
            shuffled=False, force_reload=False):
        """
        Args:
            model_name:
            session: int = session ID
            bin_width: int = bin width in ms
        """       
        # self.model_name = 'strf_model'
        # data_dir = config['neural_data_dir']
        # self.dataset = NeuralData(data_dir, session)
        # self.dataset.extract_spikes(bin_width=self.bin_width, delay=0)
        # self.fs = self.dataset.fs
        self.model_name = model_name
        self.session = str(int(session))
        self.bin_width = bin_width
        self.shuffled = shuffled
        self.force_reload = force_reload
        self.layer_ID = layer_ID
        self.dataloader = DataLoader()

        self.fs = self.dataloader.metadata.get_sampling_rate()

        sent_IDs = self.dataloader.sent_IDs
        self.testing_sent_ids = self.dataloader.test_sent_IDs
        self.training_sent_ids = sent_IDs[np.isin(sent_IDs, self.testing_sent_ids, invert=True)]

        # self.session_bw_cache = {}
        self.data_cache, self.num_channels = self.load_sent_wise_features_and_spikes()



    def load_sent_wise_features_and_spikes(self):
        """Reads data for all the sents (having single presentation)
        return the features (spectrogram) and spike pairs.
        """
        print(f"DNNDataset: Loading data for session-{self.session} at bin_width-{self.bin_width}ms.")
        raw_spikes = self.dataloader.get_session_spikes(
            session=self.session,
            bin_width=self.bin_width,
            delay=0
            )

        all_layer_features = self.dataloader.get_resampled_DNN_features(
			self.model_name, bin_width=self.bin_width, force_reload=self.force_reload,
			shuffled=self.shuffled
			)
        layer_features = all_layer_features[self.layer_ID]

        data_cache = {
            'features': layer_features,
            'spikes': raw_spikes,
        }
        num_channels = self.dataloader.get_num_channels(self.session)
        return data_cache, num_channels


    def get_data(self, sent_IDs=None):
        """Returns spectral-features, spikes pair for the 
        given sent IDs.
        """
        if sent_IDs is None:
            sent_IDs = self.training_sent_ids
        
        features = self.data_cache['features']
        spikes = self.data_cache['spikes']
        features_list = []
        spikes_list = []

        for sent in sent_IDs:
            features_list.append(features[sent])
            spikes_list.append(spikes[sent])
        
        return features_list, spikes_list

    
    def get_test_data(self):
        """Returns spectral-features, spikes (all trials)
        for the test sent IDs, given session.
            
        Returns:
            spect_list: list = each entry of list is a spect 
                for a sent audio.
            repeated_spikes_list: list of lists = 11 lists 
                corresponding to 11 trials and each one having
                entries equal to number of sentences in test set.

        """
        features_list = []
        # repeated_spikes_list = {i: [] for i in range(11)}
        all_sent_spikes = []

        features = self.data_cache['features']
        for sent in self.testing_sent_ids:
            # make sure to drop the partial sample at the end, 
            # this has already been done for repeated trials..
            features_list.append(features[sent])
            repeated_spikes = self.dataloader.get_neural_data_for_repeated_trials(
                session=self.session,
                bin_width=self.bin_width,
                delay=0,
                sent_IDs=[sent]
                )
            all_sent_spikes.append(repeated_spikes)
        all_sent_spikes = np.concatenate(all_sent_spikes, axis=1)
        return features_list, all_sent_spikes

