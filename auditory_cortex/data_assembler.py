# import math
import os
import numpy as np
from scipy.signal import resample
from abc import ABC, abstractmethod
import gc
import naplib as nl
import torch
import torch.nn as nn
from collections import OrderedDict

import librosa
from pycochleagram import cochleagram as cgram
from auditory_cortex.dataloader import DataLoader
from auditory_cortex.dnn_feature_extractor import create_feature_extractor

import logging
logger = logging.getLogger(__name__)
   
class BaseDataAssembler(ABC):
    """Base class for all the datasets."""
    def __init__(
            self, 
            bin_width,
            dataset_obj, 
            feature_extractor=None,
            mVocs=False,
            LPF=False,
            LPF_analysis_bw=20
            ):
        """
        Args:
            model_name:
            session: int = session ID
            bin_width: int = bin width in ms
        """        
        self.dataloader = DataLoader(dataset_obj, feature_extractor)

        self.bin_width = bin_width
        self.mVocs=mVocs
        # self.dataloader = DataLoader()
        self.LPF = LPF
        self.LPF_analysis_bw = LPF_analysis_bw
        ###################     padding mess              ######################################
        if self.LPF:
            bin_width_sec = LPF_analysis_bw / 1000
        else:
            bin_width_sec = bin_width / 1000
        self.n_offset = self.dataloader.dataset_obj.calculate_num_bins(self.dataloader.pad_time, bin_width_sec)
        ########################################################################################
        if self.LPF:
            logger.info(f"creating Dataset for LPF features, to predict at {self.LPF_analysis_bw}ms.")

        if self.mVocs:
            logger.info(f"creating Dataset for mVocs data.")
        else:
            logger.info(f"creating Dataset for timit data.")
        self.fs = self.dataloader.get_sampling_rate(mVocs=mVocs)
        self.training_stim_ids = self.dataloader.get_training_stim_ids(mVocs=self.mVocs)
        self.testing_stim_ids = self.dataloader.get_testing_stim_ids(mVocs=self.mVocs)

    @abstractmethod
    def load_features(self):
        """loads the features for the given session.
        
        Returns:
            features: dict = {stim_id: (time, num_features)}
        """
        ...
    
    def get_layer_id(self):
        """Returns the layer id for the DNN features."""
        return self.layer_id

    def get_session_id(self):
        """Returns the session ID for the dataset."""
        return self.dataloader.dataset_obj.session_id

    def get_bin_width(self):
        """Returns the bin width (sampling rate) for the dataset."""
        if self.LPF:
            return self.LPF_analysis_bw
        else:   
            return self.bin_width


    def load_sent_wise_features_and_spikes(self):
        """Reads data for all the sents (having single presentation)
        return the features (spectrogram) and spike pairs.
        """
        training_spikes, testing_spikes = self.load_neural_spikes()
        layer_features = self.load_features()
        
        data_cache = {
            'features': layer_features,
            'training_spikes': training_spikes,
            'testing_spikes': testing_spikes
        }
        # get the number of channels
        training_stim_ids = list(training_spikes.keys())
        channel_ids = list(training_spikes[training_stim_ids[0]].keys())
        # num_channels = self.dataloader.get_num_channels(self.mVocs)

        # Some training ids might be missing for some sessions 
        # (e.g. for mVocs in UCSF there can be bad trials that are skipped)
        self.training_stim_ids = np.array(training_stim_ids)

        self.dataloader.clear_cache()
        gc.collect()  # Force garbage collection
        return data_cache, channel_ids
    
    def load_neural_spikes(self):
        """load neural spikes for the given session."""
        if self.LPF:
            bin_width = self.LPF_analysis_bw
        else:
            bin_width = self.bin_width
        logger.info(f"Loading data for session at bin_width-{bin_width}ms.")
        training_spikes = self.dataloader.get_session_spikes(
            bin_width=bin_width,
            delay=0,
            repeated=False,
            mVocs=self.mVocs
            )
        testing_spikes = self.dataloader.get_session_spikes(
            bin_width=bin_width,
            delay=0,
            repeated=True,
            mVocs=self.mVocs
            )
        return training_spikes, testing_spikes


    def get_training_data(self, stim_ids=None):
        """Returns spectral-features, spikes (all trials) for the test sent IDs.
            
        Returns:
            features_list: list = [(time, num_dnn_units)] each entry of list is a
                feature for stim_id.
            spikes_list: list = [(time, channels)] all trials concatenated along time axis.
        """
        if stim_ids is None:
            stim_ids = self.training_stim_ids
        
        features = self.data_cache['features']
        training_spikes = self.data_cache['training_spikes']
        features_list = []
        spikes_list = []

        for stim in stim_ids:
            features_list.append(features[stim])
            # each ch_spikes has shape (n_trial, time), for unique stimuli n_trial=1
            # np.stack([spikes for spikes in training_spikes[1].values()], axis=-1)
            # stim_spikes = np.stack([ch_spikes for ch_spikes in training_spikes[stim].values()], axis=-1).squeeze()
            stim_spikes = np.stack(
                [training_spikes[stim][ch] for ch in self.channel_ids],
                axis=-1
                ).squeeze()
            spikes_list.append(stim_spikes)
        
        return features_list, spikes_list
    
    def get_testing_data(self, stim_ids=None):
        """Returns spectral-features, spikes (all trials) for the test sent IDs.
            
        Returns:
            features_list: list = [(time, channels)] each entry of list is a
                feature for stim_id.
            repeated_spikes_list: ndarray = (num_repeats, time, channels) all trials concatenated along time axis.
        """
        if stim_ids is None:
            stim_ids = self.testing_stim_ids
        
        features = self.data_cache['features']
        testing_spikes = self.data_cache['testing_spikes']
        features_list = []
        spikes_list = []

        for stim in stim_ids:
            features_list.append(features[stim])
            # each ch_spikes has shape (n_trial, time), for unique stimuli n_trial=num_repeats
            # stim_spikes = np.stack([ch_spikes for ch_spikes in testing_spikes[stim].values()], axis=-1).squeeze()
            stim_spikes = np.stack(
                [testing_spikes[stim][ch] for ch in self.channel_ids],
                axis=-1
                ).squeeze()
            spikes_list.append(stim_spikes)
        return features_list, spikes_list
    
    def read_session_spikes(self, dataset_obj):
        """Reads the neural spikes for new session, while keeping the
        features in the cache.
        """
        feature_extractor = self.dataloader.feature_extractor
        self.dataloader = DataLoader(dataset_obj, feature_extractor)
        training_spikes, testing_spikes = self.load_neural_spikes()

        self.data_cache['training_spikes'] = training_spikes
        self.data_cache['testing_spikes'] = testing_spikes

        training_stim_ids = list(training_spikes.keys())
        channel_ids = list(training_spikes[training_stim_ids[0]].keys())

        self.channel_ids = channel_ids
        self.num_channels = len(self.channel_ids)
        self.training_stim_ids = np.array(training_stim_ids)
        self.dataloader.clear_cache()
        gc.collect()  # Force garbage collection



class STRFDataAssembler(BaseDataAssembler):
    def __init__(
            self, dataset_obj, bin_width, mVocs=False,
            mel_spectrogram=False,
            num_freqs=80,
            spectrogram_type=None,
            ):
        self.mel_spectrogram = mel_spectrogram
        feature_extractor = None
        if self.mel_spectrogram:
            if spectrogram_type is None or 'speech2text' in spectrogram_type:
                spectrogram_type = 'speech2text'
            elif 'whisper' in spectrogram_type:
                spectrogram_type = 'whisper_tiny'
            elif 'deepspeech2' in spectrogram_type:
                spectrogram_type = 'deepspeech2'
            elif 'librosa' in spectrogram_type:
                spectrogram_type = 'librosa'
            assert spectrogram_type in ['speech2text', 'whisper_tiny', 'deepspeech2', 'librosa'], f"Invalid spectrogram type: {spectrogram_type}"
            logger.info(f"Using {spectrogram_type} type mel-spectrogram for STRF.")
            if spectrogram_type != 'librosa':
                # create feature extractor for mel-spectrogra
                feature_extractor = create_feature_extractor(spectrogram_type, shuffled=False)
        else:
            if spectrogram_type is None or 'wavlet' in spectrogram_type:
                logger.info(f"Using wavelet-spectrogram for STRF.")
            elif 'cochleogram' in spectrogram_type:
                logger.info(f"Using cochleogram for STRF.")
                
        self.spectrogram_type = spectrogram_type
        super().__init__(bin_width, dataset_obj, feature_extractor=feature_extractor, mVocs=mVocs)
        self.num_freqs = num_freqs # num_freqs in the spectrogram
        

        self.data_cache, self.channel_ids = self.load_sent_wise_features_and_spikes()
        self.num_channels = len(self.channel_ids)


    def load_features(self):
        """Loads spectrogram features for the given session."""
        sampling_rate = self.dataloader.get_sampling_rate(mVocs=self.mVocs)
        
        all_stim_ids = np.concatenate([self.training_stim_ids, self.testing_stim_ids])
        spect_features = {}

        for stim_id in all_stim_ids:

            aud = self.dataloader.get_stim_audio(stim_id, mVocs=self.mVocs)
            stim_duration = self.dataloader.get_stim_duration(stim_id, mVocs=self.mVocs)

            padding = np.zeros((int(self.dataloader.pad_time*sampling_rate)))
            aud = np.concatenate((padding, aud))
            stim_duration += self.dataloader.pad_time

            num_bins = self.dataloader.calculate_num_bins(stim_duration, self.bin_width/1000)


            # num_bins = self.dataloader.get_num_bins(stim_id, bin_width=self.bin_width, mVocs=self.mVocs)
            spect = self.get_spectrogram(aud, sampling_rate)
            spect = resample(spect, self.num_freqs, axis=1)
            
            spect = resample(spect, num_bins, axis=0)
            spect_features[stim_id] = spect

        return spect_features

    def get_spectrogram(self, aud, sampling_rate):
        """Transforms the given audio into the spectrogram"""
        # Getting the spectrogram at 10 ms and then resample to match the bin_width
        if sampling_rate != 16000:
            n_new = int(aud.size*16000/sampling_rate)
            aud = resample(aud, n_new)
        if self.mel_spectrogram:
            if self.spectrogram_type == 'librosa':
                spect = librosa.feature.melspectrogram(
                    y=aud, sr=sampling_rate, n_fft=2048,
                    win_length=int(0.025 * sampling_rate),
                    hop_length=int(0.010 * sampling_rate)
                    )
                spect = np.log10(spect + 1e-10) 
                spect = spect.transpose()
            else:
                # spect = self.processor(aud, padding=True, sampling_rate=16000).input_features[0]
                spect = self.dataloader.feature_extractor.process_input(aud)
        else: 
            if self.spectrogram_type is None or 'wavlet' in self.spectrogram_type:
                spect = nl.features.auditory_spectrogram(aud, 16000, frame_len=10)
            elif 'cochleogram' in self.spectrogram_type:
                spect = cgram.human_cochleagram(
                    aud,                # Your 2-second waveform (e.g., a NumPy array of shape (32000,) for 16 kHz)
                    sr=16000,           # Sampling rate of the waveform
                    n=50,              # Number of filters in the filterbank
                    low_lim=50,         # Lower frequency limit in Hz
                    hi_lim=8000,       # Upper frequency limit in Hz
                    sample_factor=4,    # Determines filter overlap (87.5% overlap for sample_factor=4)
                    downsample=None,     # Downsample envelopes to 200 Hz
                    nonlinearity='power' # Apply 3/10 power compression to simulate basilar membrane compression
                )
                spect = spect.transpose()

        return spect



    

class DNNDataAssembler(BaseDataAssembler):
    def __init__(
            self, dataset_obj, feature_extractor, 
            layer_id, bin_width, 
            mVocs=False,
            force_reload=False,
            LPF=False,
            LPF_analysis_bw=20
            ):
        """
        Args:
            model_name:
            session: int = session ID
            bin_width: int = bin width in ms
        """
        super().__init__(
            bin_width, dataset_obj, feature_extractor, mVocs=mVocs,
            LPF=LPF, LPF_analysis_bw=LPF_analysis_bw
            )       

        self.model_name = self.dataloader.feature_extractor.model_name
        self.force_reload = force_reload
        self.layer_id = layer_id

        self.data_cache, self.channel_ids = self.load_sent_wise_features_and_spikes()
        self.num_channels = len(self.channel_ids)
        # free up memory
        # del self.dataloader.DNN_feature_dict
        # del self.dataloader.DNN_shuffled_feature_dict
        # del self.dataloader.neural_spikes
        

    def load_features(self):
        """loads the DNN features for the given session."""
        all_layer_features = self.dataloader.get_resampled_DNN_features(
            bin_width=self.bin_width, mVocs=self.mVocs, 
            LPF=self.LPF, LPF_analysis_bw=self.LPF_analysis_bw,
            force_reload=self.force_reload,
            )
        layer_features = all_layer_features[self.layer_id]
        return layer_features


class DNNAllLayerAssembler(BaseDataAssembler):
    def __init__(
            self, 
            dataset_obj,
            feature_extractor, 
            layer_ids=None, 
            bin_width=50, 
            mVocs=False,
            force_reload=False,
            LPF=False,
            LPF_analysis_bw=20
            ):
        """
        Args:
            model_name:
            session: int = session ID
            bin_width: int = bin width in ms
        """
        super().__init__(
            bin_width, dataset_obj, feature_extractor, mVocs=mVocs,
            LPF=LPF, LPF_analysis_bw=LPF_analysis_bw
            )       

        self.model_name = self.dataloader.feature_extractor.model_name
        self.force_reload = force_reload

        if layer_ids is None:
            self.layer_ids = self.dataloader.get_layer_ids()
        else:
            self.layer_ids = layer_ids

        self.data_cache, self.channel_ids = self.load_sent_wise_features_and_spikes()
        self.num_channels = len(self.channel_ids)
        # free up memory
        # del self.dataloader.DNN_feature_dict
        # del self.dataloader.DNN_shuffled_feature_dict
        # del self.dataloader.neural_spikes
        # self.dataloader.clear_cache()
        # gc.collect()  # Force garbage collection

    def load_features(self):
        """loads the DNN features for the given session."""
        all_layer_features = self.dataloader.get_resampled_DNN_features(
            bin_width=self.bin_width, mVocs=self.mVocs, 
            LPF=self.LPF, LPF_analysis_bw=self.LPF_analysis_bw,
            force_reload=self.force_reload,
            )
        
        stim_ids = list(all_layer_features[self.layer_ids[0]].keys())

        features = {}
        for stim_id in stim_ids:
            stim_features = np.concatenate(
                [all_layer_features[layer_id][stim_id] for layer_id in self.layer_ids],
                axis=1
                )
            features[stim_id] = stim_features
        return features
    

class RandProjAssembler(BaseDataAssembler):
    def __init__(
            self, 
            dataset_obj,
            feature_extractor, 
            layer_id, 
            bin_width, 
            mVocs=False,
            force_reload=False,
            LPF=False,
            LPF_analysis_bw=20,
            conv_layers=True,
            non_linearity=False
            ):
        """
        Args:
            model_name:
            session: int = session ID
            bin_width: int = bin width in ms
        """
        super().__init__(
            bin_width, dataset_obj, feature_extractor, mVocs=mVocs,
            LPF=LPF, LPF_analysis_bw=LPF_analysis_bw
            )       

        self.model_name = self.dataloader.feature_extractor.model_name
        self.force_reload = force_reload
        self.layer_id = layer_id
        self.non_linearity = non_linearity

        self.conv_layers = conv_layers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.linear_stack = self.get_random_linear_stack(conv_layers)
        self.linear_stack = self.linear_stack.to(self.device)

        self.data_cache, self.channel_ids = self.load_sent_wise_features_and_spikes()
        self.num_channels = len(self.channel_ids)


    def get_random_linear_stack(self, conv_layers):
        """Returns a random stack of linear layers"""
        all_layer_features = self.dataloader.get_resampled_DNN_features(
            bin_width=self.bin_width, mVocs=self.mVocs, 
            LPF=self.LPF, LPF_analysis_bw=self.LPF_analysis_bw,
            force_reload=self.force_reload,
            )
        stim_ids = list(all_layer_features[self.layer_id].keys())

        # Get the dimensions of the spectrogram features
        test_aud = np.random.randn(1, 16000)
        spect = self.dataloader.feature_extractor.process_input(test_aud)
        spect_dims = spect.shape[1]

        # Get the dimensions of the features for each layer
        feature_dims = {}
        for lid in range(self.layer_id,-1,-1):
            feature_dims[lid] = all_layer_features[lid][stim_ids[0]].shape[1]
        feature_dims[-1] = spect_dims

        # Create a random linear stack
        linear_proj_layers = OrderedDict()
        for lid in range(self.layer_id+1):
            if conv_layers:
                if lid == 1:
                    stride = 2
                else:
                    stride = 1
                linear_proj_layers[f'rand_linear_{lid}'] = nn.Conv1d(
                    feature_dims[lid-1], feature_dims[lid],
                    kernel_size=3, stride=stride, padding=1
                    )
            else:
                linear_proj_layers[f'rand_linear_{lid}'] = nn.Linear(feature_dims[lid-1], feature_dims[lid])
            if self.non_linearity:
                linear_proj_layers[f'non_linearity_{lid}'] = nn.GELU()
        linear_stack = nn.Sequential(linear_proj_layers)
        return linear_stack

    @torch.no_grad()
    def load_features(self):
        """Loads spectrogram features for the given session."""
        sampling_rate = self.dataloader.get_sampling_rate(mVocs=self.mVocs)
        
        all_stim_ids = np.concatenate([self.training_stim_ids, self.testing_stim_ids])
        features = {}
        for stim_id in all_stim_ids:

            aud = self.dataloader.get_stim_audio(stim_id, mVocs=self.mVocs)
            num_bins = self.dataloader.get_num_bins(stim_id, bin_width=self.bin_width, mVocs=self.mVocs)

            if self.dataloader.pad_time is not None:
                num_bins += self.n_offset
                padding = np.zeros((int(self.dataloader.pad_time*sampling_rate)))
                aud = np.concatenate((padding, aud))
            spect = self.get_spectrogram(aud, sampling_rate)
            # spect = resample(spect, self.num_freqs, axis=1)
            # spect_features[stim_id] = spect
            if self.conv_layers:
                spect = spect.transpose()   # (num_freqs, t)
                spect = np.expand_dims(spect, axis=0)  # (1, num_freqs, t)
            spect = torch.from_numpy(spect).to(self.device)
            feats = self.linear_stack(spect).cpu().numpy()
            if self.conv_layers:
                feats = feats.squeeze().transpose() # (t, num_freqs)
            features[stim_id] = resample(feats, num_bins, axis=0)   # (t, num_freqs)
        return features
    
    def get_spectrogram(self, aud, sampling_rate):
        """Transforms the given audio into the spectrogram
        
        Args:
            aud: torch.Tensor = audio signal (t,)
            sampling_rate: int = sampling rate of the audio signal

        Returns:
            spect: torch.Tensor = spectrogram of the audio signal (t, num_freqs)
        """
        # Getting the spectrogram at 10 ms and then resample to match the bin_width
        aud = aud.squeeze()
        assert aud.ndim == 1, f"Audio should be 1D, not batched, but got {aud.shape}."
        if sampling_rate != 16000:
            n_new = int(aud.size*16000/sampling_rate)
            aud = resample(aud, n_new)
        spect = self.dataloader.feature_extractor.process_input(aud) #(t, num_freqs)
        return spect
    
