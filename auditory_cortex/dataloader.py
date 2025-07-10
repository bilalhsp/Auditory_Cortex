"""
Provid access to neural data and DNN features.

'dataloader' module provides an interface to access neural spikes 
and features from DNN models. Hence, this can be used to compare 
these two spaces of high dimensional representation using 
Regression Analysis, Representation Similarity Analysis etc.

"""
import gc
import numpy as np
from scipy import linalg, signal
from memory_profiler import profile

from auditory_cortex.io_utils import io
from auditory_cortex import config


# from auditory_cortex.io_utils.io import read_cached_spikes, write_cached_spikes
# from auditory_cortex.io_utils.io import read_cached_features, write_cached_features
# from auditory_cortex.io_utils.io import read_context_dependent_normalizer

import logging
logger = logging.getLogger(__name__)

class DataLoader:

    def __init__(self, dataset_obj, feature_extractor=None, pad_time=None):

        self.dataset_obj = dataset_obj
        self.feature_extractor = feature_extractor

        if pad_time is None:
            self.pad_time = config['pad_time'] #0.35 # seconds
        self.neural_spikes = {} 	
        self.num_channels = None	
        self.DNN_feature_dict = {}
        self.DNN_shuffled_feature_dict = {}

    def clear_cache(self):
        self.DNN_feature_dict.clear()
        self.DNN_shuffled_feature_dict.clear()
        self.neural_spikes.clear()


    def get_layer_ids(self):
        """Returns the number of layers in the DNN model."""
        return self.feature_extractor.get_layer_ids()
    
    def calculate_num_bins(self, duration, bin_width_sec):
        """Returns the number of bins for the given duration and bin_width
        Args:
            duration: float = duration of the stimulus in seconds.
            bin_width_sec: float = size of the binning window in seconds.
        """
        return self.dataset_obj.calculate_num_bins(duration, bin_width_sec)

    # new methods...
    def get_stim_audio(self, stim_id, mVocs=False):
        """Return audio for stimulus (timit or mVocs) id"""
        return self.dataset_obj.get_stim_audio(stim_id, mVocs=mVocs)
    
    def get_stim_duration(self, stim_id, mVocs=False):
        """Return duration for stimulus (timit or mVocs) id"""
        return self.dataset_obj.get_stim_duration(stim_id, mVocs=mVocs)
        
    def get_num_bins(self, stim_id, bin_width, mVocs=False):
        """Returns number of bins for the given duration and bin_width"""
        return self.dataset_obj.get_num_bins(stim_id, bin_width, mVocs=mVocs)

    def get_sampling_rate(self, mVocs=False):
        """Returns the sampling rate of the dataset."""
        return self.dataset_obj.get_sampling_rate(mVocs=mVocs)
    
    def get_training_stim_ids(self, mVocs=False):
        """Returns the stim ids for training set.
        
        Args:
            mVocs: bool = If True, returns ids for mVocs,
                otherwise for timit stimuli.
        """
        return self.dataset_obj.get_training_stim_ids(mVocs=mVocs)
    
    def get_testing_stim_ids(self, mVocs=False):
        """Returns the stim ids for testing set.
        
        Args:
            session: int = session ID, needed ONLY if mVocs=True.
            mVocs: bool = If True, returns ids for mVocs,
                otherwise for timit stimuli.
        """
        return self.dataset_obj.get_testing_stim_ids(mVocs=mVocs)
    

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
        stim_ids, stim_duration = self.dataset_obj.metadata.sample_stim_ids_by_duration(
            percent_duration, repeated=repeated, mVocs=mVocs
            )
        return stim_ids, stim_duration
    

    def get_session_spikes(self, bin_width=50, delay=0, repeated=False, mVocs=False):
        """Reads neural spikes from the cache directiory, extracts again
        if not found there.

        Args:
            bin_width: int = size of the binning window in ms.
                1000 ms is treated as special case, where total number of
                spikes for each sentence are returned.
            delay: int: neural delay in ms.
            repeated: bool = If True, returns spikes for repeated stimuli.
            mVocs: bool = If True, returns mVocs spikes

        Returns:
            dict = dict of neural spikes with stim IDs as keys.
                {stim_id: {channel_id: (n_trials, n_bins)}} 	
        """
        stim_group = 'repeated' if repeated else 'unique'
        spikes_key = f"{stim_group}-{bin_width:04d}-{delay:04d}"
        if mVocs:
            spikes_key = "mVocs_"+spikes_key

        if spikes_key not in self.neural_spikes.keys():

            spikes = self.dataset_obj.extract_spikes(
                bin_width=bin_width, delay=delay, repeated=repeated, mVocs=mVocs
                )
            # keep for future use...
            self.neural_spikes[spikes_key] = spikes

        self.num_channels = len(list(self.neural_spikes[spikes_key].values())[0])
        return self.neural_spikes[spikes_key]
    
    def get_num_channels(self, mVocs=False):
        """Returns the number of channels in the dataset."""
        if self.num_channels is None:
            _ = self.get_session_spikes(mVocs=mVocs)
        return self.num_channels

    def get_raw_DNN_features(
            self, mVocs=False, force_reload=False, contextualized=False, scale_factor=None
        ):
        """Retrieves raw features, starts by attempting to read cached features,
        if not found, extract features and also cache them, for future use.

        Args:
            model_name: str = assigned name of DNN model of interest.
            force_reload: bool = Force reload features, even if cached already..Default=False.
            shuffled: bool = If True, loads features for shuffled network.
            contextualized: bool = If True, extracts 'contextualized' features. Deprecated.
            scale_factor: float = If not None, scales the network weights by this factor.

        Returns:
            raw_features: list of dict = 
        """
        if self.feature_extractor is None:
            raise ValueError("Feature extractor object is not available.")
        model_name = self.feature_extractor.model_name
        shuffled = self.feature_extractor.shuffled
        if not force_reload:
            raw_DNN_features = io.read_cached_features(
                model_name, dataset_name=self.dataset_obj.dataset_name,
                contextualized=contextualized,
                shuffled=shuffled, mVocs=mVocs,
                )
        if force_reload or raw_DNN_features is None:
            training_stim_ids = self.get_training_stim_ids(mVocs)
            testing_stim_ids = self.get_testing_stim_ids(mVocs)
            all_stim_ids = np.concatenate([training_stim_ids, testing_stim_ids])
            stim_audios = {}
            stim_durations = {}
            for stim_id in all_stim_ids:
                stim_audios[stim_id] = self.get_stim_audio(stim_id, mVocs=mVocs)
                stim_durations[stim_id] = self.get_stim_duration(stim_id, mVocs=mVocs)

            sampling_rate = self.get_sampling_rate(mVocs)
            if contextualized:	# deprecated...
                long_audio, total_duration, *_ = self.get_contextualized_stim_audio(include_repeated_trials=True)
                raw_DNN_features = self.get_DNN_obj(
                    model_name, shuffled=shuffled, scale_factor=scale_factor
                    ).extract_features_for_audio(long_audio, total_duration)
            else:
                logger.info(f"Extracting DNN features for '{model_name}'...")
                raw_DNN_features = self.feature_extractor.extract_features(
                    stim_audios, sampling_rate, stim_durations, self.pad_time
                    )
                

            # delete temporary variables to avoid memory issues
            del stim_audios, stim_durations
            collected = gc.collect()
            logger.info(f"Garbage collector: collected {collected} objects.")
            # cache features for future use...
            io.write_cached_features(
                model_name, raw_DNN_features, dataset_name=self.dataset_obj.dataset_name,
                contextualized=contextualized, shuffled=shuffled, mVocs=mVocs
                )
        return raw_DNN_features
    
    def get_resampled_DNN_features(
            self, bin_width, mVocs=False, LPF=False, LPF_analysis_bw=20, force_reload=False, 
        ):
        """
        Retrieves resampled all DNN layer features to specific bin_width

        Args:
            bin_width (float): width of data samples in ms.
            mVocs: bool=If true, loads features for mVocs
            LPF: bool = If true, low-pass-filters features to the bin width specified
                and resamples again at predefined bin-width (e.g. 10ms)
            LPF_analysis_bw: int = bin-width for LPF analysis in ms.
            force_reload: bool = Force reload features, even if cached already..Default=False.
        Returns:
            List of dict: all layer features (resampled at required sampling_rate).
                {layer_id: {stim_id: features}}
        """
        if self.feature_extractor is None:
            raise ValueError("Feature extractor object is not available.")
        model_name = self.feature_extractor.model_name
        if self.feature_extractor.shuffled:
            DNN_feature_dict = self.DNN_shuffled_feature_dict
        else:
            DNN_feature_dict = self.DNN_feature_dict

        if mVocs:
            features_key = 'mVocs_'+model_name
        else:
            features_key = model_name
        
        if LPF:
            features_key = features_key+'_LPF'

        if features_key not in DNN_feature_dict.keys():
            DNN_feature_dict[features_key] = {}

        model_features = DNN_feature_dict[features_key]
        if bin_width not in model_features.keys() or force_reload:
            raw_features = self.get_raw_DNN_features(mVocs=mVocs, force_reload=force_reload)

            resampled_features = {layer_id:{} for layer_id in raw_features.keys()}
            
            layer_ids = list(raw_features.keys())
            # reads first 'value' to get list of sent_IDs
            stim_ids = raw_features[layer_ids[0]].keys()

            logger.info(f"Resamping ANN features at bin-width: {bin_width}")
            bin_width_sec = bin_width/1000 # ms
            for stim_id in stim_ids:
                duration = self.get_stim_duration(stim_id, mVocs)
                # if self.pad_time is not None:
                #     duration += self.pad_time
                n = self.dataset_obj.calculate_num_bins(duration, bin_width_sec)

                if self.pad_time is not None:
                    # extra number of bins because of padding..
                    n += self.dataset_obj.calculate_num_bins(self.pad_time, bin_width_sec)
                if LPF:
                    analysis_bw_sec = LPF_analysis_bw/1000
                    n_final = self.dataset_obj.calculate_num_bins(duration, analysis_bw_sec)
                    if self.pad_time is not None:
                        # extra number of bins because of padding..
                        n_final += self.dataset_obj.calculate_num_bins(self.pad_time, analysis_bw_sec)

                for layer_id in layer_ids:
                    if bin_width == 1000:
                        # treat this as a special case, and sum all samples across time...
                        tmp = np.sum(raw_features[layer_id][stim_id].numpy(), axis=0)[None, :]
                    else:
                        tmp = signal.resample(raw_features[layer_id][stim_id], n, axis=0)
                        if LPF:
                            tmp = signal.resample(tmp, n_final, axis=0)

                    resampled_features[layer_id][stim_id] = tmp

            if LPF:
                logger.info(f"Resampled ANN features at LPF bin-width: {LPF_analysis_bw}")
            DNN_feature_dict[features_key][bin_width] = resampled_features
        return DNN_feature_dict[features_key][bin_width]

