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

# from auditory_cortex.models import Regression
# from auditory_cortex.neural_data.neural_meta_data import NeuralMetaData
# from auditory_cortex.neural_data.dataset import NeuralData
from auditory_cortex.neural_data import NeuralData, NeuralMetaData
from auditory_cortex.analyses import Correlations
from auditory_cortex.computational_models.feature_extractors import DNNFeatureExtractor

from auditory_cortex.io_utils.io import read_cached_spikes, write_cached_spikes
from auditory_cortex.io_utils.io import read_cached_features, write_cached_features

from auditory_cortex.io_utils.io import read_cached_spikes_session_wise
from auditory_cortex.io_utils.io import write_cached_spikes_session_wise


class DataLoader:

    def __init__(self):
        
        # Needs to be replaced with more robust way of getting the normalizers..
        self.corr_obj = Correlations('wave2letter_modified_normalizer2')
        self.metadata = NeuralMetaData()
        self.test_sent_IDs = [12,13,32,43,56,163,212,218,287,308]
        self.sent_IDs = self.metadata.sent_IDs
        self.spike_datasets = {}
        self.neural_spikes = {}
        self.num_channels = {}
        self.DNN_models = {}
        self.DNN_layer_ids = {}
        self.DNN_feature_dict = {}
        self.raw_DNN_features = {}

    def _create_DNN_obj(self, model_name='waveletter_modified'):
        """Creates DNN feature extractor for the given model_name"""
        # self.DNN_models[model_name] = Regression(model_name, load_features=False)
        self.DNN_models[model_name] = DNNFeatureExtractor(model_name)
        self.DNN_layer_ids[model_name] = self.DNN_models[model_name].layer_IDs
        self.DNN_feature_dict[model_name] = {}

    def get_DNN_obj(self, model_name='waveletter_modified'):
        """Retrieves DNN model for the given name, create new if not already 
        exists.
        """
        if model_name not in self.DNN_models.keys():
            self._create_DNN_obj(model_name=model_name)
        return self.DNN_models[model_name]

    def get_raw_DNN_features(self, model_name, force_reload=False):
        """Retrieves raw features for the 'model_name', starts by
        attempting to read cached features, if not found, extract
        features and also cache them, for future use.

        Args:
            model_name: str = assigned name of DNN model of interest.
            force_reload: bool = Force reload features, even if cached already..Default=False.
        Returns:
            raw_features: list of dict = 
        """
        # self.raw_DNN_features[model_name] = read_cached_features(model_name)
        # stop saving 'raw_DNN_features' to save the memory....
        # only need to save the resampled...that I am already doing...
        raw_DNN_features = read_cached_features(model_name)
        if raw_DNN_features is None or force_reload:
            # self.get_DNN_obj(model_name).load_features(resample=False)
            # self.raw_DNN_features[model_name] = self.get_DNN_obj(model_name).sampled_features

            raw_DNN_features = self.get_DNN_obj(model_name).extract_DNN_features()
            # cache features for future use...
            write_cached_features(model_name, raw_DNN_features)
        return raw_DNN_features
        
    def get_resampled_DNN_features(self, model_name, bin_width, force_reload=False):
        """
        Retrieves resampled all DNN layer features to specific bin_width

        Args:
            model_name: str = assigned name of DNN model of interest.
            bin_width (float): width of data samples in ms (1000/sampling_rate).
            force_reload: bool = Force reload features, even if cached already..Default=False.

        Returns:
            List of dict: all layer features (resampled at required sampling_rate).
        """

        if bin_width not in self.DNN_feature_dict.keys() or force_reload:
            raw_features = self.get_raw_DNN_features(model_name, force_reload=force_reload)
            # num_layers = len(raw_features)
            resampled_features = {layer_id:{} for layer_id in raw_features.keys()}
            
            layer_IDs = list(raw_features.keys())
            # reads first 'value' to get list of sent_IDs
            sent_IDs = raw_features[layer_IDs[0]].keys()

            print(f"Resamping ANN features at bin-width: {bin_width}")
            bin_width_sec = bin_width/1000 # ms
            for sent_ID in sent_IDs:
                # 'self.audio_padding_duration' will be non-zero in case of audio-zeropadding
                sent_duration = self.metadata.stim_duration(sent_ID)
                n = int(np.ceil(round(sent_duration/bin_width_sec, 3)))

                for layer_ID in layer_IDs:
                    tmp = signal.resample(raw_features[layer_ID][sent_ID], n, axis=0)
                    # mean = np.mean(tmp, axis=0)
                    # resampled_features[j][sent] = tmp #- mean
                        
                    resampled_features[layer_ID][sent_ID] = tmp
            self.DNN_feature_dict[bin_width] = resampled_features
        return self.DNN_feature_dict[bin_width]
    
    def get_DNN_layer_features(self, model_name, layer_ID, bin_width=20):
        """Retrieves layer features, sampled according to bin_width,
        stores the results in a dict (bin_width) as key, so that 
        the features can be reused for rest of the layers (at the same bin_width).
        
        Args:
            model_name: str = assigned name of DNN model of interest.
            layer_ID: int = ID of the layer to get the features for.
            bin_width: int = bin width in ms.

        Returns:
            features for the layer
        """
        resampled_features = self.get_resampled_DNN_features(
                    model_name, bin_width=bin_width
                    )
        if layer_ID not in resampled_features.keys():
            resampled_features = self.get_resampled_DNN_features(
                    model_name, bin_width=bin_width, force_reload=True
                    )
        try: return resampled_features[layer_ID]
        except:
            raise KeyError(f"Layer ID '{layer_ID}' is not included in the network FE configuration.")
        
        # if bin_width not in self.DNN_feature_dict.keys():
        #     # print(f"Resampling raw features for bin_width={bin_width}..")
                # features = self.get_resampled_DNN_features(
                #     model_name, bin_width=bin_width
                #     )
            # self.DNN_feature_dict[bin_width] = features
        # return self.DNN_feature_dict[bin_width][layer_ID]
    
    def get_layer_index(self, model_name, layer_id):
        """Returns index for the layer_id (assigned in model specific config file),
        for model_name.
        """
        if model_name not in self.DNN_layer_ids:
            self._create_DNN_obj(model_name=model_name)
        try: 
            return self.DNN_layer_ids[model_name].index(layer_id)
        except:
            raise ValueError(f"Layer ID '{layer_id}' is not included in the {model_name} configuration.")
        
    def get_layer_IDs(self, model_name):
        """Retrieves layer IDs in DNN configurations..."""
        if model_name not in self.DNN_layer_ids:
            self._create_DNN_obj(model_name=model_name)
        return self.DNN_layer_ids[model_name]
    
    def get_DNN_feature_dims(self, model_name):
        """Retrives num_dims of DNN features."""



    def list_loaded_sessions(self):
        """Returns the list of sessions for which neural data has
        been loaded."""
        return self.spike_datasets.keys()
    
    def _load_dataset_session(self, session):
        """Create dataset object for the 'session'"""
        self.spike_datasets[session] = NeuralData(session)
        # self.neural_spikes[session] = {}

    def get_dataset_object(self, session):
        """Returns spike dataset object if neural data for the input 
        session has already been loaded, otherwise load new object."""
        session = str(int(session))
        if session not in self.spike_datasets.keys():
            # while creating new dataset, get rid of the
            # other datasets already loaded, to manage memory,
            # in-short, keeping only ONE session loaded at a time.
            self.spike_datasets.clear()
            self.neural_spikes.clear()
            gc.collect()
            self._load_dataset_session(session)
        return self.spike_datasets[session]

        #     return self.spike_datasets[session]
        # except:
        #     raise AttributeError(f"Create dataset object for session-{session} before using it.")

    # def get_raw_neural_spikes(self, session, bin_width=20, delay=0):
    def _extract_session_spikes(self, session, bin_width=20, delay=0):
        """Returns neural spikes in the raw form (not unrolled),
        for individual recording site (session).

        Args:
            session: = recording site (session) ID 
            bin_width: int = size of the binning window in ms.
            delay: int: neural delay in ms.
        Returns:
            dict = dict of neural spikes with sent IDs as keys.
        """
        print(f"DataLoader: Extracting spikes for session-{session}...", end='')
        session = str(int(session))
        # combination of session, bin_width and delay becomes the key self.neural_spikes
        spikes_key = f"{int(session):06d}-{bin_width:04d}-{delay:04d}"
        if spikes_key not in self.neural_spikes.keys():
            self.get_dataset_object(session).extract_spikes(bin_width, delay)#, sents=sents)
            # self.num_channels[session] = self.get_dataset_object(session).num_channels
            self.neural_spikes[spikes_key] = self.get_dataset_object(session).raw_spikes
        print(f"Done.")
        return self.neural_spikes[spikes_key]
    
    def get_session_spikes(self, session, bin_width=20, delay=0):
        """Reads neural spikes from the cache directiory, extracts again
        if not found there.

        Args:
            session: = recording site (session) ID 
            bin_width: int = size of the binning window in ms.
            delay: int: neural delay in ms.
        Returns:
            dict = dict of neural spikes with sent IDs as keys."""
        session = str(int(session))
        spikes_key = f"{int(session):06d}-{bin_width:04d}-{delay:04d}"
        if spikes_key not in self.neural_spikes.keys():
            session_wise_spikes = read_cached_spikes_session_wise(bin_width=bin_width, delay=delay)
            if session_wise_spikes is None or session not in session_wise_spikes.keys():
                spikes = self._extract_session_spikes(session, bin_width=bin_width,
                                                    delay=delay)
                write_cached_spikes_session_wise(
                    spikes, session=session, bin_width=bin_width, delay=delay
                    )
                self.neural_spikes[spikes_key] = spikes
            else:
                self.neural_spikes[spikes_key] = session_wise_spikes[session]
        # saving num of channels for the session..
        self.num_channels[session] = next(iter(self.neural_spikes[spikes_key].values())).shape[-1]
        return self.neural_spikes[spikes_key]
    
    def get_num_channels(self, session):
        """Returns the number of channels in the dataset."""
        session = str(int(session))
        if session not in self.num_channels.keys():
            _ = self.get_session_spikes(session=session)
        return self.num_channels[session]

    

            

        # self.raw_DNN_features[model_name] = read_cached_features(model_name)
        # if self.raw_DNN_features[model_name] is None or force_reload:
        #     # self.get_DNN_obj(model_name).load_features(resample=False)
        #     # self.raw_DNN_features[model_name] = self.get_DNN_obj(model_name).sampled_features

        #     self.raw_DNN_features[model_name] = self.get_DNN_obj(model_name).extract_DNN_features()
        #     # cache features for future use...
        #     write_cached_features(model_name, self.raw_DNN_features[model_name])
        # return self.raw_DNN_features[model_name]



        # # check if session is already loaded, reuse it, otherwise clear all sessions to saved memory.
        # if session not in self.list_loaded_sessions():
        #     self.spike_datasets.clear()
        #     gc.collect()
        #     # de-allocation of memory ends here...

        #     self.spike_datasets = {}
        #     self.num_channels = {}
        #     self._load_dataset_session(session)
        #     self.get_dataset_object(session).extract_spikes(bin_width, delay)#, sents=sents)
        #     self.num_channels[session] = self.get_dataset_object(session).num_channels
        # return self.get_dataset_object(session).raw_spikes

    def get_all_neural_spikes(self, bin_width=20, threshold=0.068, **kwargs):
        """"Retrieves cached neural spikes, for a group (core, belt or all)of
        recording sites (sessions), extract spikes if not already cached.

        Args:
            bin_width: int= bin_width in ms

        kwargs:
            area: str = ['core', 'belt'], default=None.
        """
        if 'area' in kwargs:
            area = kwargs.pop('area')
            if area != 'all':
                assert area in ['core', 'belt'], "Incorrect brain" + \
                    "area specified, choose from ['core', 'belt']"
        else:
            area = 'all'
        
        spikes = read_cached_spikes(bin_width=bin_width, threshold=threshold)
        if spikes is None or area not in spikes.keys():
            z_stim = self._extract_sig_neural_data(
                threshold=threshold, bin_width=bin_width, area=area
                )
            write_cached_spikes(z_stim, bin_width=bin_width, area=area, threshold=threshold)
        else:
            z_stim = spikes[area]
        
        return z_stim 
    
    def _extract_sig_neural_data(self, threshold=0.068, bin_width=20, **kwargs):
        """Get stimulus-wise neural spike data, for group of session 
        (core, belt or all), with significant sessions from these
        sessions concatenated together for each stimulus.

        Args:
            threshold: float = significance threshold (0,1).
            bin_width: int= bin_width in ms

        kwargs:
            area: str = ['core', 'belt', 'all'], default=None.
        
        Returns:
            dict: dictionary of stimulus representations, with sent
                ID's as keys.
        """
        if 'area' in kwargs:
            area = kwargs.pop('area')
            if area != 'all':
                assert area in ['core', 'belt'], "Incorrect brain" + \
                    "area specified, choose from ['core', 'belt']"
        else:
            area = 'all'
        print(f"DataLoader: Extracting all neural spikes for '{area}' area...", end='')
        # if len(kwargs) != 0: raise ValueError("Unrecognizable keyword args.")
        sig_sessions = self.corr_obj.get_significant_sessions(threshold=threshold)
        if area != 'all':
            sessions = self.metadata.get_all_sessions(area)
            sig_sessions = sig_sessions[np.isin(sig_sessions, sessions)]
        z_stim = {}

        for session in sig_sessions:
            spikes_sess = self.get_session_spikes(session, bin_width=bin_width)
            # spikes_sess = self._extract_session_spikes(session, bin_width=bin_width)

            # keep only good channels...
            good_channels = np.array(
                self.corr_obj.get_good_channels(session, threshold=threshold),
                dtype=np.uint32
                )
            if len(z_stim) != 0:
                z_stim = {
                    sent: np.concatenate([z_val, z_sess_val[...,good_channels]], axis=1) \
                    for (sent, z_val), z_sess_val in zip(z_stim.items(), spikes_sess.values())}
            else:
                z_stim = {sent: array[...,good_channels] for sent, array in spikes_sess.items()}
        print("Done.")
        return z_stim
    
    def get_neural_data_for_repeated_trials(self, session, bin_width=20, delay=0):
        """Retrieves neural data only for sent-stimuli with repeated
        trials i.e. test_sent_IDs"""

        return self.get_dataset_object(session=session).get_repeated_trials(
            sents=self.test_sent_IDs, bin_width=bin_width, delay=delay)


