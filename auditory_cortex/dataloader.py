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
# from auditory_cortex import LPF_analysis_bw

from auditory_cortex.io_utils.io import read_cached_spikes, write_cached_spikes
from auditory_cortex.io_utils.io import read_cached_features, write_cached_features

# from auditory_cortex.io_utils.io import read_cached_spikes_session_wise
# from auditory_cortex.io_utils.io import write_cached_spikes_session_wise

from auditory_cortex.io_utils.io import read_context_dependent_normalizer




class DataLoader:

    def __init__(self):
        
        # Needs to be replaced with more robust way of getting the normalizers..
        self.corr_obj = Correlations('wav2letter_modified_normalizer2')
        self.metadata = NeuralMetaData()
        self.test_sent_IDs = self.metadata.test_sent_IDs #[12,13,32,43,56,163,212,218,287,308]
        self.sent_IDs = self.metadata.sent_IDs
        self.spike_datasets = {}
        self.neural_spikes = {}
        self.num_channels = {}
        self.DNN_models = {}
        self.DNN_layer_ids = {}
        self.DNN_feature_dict = {}
        self.DNN_shuffled_feature_dict = {}
        # self.raw_DNN_features = {}

    def get_stim_aud(self, stim_id, mVocs=False):
        """Return audio for stimulus (timit or mVocs) id, resampled at 16kHz"""
        if mVocs:
            return self.metadata.get_mVoc_aud(stim_id)
        else:
            return self.metadata.stim_audio(stim_id)
        
    def get_stim_dur(self, stim_id, mVocs=False):
        """Return duration for stimulus (timit or mVocs) id"""
        if mVocs:
            return self.metadata.get_mVoc_dur(stim_id)
        else:
            return self.metadata.stim_duration(stim_id)
        
    def get_num_bins(self, stim_id, bin_width, mVocs=False):
        """Returns number of bins for the given duration and bin_width"""
        duration = self.get_stim_dur(stim_id, mVocs)
        # return int(np.ceil(duration/(bin_width/1000)))
        return int(np.ceil(round(duration/(bin_width/1000), 3)))

    def _create_DNN_obj(self, model_name='waveletter_modified', shuffled=False, scale_factor=None):
        """Creates DNN feature extractor for the given model_name"""
        # self.DNN_models[model_name] = Regression(model_name, load_features=False)
        self.DNN_models[model_name] = DNNFeatureExtractor(model_name, shuffled=shuffled, scale_factor=scale_factor)
        self.DNN_layer_ids[model_name] = self.DNN_models[model_name].layer_IDs
        self.DNN_feature_dict[model_name] = {}

    def get_DNN_obj(self, model_name='waveletter_modified', shuffled=False, scale_factor=None):
        """Retrieves DNN model for the given name, create new if not already 
        exists.
        """
        if model_name not in self.DNN_models.keys():
            self._create_DNN_obj(model_name=model_name, shuffled=shuffled, scale_factor=scale_factor)
        return self.DNN_models[model_name]
    

    def get_raw_DNN_features_for_mVocs(
            self, model_name, force_reload=False, contextualized=False, shuffled=False
        ):
        """Retrieves raw features for the 'model_name', starts by
        attempting to read cached features, if not found, extract
        features and also cache them, for future use.

        Args:
            model_name: str = assigned name of DNN model of interest.
            force_reload: bool = Force reload features, even if cached already..Default=False.
        Returns:
            raw_features: list of dict = 
        """
        raw_DNN_features = read_cached_features(
            model_name, contextualized=contextualized, shuffled=shuffled, mVocs=True)
        if raw_DNN_features is None or force_reload:
            # self.get_DNN_obj(model_name).load_features(resample=False)
            # self.raw_DNN_features[model_name] = self.get_DNN_obj(model_name).sampled_features

            if contextualized:
                ...
            else:
                raw_DNN_features = self.get_DNN_obj(
                    model_name, shuffled=shuffled
                    ).extract_DNN_features_for_mVocs()
            # cache features for future use...
            write_cached_features(
                model_name, raw_DNN_features, contextualized=contextualized,
                shuffled=shuffled, mVocs=True
                )
        return raw_DNN_features

    def get_raw_DNN_features(
            self, model_name, force_reload=False, contextualized=False, shuffled=False,
            scale_factor=None
        ):
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

        # shuffled = self.get_DNN_obj(model_name).shuffled
        raw_DNN_features = read_cached_features(model_name, contextualized=contextualized, shuffled=shuffled)
        if raw_DNN_features is None or force_reload:
            # self.get_DNN_obj(model_name).load_features(resample=False)
            # self.raw_DNN_features[model_name] = self.get_DNN_obj(model_name).sampled_features

            if contextualized:
                long_audio, total_duration, *_ = self.get_contextualized_stim_audio(include_repeated_trials=True)
                raw_DNN_features = self.get_DNN_obj(
                    model_name, shuffled=shuffled, scale_factor=scale_factor
                    ).extract_features_for_audio(long_audio, total_duration)
            else:
                dnn_obj = self.get_DNN_obj(
                    model_name, shuffled=shuffled, scale_factor=scale_factor
                    )
                raw_DNN_features = dnn_obj.extract_DNN_features()
                if shuffled:
                    dnn_obj.save_state_dist()
            # cache features for future use...
            write_cached_features(model_name, raw_DNN_features, contextualized=contextualized, shuffled=shuffled)
        return raw_DNN_features
        
    def get_resampled_DNN_features(
            self, model_name, bin_width, force_reload=False, 
            shuffled=False, mVocs=False, LPF=False, LPF_analysis_bw=20
        ):
        """
        Retrieves resampled all DNN layer features to specific bin_width

        Args:
            model_name: str = assigned name of DNN model of interest.
            bin_width (float): width of data samples in ms (1000/sampling_rate).
            force_reload: bool = Force reload features, even if cached already..Default=False.
            shuffled: bool = If True, loads features for shuffled network
            mVocs: bool=If true, loads features for mVocs
            LPF: bool = If true, low-pass-filters features to the bin width specified
                and resamples again at predefined bin-width (e.g. 10ms)
        Returns:
            List of dict: all layer features (resampled at required sampling_rate).
        """
        if shuffled:
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
            if mVocs:
                raw_features = self.get_raw_DNN_features_for_mVocs(
                    model_name, force_reload=force_reload, shuffled=shuffled
                    )
            else:
                raw_features = self.get_raw_DNN_features(
                    model_name, force_reload=force_reload, shuffled=shuffled
                    )
            # num_layers = len(raw_features)
            resampled_features = {layer_id:{} for layer_id in raw_features.keys()}
            
            layer_IDs = list(raw_features.keys())
            # reads first 'value' to get list of sent_IDs
            stim_IDs = raw_features[layer_IDs[0]].keys()

            print(f"Resamping ANN features at bin-width: {bin_width}")
            bin_width_sec = bin_width/1000 # ms
            for stim_ID in stim_IDs:
                # 'self.audio_padding_duration' will be non-zero in case of audio-zeropadding
                if mVocs:
                    duration = self.metadata.get_mVoc_dur(stim_ID)
                else:
                    duration = self.metadata.stim_duration(stim_ID)
                n = int(np.ceil(round(duration/bin_width_sec, 3)))
                # int(np.ceil(duration/(bin_width/1000)))
                if LPF:
                    # LPF_analysis_bw = 10
                    analysis_bw_sec = LPF_analysis_bw/1000
                    n_final = int(np.ceil(round(duration/analysis_bw_sec, 3)))

                for layer_ID in layer_IDs:
                    if bin_width == 1000:
                        # treat this as a special case, and sum all samples across time...
                        tmp = np.sum(raw_features[layer_ID][stim_ID].numpy(), axis=0)[None, :]
                    else:
                        tmp = signal.resample(raw_features[layer_ID][stim_ID], n, axis=0)
                        if LPF:
                            tmp = signal.resample(tmp, n_final, axis=0)

                    resampled_features[layer_ID][stim_ID] = tmp

            if LPF:
                print(f"Resampled ANN features at LPF bin-width: {LPF_analysis_bw}")
            DNN_feature_dict[features_key][bin_width] = resampled_features
        return DNN_feature_dict[features_key][bin_width]
    
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
    def _extract_session_spikes(self, session, bin_width=20, delay=0, mVocs=False):
        """Returns neural spikes in the raw form (not unrolled),
        for individual recording site (session).

        Args:
            session: = recording site (session) ID 
            bin_width: int = size of the binning window in ms.
            delay: int: neural delay in ms.
            mVocs: bool = If True, returns mVocs spikes
        Returns:
            dict = dict of neural spikes with sent IDs as keys.
        """
        print(f"DataLoader: Extracting spikes for session-{session}...", end='\n')
        session = str(int(session))
        if mVocs:
            return self.get_dataset_object(session).extract_mVocs_spikes(bin_width, delay)
        else:
            return self.get_dataset_object(session).extract_spikes(bin_width, delay)
        # combination of session, bin_width and delay becomes the key self.neural_spikes
        # spikes_key = f"{int(session):06d}-{bin_width:04d}-{delay:04d}"
        # if spikes_key not in self.neural_spikes.keys():
        #     # self.get_dataset_object(session).extract_spikes(bin_width, delay)#, sents=sents)
        #     # # self.num_channels[session] = self.get_dataset_object(session).num_channels
        #     # self.neural_spikes[spikes_key] = self.get_dataset_object(session).raw_spikes
            
        #     self.neural_spikes[spikes_key] = self.get_dataset_object(session).extract_spikes(bin_width, delay)#, sents=sents)

        # print(f"Done.")
        # return self.neural_spikes[spikes_key]
    
    def get_session_spikes(self, session, bin_width=50, delay=0, mVocs=False):
        """Reads neural spikes from the cache directiory, extracts again
        if not found there.

        Args:
            session: = recording site (session) ID 
            bin_width: int = size of the binning window in ms.
                1000 ms is treated as special case, where total number of
                spikes for each sentence are returned.
            delay: int: neural delay in ms.
            mVocs: bool = If True, returns mVocs spikes
        Returns:
            dict = dict of neural spikes with sent IDs as keys."""
        session = str(int(session))
        sum_spikes = False
        # stop caching to memory, to avoid unnessary complexity...
        force_redo = True
        spikes_key = f"{int(session):06d}-{bin_width:04d}-{delay:04d}"
        if mVocs:
            spikes_key = "mVocs_"+spikes_key

        if spikes_key not in self.neural_spikes.keys():
            # if bin_width == 1000:
            #     # treat this as a special case, where all samples are summed across time.
            #     bin_width = 20
            #     sum_spikes = True
            # # session_wise_spikes = read_cached_spikes_session_wise(bin_width=bin_width, delay=delay)
            # if session_wise_spikes is None or session not in session_wise_spikes.keys():
            spikes = self._extract_session_spikes(
                session, bin_width=bin_width, delay=delay, mVocs=mVocs
                )
            #     write_cached_spikes_session_wise(
            #         spikes, session=session, bin_width=bin_width, delay=delay
            #         )
            self.neural_spikes[spikes_key] = spikes
            # else:
            #     self.neural_spikes[spikes_key] = session_wise_spikes[session]

            # if sum_spikes:
            #     self.neural_spikes[spikes_key] = {
            #         sent_id: np.sum(spike_signal, axis=0)[None, :] for sent_id, spike_signal in self.neural_spikes[spikes_key].items()
            #         }
        # saving num of channels for the session..
        # self.num_channels[session] = next(iter(self.neural_spikes[spikes_key].values())).shape[-1]
        self.num_channels[session] = self.get_dataset_object(session).num_channels
        return self.neural_spikes[spikes_key]
    
    def get_num_channels(self, session, mVocs=False):
        """Returns the number of channels in the dataset."""
        session = str(int(session))
        if session not in self.num_channels.keys():
            _ = self.get_session_spikes(session=session, mVocs=mVocs)
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

    def get_all_neural_spikes(
            self, bin_width=20, threshold=0.068, force_redo=False, **kwargs
        ):
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
        if spikes is None or area not in spikes.keys() or force_redo:
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
    
    def get_neural_data_for_repeated_trials(
            self, session, bin_width=20, delay=0, stim_ids: list=None, mVocs=False
            ):
        """Retrieves neural data only for sent-stimuli with repeated
        trials i.e. test_sent_IDs"""
        if mVocs:
            if stim_ids is None:
                stim_ids = self.metadata.mVoc_test_stimIds
            return self.get_dataset_object(session=session).get_repeated_mVoc_trials(
                    stim_ids, bin_width=bin_width, delay=delay
                    )
        else:
            if stim_ids is None:
                stim_ids = self.test_sent_IDs
            return self.get_dataset_object(session=session).get_repeated_trials(
                sents=stim_ids, bin_width=bin_width, delay=delay)
    

    def get_stimulus_audio_with_history(self, stim_number=12, num_previous_stimuli=9, session = '180613'):
        """Returns long audio with previous stimuli concatenated with dead intervals in between"""
        # stim_number = 12
        # num_previous_stimuli = 9
        
        dataset = self.get_dataset_object(session)
        ordered_sent_IDs, dead_intervals, trial_IDs = dataset.get_ordered_sent_IDs_and_trial_IDs()

        initial_dead_time = 0.3 # seconds
        sampling_rate = 16000

        num_dead_samples = int(initial_dead_time*sampling_rate)
        dead_stimulus = np.zeros(num_dead_samples)
        # starting with dead stimulus...
        long_audio = dead_stimulus
        total_duration = 0.3
        for num in range(stim_number - num_previous_stimuli, stim_number):
            stim_id = ordered_sent_IDs[num]
            
            stim_audio = self.metadata.stim_audio(stim_id)
            long_audio = np.concatenate([long_audio, stim_audio, dead_stimulus])

            stim_duration = self.metadata.stim_duration(stim_id)
            total_duration += (stim_duration+0.3)

        stim_id = ordered_sent_IDs[stim_number]
        stim_audio = self.metadata.stim_audio(stim_id)
        long_audio = np.concatenate([long_audio, stim_audio])
        total_duration += stim_duration

        return long_audio, total_duration, ordered_sent_IDs
    
    def get_contextualized_stim_audio(self, include_repeated_trials=False):
        """Get long stimulus audio, with dead intervals in between.

        Args:
            include_repeated_trials (bool): If True, include repeated trails in all the returned results
    
        """
        # stim_number = 12
        # num_previous_stimuli = 9
        # since all sessions present stimuli in the same order,
        # we could use any session here...
        session = '180613'
        dataset = self.get_dataset_object(session)
        dead_interval = 0.3 # seconds
        sampling_rate = dataset.fs  # 16000
        total_num_stimuli = self.metadata.sent_IDs.size
        if include_repeated_trials:
            total_num_stimuli += 100

        ordered_sent_IDs = dataset.ordered_sent_IDs

        num_dead_samples = int(dead_interval*sampling_rate)
        dead_stimulus = np.zeros(num_dead_samples)
        # starting with dead stimulus...
        long_audio = dead_stimulus
        total_duration = dead_interval
        for num in range(total_num_stimuli):
            stim_id = ordered_sent_IDs[num]
            
            stim_audio = self.metadata.stim_audio(stim_id)
            stim_duration = self.metadata.stim_duration(stim_id)
            
            if num == total_num_stimuli - 1:    # for last (499th) sentence..
                long_audio = np.concatenate([long_audio, stim_audio])
                total_duration += stim_duration
            else:
                long_audio = np.concatenate([long_audio, stim_audio, dead_stimulus])
                total_duration += (stim_duration+dead_interval)

        # audio duration before every sentence ID
        duration_before_sent = {}
        for i, sent in enumerate(ordered_sent_IDs[:total_num_stimuli]):
            if i<1:
                duration_before_sent[sent] = [dead_interval]
            else:
                previous_sent = ordered_sent_IDs[i-1]
                previous_sent_duration = self.metadata.stim_duration(previous_sent)
                tmp = duration_before_sent[previous_sent][-1] + dead_interval + previous_sent_duration
                # note that the durations before sents are stored as lists, to accomodate sents with repititions..
                # for previous sent, taking the most recent value, the last value in the list...
                if sent not in duration_before_sent.keys():
                    duration_before_sent[sent] = [tmp]
                else:
                    duration_before_sent[sent].append(tmp)

        return long_audio, total_duration, ordered_sent_IDs, duration_before_sent

    
    def get_features_for_long_stimulus(
            self, model_name, stim_number=12, num_previous_stimuli=9, session = '180613'
            ):
        """Returns features for the 'long' stimulus, returns only for last stimulus and discards the rest."""

        long_audio, total_duration, ordered_sent_IDs = self.get_stimulus_audio_with_history(
            stim_number=stim_number, num_previous_stimuli=num_previous_stimuli, session = session
        )

        features = self.get_DNN_obj(model_name).extract_features_for_audio(long_audio, total_duration)


        return features, ordered_sent_IDs[stim_number], total_duration

    def get_context_depedent_normalizer(self, model_name, bin_width=20):
        """Read context dependent normalizer from the cached results.
        Raises exception if results not stored.
        
        Args:
            model_name: str = model name
            bin_width: int = bin width in ms
        """
        return read_context_dependent_normalizer(model_name=model_name, bin_width=bin_width)


    # def get_neural_prediction(
    #         self, model_name, session, bin_width: int, sents: list,
    #         layer_IDs: list=None, force_reload: bool=False
    #     ):
    #     """
    #     Returns prediction for neural activity, for the specified setting. 

    #     Args:
    #         model_name: str = select model name.
    #         session: str = session ID
    #         bin_width: int= bin_width in ms
    #         sents (list int): index of sentence IDs
    #         layer_IDs: list = layer IDs to get the predictions for.
    #         force_reload: bool = force load the network features or not.

    #     Returns:
    #         ndarray : (time, ch, layers) Prdicted neural activity 
    #     """

    #     reg_obj = self.get_DNN_obj(model_name)
    #     predicted_spikes = reg_obj.neural_prediction(
    #         session, bin_width=bin_width, sents=sents, layer_IDs=layer_IDs,
    #         force_reload=force_reload
    #         )
    #     return predicted_spikes



