import numpy as np
import cupy as cp
import torch
import gc
import os
import time
import yaml
import pandas as pd
from scipy import linalg, signal
from transformers import Speech2TextForConditionalGeneration, Speech2TextProcessor,Wav2Vec2Processor, Wav2Vec2ForCTC


# local
from auditory_cortex import config_dir
from auditory_cortex.dataset import Neural_Data
# from auditory_cortex.feature_extractors import Feature_Extractor_S2T,Feature_Extractor_GRU,FeatureExtractorW2L
from auditory_cortex.feature_extractors import FeatureExtractor
import auditory_cortex.feature_extractors as feature_extractors
import auditory_cortex.utils as utils

from sklearn.decomposition import PCA
# import rnn_model.speech_recognition as speech_recognition
import matplotlib.pyplot as plt
import torchaudio

class transformer_regression():
    def __init__(self, model_name='speech2text', load_features = True, delay_features=False):

        config_file = os.path.join(config_dir, f"{model_name}_config.yml")
        with open(config_file, 'r') as f:
            self.config = yaml.load(f, yaml.FullLoader)
                        
        self.data_dir = self.config['neural_data_dir']




        # self.session = subject
        # self.data_dir = dir
        # self.dir = os.path.join(self.data_dir, subject)
        # print("Regression object...")
        # This dataset will be used for accessing stimuli to extract features...
        self.dataset = Neural_Data(self.data_dir, '180810')
        self.sents = np.arange(1,500)
        self.spike_datasets = {}

        print(f"Creating regression obj for: '{model_name}'")
        # self.model = model
        self.model_extractor = FeatureExtractor(model_name)
        self.model_name = model_name
        self.layers = self.model_extractor.layers
        self.layer_ids = self.model_extractor.layer_ids
        self.receptive_fields = self.model_extractor.receptive_fields
        self.features_delay_trim = None
        self.use_pca = self.model_extractor.use_pca
        if self.use_pca:
            self.pca_comps = self.model_extractor.pca_comps
            self.pca = {}
            self.feature_dims = self.pca_comps
        # self.num_channels = self.dataset.num_channels
        self.num_layers = len(self.layers)
        self.B = {}

        if load_features:
            # print("Loading model features now...!")
        #     # self.load_features()
        #     self.raw_features = self.extract_features()
            self.load_features(delay_features=delay_features)


    ##############################
    ## New functions...
    ##############################
    def get_features(self, layer):
        try:
            feats = self.features[layer]
        except AttributeError:
            raise AttributeError("Run 'load_features_and_spikes()' method before using hidden features...")
        return feats

    def get_neural_spikes(
            self, session, bin_width=20, delay=0, sents=None, force_reload=False, numpy=False
        ):
        """Retrieves neural spikes for the argument session, loads spikes 
        if not already loaded or force_reload=True."""
        session = str(session)
        # check if session is already loaded, reuse it, otherwise clear all sessions to saved memory.
        if session not in self.list_loaded_sessions()  or force_reload:
            # de-allocate memory for the previously loaded sessions...
            # for sess in self.spike_datasets.keys():
            #     del self.spike_datasets[sess]
            #     del self.num_channels[sess]
            self.spike_datasets.clear()
            gc.collect()
            # de-allocation of memory ends here...

            self.spike_datasets = {}
            self.num_channels = {}
            self._load_dataset_session(session)
            self.get_dataset_object(session).extract_spikes(bin_width, delay, sents=sents)
            self.num_channels[session] = self.get_dataset_object(session).num_channels

        spikes = self.spike_datasets[session].unroll_spikes(sents=sents, features_delay_trim=self.features_delay_trim)
        # else:
        #     print(f"Creating new dataset object for session-{session}...")
        #     self.spike_datasets[session] = Neural_Data(self.data_dir, session)
        #     spikes = self.spike_datasets[session].load_spikes(sents=sents)
        if not numpy:
            spikes = cp.array(spikes)

        return spikes
    
    
    def list_loaded_sessions(self):
        """Returns the list of sessions for which neural data has
        been loaded."""
        return self.spike_datasets.keys()
    
    def _load_dataset_session(self, session):
        """Create dataset object for the 'session'"""
        self.spike_datasets[session] = Neural_Data(self.data_dir, session)

    def get_normalizer(self, session, sents=None, bin_width=20, delay=0):
        """Compute dist. of normalizer and return median."""
        if session not in self.list_loaded_sessions():
            self._load_dataset_session(session)
            _ = self.get_dataset_object(session).extract_spikes(bin_width, delay, sents=sents)
        return self.spike_datasets[session].get_normalizer(sents=sents, bin_width=bin_width, delay=delay)
        

    def get_dataset_object(self, session):
        """Returns spike dataset object if neural data for the input 
        session has already been loaded, otherwise return False."""
        try:
            return self.spike_datasets[session]
        except:
            raise AttributeError(f"Create dataset object for session-{session} before using it.")

    def extract_features(self):
        """
        Returns all layer features for given 'sents'

        Args:
            sents (list, optional): List of sentence ID's to get the features for. 

        Returns:
            List of dict: List index corresponds to layer number carrying 
                            dict of extracted features for all sentences. 
        """
        sents = self.sents
        features = [{} for _ in range(self.num_layers)]
        for x, i in enumerate(sents):
            self.model_extractor.translate(self.dataset.audio(i), grad = False)
            for j in range(self.num_layers):
                features[j][i] = self.model_extractor.get_features(j)

        return features

    def resample(self, raw_features, bin_width, delay_features=False):
        """
        resample all layer features to specific bin_width

        Args:
            bin_width (float): width of data samples in ms (1000/sampling_rate).

        Returns:
            List of dict: all layer features (resampled at required sampling_rate).
        """
        resampled_features = [{} for _ in range(len(self.layers))]
        # computing features delay: based on half of RF for all the layers all layers..
        if delay_features:
            layer_delays = (np.array(self.receptive_fields)/(2.0*bin_width)).astype(int)
            max_layer_delay = np.max(layer_delays)
            self.features_delay_trim = max_layer_delay
            print(f"Layer-wise delays (in samples) will be: {layer_delays} ")

        bin_width = bin_width/1000 # ms
        for sent in raw_features[0].keys():
            n = int(np.ceil(round(self.dataset.duration(sent)/bin_width, 3)))
            for j, l in enumerate(self.layers):
                tmp = signal.resample(raw_features[j][sent],n, axis=0)
                # mean = np.mean(tmp, axis=0)
                # resampled_features[j][sent] = tmp #- mean
                if delay_features:
                    seq_len = tmp.shape[0]
                    # delay by removing last samples, and trimming the extra length at the start..
                    tmp =   tmp[max_layer_delay-layer_delays[j] : seq_len-layer_delays[j]]
                    
                resampled_features[j][sent] = tmp
        return resampled_features

    def unroll_features(self, sents = None, numpy=True, return_dict=False, train_pca=False):
        """
        Unroll and concatenate time axis of extracted features.

        Args:
            sents (List of int ID's): ID's of sentences 

        Returns:
            dict: 
        """
        if sents is None:
            sents = self.sents
        # sampled_features = self.resample(bin_width)
        feats = {}
        for j, l in enumerate(self.layers):
            feats[j] = np.concatenate([self.sampled_features[j][sent] for sent in sents], axis=0)
            if self.use_pca:
                if train_pca:
                    self.pca[j] = PCA(n_components=self.pca_comps)
                    feats[j] = self.pca[j].fit_transform(feats[j])
                else:
                     feats[j] = self.pca[j].transform(feats[j])
            if not numpy:
                feats[j] = cp.array(feats[j])
        if not return_dict:
            feats = np.stack([feats[i] for i in range(self.num_layers)], axis=0)
        return feats
        
    def load_features(self, bin_width=20, delay_features=False):
        print(f"Loading ANN features at bin-width: {bin_width}")
        # if sents is None:
        #     sents = self.sents
        if delay_features:
            print("Features Delay requested: \n \
                  - Delaying features by half of RF for each layer \n \
                  - Trimming spikes by max feature delay (across all layers) ")
        raw_features = self.extract_features()
        if not self.use_pca:
            self.feature_dims = raw_features[0][1].shape[1]
        self.sampled_features = self.resample(raw_features, bin_width, delay_features=delay_features)
        # self.features = self.unroll_features(sents = sents, numpy=numpy, return_dict=True)

    # def load_spikes(self, bin_width=20, delay=0, offset=0, sents=None, numpy=False):
    #     if sents is None:
    #         sents = self.sents
    #     self.raw_spikes = self.extract_spikes(bin_width=bin_width, delay=delay,
    #                 offset=offset, sents=sents, numpy=numpy)
    #     return self.unroll_spikes(numpy=numpy)

    # def load_features_and_spikes(self, session, bin_width=20, delay=0, offset=0, sents=None, load_raw=False, numpy=True):
    #     session = str(session)
    #     if sents is None:
    #         sents = self.sents
    #     self.features = self.load_features(bin_width=bin_width, sents=sents, load_raw=load_raw, numpy=numpy)
    #     # self.spikes = self.load_spikes(bin_width=bin_width, delay=delay, offset=offset, sents=sents, numpy=numpy)
    #     self.spikes[session] = self.get_neural_spikes(session, sents=sents, numpy=numpy)
        
    # def features_to_cp(self, features):
    #     # converts complicated data structure containing np-arrays to cp-arrays
    #     features_cp = []
    #     for l in range(len(features)):
    #         feats = {}
    #         for k,v in features[l].items():
    #             feats[k] = cp.array(v)
    #         features_cp.append(feats)
    #     return features_cp

    # def spikes_to_cp(self, spikes):
    #     # converts complicated data structure containing np-arrays to cp-arrays
    #     spikes_cp = {}
    #     for k,v in spikes.items():
    #         spikes_cp[k] = cp.array(v)
    #     return spikes_cp


    def save_corr_coeffs(self, win, delay, file_path, null_dist=False):
        print(f"Working on win: {win}ms, delay: {delay}ms")
        self.load_features_and_spikes(bin_width=win, delay=delay)
        # train_cc, val_cc, test_cc = self.corr_coeffs()
        train_cc = []
        val_cc = []
        test_cc = []
        for layer in range(len(self.layers)):
            print(f"Computing correlations for layer {layer}")
            a,b,c = self.compute_cc_norm_layer(layer=layer, null_dist=null_dist)
            train_cc.append(a.squeeze())
            val_cc.append(b.squeeze())
            test_cc.append(c.squeeze())
        corr = {'train': np.array(train_cc), 'val': np.array(val_cc), 'test': np.array(test_cc),
                'win': win, 'delay': delay}
        data = utils.write_to_disk(corr, file_path)

    def compute_cc_norm_layer(self, layer, sp=1, normalize=False, null_dist = False):
        """
        Compute correlation coefficients for the whole layer,
        optimized linear regression using 'lstsq'

        Args:
            layer (int): index of the layer
        """
        n_channels = self.dataset.num_channels
        x = self.features[layer]
        y = np.stack([self.spikes[i] for i in range(n_channels)], axis=1)
        # x.shape  = n_samples x n_dims
        # y.shape = n_samples x channels

        if null_dist:
            np.random.shuffle(y)

        # provide 'sp' for normalized correlation coefficient...!
        r2t = np.zeros((1, n_channels))
        r2v = np.zeros((1, n_channels))
        r2tt = np.zeros((1, n_channels))

        m = int(x.shape[0])
        n2 = int(m*0.9)
        x_test = x[n2:,:]
        y_test = y[n2:,:]    

        # signal power, will be used for normalization
        #sp = self.dataset.signal_power(win, channel)
        for i in range(5):
            a = int(i*0.2*n2)
            b = int((i+1)*0.2*n2)

            x_val = x[a:b,:] 
            y_val = y[a:b,:] 
            
            x_train = np.concatenate((x[:a,:], x[b:n2,:]), axis=0)
            y_train = np.concatenate((y[:a], y[b:n2]), axis=0)
            
            # Linear Regression...!
            B = utils.regression_param(x_train, y_train)
            y_hat_train = self.predict(x_train, B)
            y_hat_val = self.predict(x_val, B)
            y_hat_test = self.predict(x_test, B)
            
            #Normalized correlation coefficient
            r2t += utils.cc_norm(y_hat_train, y_train, sp, normalize=normalize)
            r2v += utils.cc_norm(y_hat_val, y_val, sp, normalize=normalize)
            r2tt += utils.cc_norm(y_hat_test, y_test, sp, normalize=normalize)
            
        r2t /= 5
        r2v /= 5
        r2tt /= 5

        return r2t, r2v,r2tt  

    def linear_shift_null_dist(self, layer, N=50, bin_width=40, delay=0):
        """
        Compute correlations for null dist 'num_repeats' times
            for the whole layer,

        Args:
            layer (int): index of the layer
            num_repeats (int): number of simulations
        """
        null_dist = pd.DataFrame(columns=['session','layer','channel','bin_width','delay','shift','method','cc'], dtype=np.float64)
        n_channels = self.dataset.num_channels
        session = float(self.session)
        x = self.get_features(layer)
        y = np.stack([self.spikes[:,i] for i in range(n_channels)], axis=1)
        # x.shape  = n_samples x n_dims
        # y.shape = n_samples x channels
        T = x.shape[0]
        r2t = np.zeros((n_channels, 2*N+1))
        for s in range(-N,N+1):
            score = utils.fit_and_score(x[N+s:T-N+s], y[N:T-N])
            data = pd.DataFrame(np.array([
                                    np.ones_like(score)*session,
                                    np.ones_like(score)*layer,
                                    np.arange(n_channels),
                                    np.ones_like(score)*bin_width,
                                    np.ones_like(score)*delay,
                                    np.ones_like(score)*s,
                                    n_channels*['linear_shift'],
                                    score
                                ]).transpose(),
                                columns=null_dist.columns#['layer','channel','bin_width','delay','shift','method','cc']
            )
            null_dist = pd.concat([null_dist, data], axis=0, ignore_index=True)
        return null_dist

    def circular_shift_null_dist(self, layer, N=100, bin_width=20, delay=0):
        """
        Compute correlations for null dist 'N' times using circular shift method
            for the whole layer,

        Args:
            layer (int): index of the layer
            num_repeats (int): number of simulations
        """
        null_dist = pd.DataFrame(columns=['session','layer','channel','bin_width','delay','shift','method','cc'])
        n_channels = self.dataset.num_channels
        session = float(self.session)
        x = self.get_features(layer)
        y = np.stack([self.spikes[:,i] for i in range(n_channels)], axis=1)
        # x.shape  = n_samples x n_dims
        # y.shape = n_samples x channels
        T = x.shape[0]
        # dim = x.shape[1]
        # if dim > 500:
        #   factor = 8
        #   x = utils.down_sample(x.transpose(), factor).transpose()
        # r2t = np.zeros((n_channels, 2*N+1))

        for n in range(N):
            s = np.random.randint(int(0.2*T), int(0.7*T))
            score = utils.fit_and_score(np.concatenate([x[s:T],x[0:s]], axis=0), y)
            data = pd.DataFrame(np.array([
                                    np.ones_like(score)*session,
                                    np.ones_like(score)*layer,
                                    np.arange(n_channels),
                                    np.ones_like(score)*bin_width,
                                    np.ones_like(score)*delay,
                                    np.ones_like(score)*(n+1),
                                    n_channels*['circular_shift'],
                                    score
                                ]).transpose(),
                                columns=null_dist.columns#['layer','channel','bin_width','delay','shift','method','cc']
                                )
            null_dist = pd.concat([null_dist, data], axis=0, ignore_index=True)

        float_cols = ['session','layer','channel','bin_width','delay','shift','cc']
        for col in float_cols:
            null_dist[col] = null_dist[col].astype('float64')
        return null_dist


    def get_betas(self, session, use_cpu=False):
        """
        Returns betas for all channels of the layer,

        Args:
            layer (int): index of the layer
        """
        features = self.unroll_features(numpy=use_cpu)
        spikes = self.get_neural_spikes(session, numpy=use_cpu)
        print("loading features and spikes for beta calculation in regression class...")
        # print(type(features))
        # print(type(spikes))
        B = utils.reg(features, spikes)
        return B
        # n_channels = self.dataset.num_channels
    #     # x = self.features[layer]
    #     # y = np.stack([self.spikes[i] for i in range(n_channels)], axis=1)
    #     # # x.shape  = n_samples x n_dims
    #     # # y.shape = n_samples x channels
    #     # B = utils.regression_param(x, y)
    #     return B
    # def get_repeated_trials(self, sents=None, bin_width=20, delay=0):
    #     """Get repeated trials for given sents as 'ndarray'. """
    #     if sents is None:
    #         sents = [12,13,32,43,56,163,212,218,287,308]
    #     spikes_dict = {}
    #     min_repeats = 500   #repetition of trials (mostly it is 11)
    #     for s in sents:
    #         spikes_sentence = self.dataset.retrieve_spike_counts_for_all_trials(sent=s, win = bin_width, delay=delay)
    #         spikes_dict[s] = np.stack([spikes_sentence[ch] for ch in range(self.num_channels)], axis=-1)
    #         if spikes_dict[s].shape[0] < min_repeats:
    #             min_repeats = spikes_dict[s].shape[0] 
    #     all_repeated_trials = np.concatenate([spikes_dict[s][:min_repeats,:,:] for s in sents], axis=1)
    #     return all_repeated_trials

    # def get_normalizer(self, sents=None, bin_width=20, delay=0):
    #     """Compute dist. of normalizer and return median."""
    #     if sents is None:
    #         sents = [12,13,32,43,56,163,212,218,287,308]
    #     all_repeated_trials = self.get_repeated_trials(sents=sents, bin_width=bin_width, delay=delay)
    #     normalizer_all = utils.inter_trial_corr(all_repeated_trials)
    #     normalizer_all_med = np.median(normalizer_all, axis=0)
    #     return normalizer_all_med

    

    def cross_validated_regression(self, session, bin_width=40, delay=0, k=10, num_lmbdas=20, N=10, N_sents=500,
                return_dict=False, numpy=False, sents=None):
        """
        Returns distribution of correlations for all (12) layers and all channels

        Args:
            session:                session id (int or str) 
            bin_width (int):        bin width in ms.
            delay (int):            delay (ms) (post onset time) to extract neural activity
            k (int):                k-fold cross validation parameter
            lmbdas (list):          list of lmbdas to consider for cross-validation
            N (int):                Number of iterations of cross-validation (to get the distribution)
            load_features (bool):   flag for loading features (required if features and spikes not already loaded)
            return_dict (bool):     flag to return dict (ready to save format) when true, otherewise return 
                                    distribution of correlations computed.

        Returns:
            corr_coeff (3d-array):  distribution of correlations for all layers and channels (if return_dict=False)
            corr (dict):  median(corr_coeff) stored in dict, along with other details, ready to save (if return_dict=True)
        """

        # if load_features:
        #     start_load = time.time()
        #     self.load_features(bin_width=bin_width, sents=sents, load_raw=True, numpy=numpy)
        #     # self.load_features_and_spikes(bin_width=bin_width, delay=delay, numpy=numpy)
        #     end_load = time.time()
        #     print(f"It takes {end_load - start_load:.2f} sec to load features...!")
        # self.spikes[session] = self.get_neural_spikes(session, sents=sents, numpy=numpy)
        if numpy:
            module = np
        else:
            module = cp

        if sents is None:
            sents = self.sents
        if N_sents > len(sents):
            N_sents = len(sents)
        
        session = str(session)
        # this creates a new dataset object and extracts the spikes
        _ = self.get_neural_spikes(session, bin_width=bin_width, delay=delay)

        num_channels = self.spike_datasets[session].num_channels

        # feature_dims = self.sampled_features[0].shape[1]
        lmbdas = module.logspace(start=-4, stop=-1, num=num_lmbdas)
        B = module.zeros((self.num_layers, self.feature_dims, num_channels))
        corr_coeff = np.zeros((N, num_channels, self.num_layers))
        corr_coeff_train = np.zeros((N, num_channels,self.num_layers))
        # stimuli = np.array(list(self.raw_features[0].keys()))

        stimuli = np.random.permutation(sents)[0:N_sents]
        mapping_sents = int(N_sents*0.7) # 70% test set...!
        # size_of_chunk = int(mapping_sents/k)
        print(f"# of iterations requested: {N}, \n# of lambda samples per iteration: {len(lmbdas)}")
        time_itr = 0
        time_lmbda = 0
        time_map = 0
        # time_fold = 0
        for n in range(N): 
            print(f"Itr: {n+1}:")
            start_itr = time.time()
            
            np.random.shuffle(stimuli)
            mapping_set = stimuli[:mapping_sents]
            test_set = stimuli[mapping_sents:]
            
            # lmbda_loss = module.zeros(((len(lmbdas), num_channels, self.num_layers)))
            start_lmbda = time.time()
            lmbda_loss = self.k_fold_CV(session, mapping_set=mapping_set, lmbdas=lmbdas, k=k)
            
            end_lmbda = time.time()
            time_lmbda += end_lmbda-start_lmbda
            optimal_lmbdas = lmbdas[np.argmin(lmbda_loss, axis=0)]
            start_map = time.time()
            # Loading Mapping set...!
            mapping_x = self.unroll_features(sents=mapping_set, numpy=numpy, train_pca=True)
            # mapping_x = module.stack([mapping_x[i] for i in range(self.num_layers)], axis=0)
            # mapping_y = self.unroll_spikes(session, sents=mapping_set, numpy=numpy)
            mapping_y = self.get_neural_spikes(session, sents=mapping_set, numpy=numpy)
            
            #computing betas
            for l in range(self.num_layers):
                for ch in range(num_channels):
                    B[l,:,ch] = utils.reg(mapping_x[l,:,:], mapping_y[:,ch], optimal_lmbdas[ch,l])
            self.B = B

            # Loading test set...!
            test_x = self.unroll_features(sents=test_set, numpy=numpy)
            # test_x = module.stack([test_x[i] for i in range(self.num_layers)], axis=0)
            # test_y = self.unroll_spikes(sents=test_set, numpy=numpy)
            test_y = self.get_neural_spikes(session, sents=test_set, numpy=numpy) 

            train_pred = utils.predict(mapping_x, B)
            test_pred = utils.predict(test_x, B)
            
            corr_coeff[n] = utils.cc_norm(test_y,test_pred)
            corr_coeff_train[n] = utils.cc_norm(mapping_y, train_pred)
            end_map = time.time()
            end_itr = time.time()
            time_map += end_map - start_map
            time_itr += (end_itr - start_itr)
        # deallocate the memory of Neural data for current session, this will save memory used.
        del self.spike_datasets[session]
        #         print(f"itr-{n}: It takes {(end_itr - start_itr):.2f} seconds for all lambdas")
        # print(f"It takes (on avg.) {time_fold/(k*N*len(lmbdas)):.2f} sec for each step of cross validation (1 fold)")
        print(f"It takes (on avg.) {time_lmbda/(N):.2f} sec (all lmbdas). (time for {k}-folds)")
        print(f"It takes (on avg.) {time_map/(N):.2f} sec/mapping.")
        print(f"It takes (on avg.) {time_itr/(N*60):.2f} minutes/iteration...!")
        corr_coeff = cp.asnumpy(corr_coeff.transpose((0,2,1)))
        corr_coeff_train = cp.asnumpy(corr_coeff_train.transpose((0,2,1)))
        if return_dict:
            corr_coeff = np.median(corr_coeff, axis=0)
            corr_coeff_train = np.median(corr_coeff_train, axis=0)
            corr = {'test_cc_raw': corr_coeff,
                    'train_cc_raw': corr_coeff_train,
                    'win': bin_width,
                    'delay': delay, 
                    'session': session,
                    'model': self.model_name,
                    'N_sents': N_sents,
                    'layer_ids': self.layer_ids
                    }
            return corr
        return corr_coeff, B



    def k_fold_CV(self, session, mapping_set, lmbdas, k=5, use_cpu=False):
        """Return MSE loss for k-fold CV regression.

        Args:
            mapping_set (list): sent ID to be used for CV
            lmbda (float): regularization paramter
            k (int): # of folds
            use_cpu (bool): default 'False', use numpy or cupy? 

        Returns:
            avg. MSE loss for validation set.     
        """
        print(f"K_fold for session: {session}")
        num_channels = self.spike_datasets[session].num_channels
        lmbda_loss = np.zeros(((len(lmbdas), num_channels, self.num_layers)))
        size_of_chunk = int(len(mapping_set)/k)
        for i, lmbda in enumerate(lmbdas):
            loss = 0
            for r in range(k):
                start_fold = time.time()
                if r<(k-1):
                    val_set = mapping_set[r*size_of_chunk:(r+1)*size_of_chunk]
                else:
                    val_set = mapping_set[r*size_of_chunk:]
                
                train_set = mapping_set[np.isin(mapping_set, val_set, invert=True)]
                train_x = self.unroll_features(sents=train_set, numpy=use_cpu, train_pca=True)
                # train_x = np.stack([train_x[i] for i in range(self.num_layers)], axis=0)
                train_y = self.get_neural_spikes(session, sents=train_set, numpy=use_cpu)

                Beta = utils.reg(train_x, train_y, lmbda)

                # de-allocate train_x and train_y to reduce memroy utilization...
                del train_x
                del train_y
                gc.collect()
                # de-allocation of memory ends here...


                val_x = self.unroll_features(sents=val_set, numpy=use_cpu)
                # val_x = np.stack([val_x[i] for i in range(self.num_layers)], axis=0)
                val_y = self.get_neural_spikes(session, sents=val_set, numpy=use_cpu)
                


                # train_y = self.unroll_spikes(sents=train_set, numpy=use_cpu)
                # val_y = self.unroll_spikes(sents=val_set, numpy=use_cpu)


                val_pred = utils.predict(val_x, Beta)
                # to be defined...
                loss += utils.mse_loss(val_y, val_pred)
            lmbda_loss[i] = cp.asnumpy((loss/k))
        return lmbda_loss

    def map_and_score(self, mapping_set, test_set, optimal_lmbdas, use_cpu=False):
        feature_dims = self.features[0].shape[1]
        B = cp.zeros((self.num_layers, feature_dims, self.num_channels))
        corr_coeff = np.zeros((self.num_channels, self.num_layers))
        mapping_x = self.unroll_features(mapping_set, numpy=use_cpu)
        mapping_x = np.stack([mapping_x[i] for i in range(self.num_layers)], axis=0)
        mapping_y = self.unroll_spikes(sents=mapping_set, numpy=use_cpu)
        
        test_x = self.unroll_features(sents=test_set, numpy=use_cpu)
        # test_x = np.stack([test_x[i] for i in range(self.num_layers)], axis=0)
        test_y = self.unroll_spikes(sents=test_set, numpy=use_cpu) 
        
        for l in range(self.num_layers):
            for ch in range(self.num_channels):
                B[l,:,ch] = utils.reg(mapping_x[l,:,:], mapping_y[:,ch], optimal_lmbdas[ch,l])
        
        test_pred = utils.predict(test_x, B)
        corr_coeff = utils.cc_norm(test_y,test_pred)
        return corr_coeff, cp.asnumpy(B)


    def grid_search_CV(self, bin_width=20, delays=None, k=5, num_lmbdas=10, N=1, N_sents=500,
                load_features=True, sents=None, use_cpu=False):
        if sents is None:
            sents = self.sents
        if N_sents > len(sents):
            N_sents = len(sents)
        if delays is None:
            delays = [0, 10, 20, 30]    
        if load_features:
            start_load = time.time()
            self.features = self.load_features(sents = sents, bin_width=bin_width, numpy=use_cpu)
            end_load = time.time()
            print(f"It takes {end_load - start_load:.2f} sec to load features...!")

        lmbdas = cp.logspace(start=-4, stop=-1, num=num_lmbdas)
        # B = cp.zeros((12, 250, self.num_channels))
        corr_coeff = np.zeros((len(delays), self.num_channels,12))
        stimuli = np.random.permutation(sents)[0:N_sents]
        mapping_sents = int(N_sents*0.7) # 70% test set...!

        mapping_set = stimuli[0:mapping_sents]
        test_set = stimuli[mapping_sents:]

        losses = np.zeros((len(delays), len(lmbdas), self.num_channels, 12))
        strt = time.time()
        delay_time = 0

        for i, delay in enumerate(delays):
            delay_str = time.time()
            self.raw_spikes = self.extract_spikes(bin_width=bin_width, delay=delay,
                    numpy=use_cpu)
            losses[i,:,:,:] = self.k_fold_CV(mapping_set=mapping_set, lmbdas=lmbdas, k=k)
            
            optimal_lmbdas = lmbdas[np.argmin(losses[i,:,:,:], axis=0)]
            corr_coeff[i,:,:], self.B[int(delay)] =  self.map_and_score(mapping_set,
                    test_set, optimal_lmbdas)
            
            delay_end = time.time()
            delay_time += delay_end - delay_str
        delay_time /= len(delays)
        end = time.time()
        print(f"It takes {delay_time:.2f} sec (on avg.) to run all lmbdas for specific delay...!")
        print(f"It takes {end - strt :.2f}sec to run {k}-folds for all lmbdas and delays...!")

        return corr_coeff, losses
    ##########



    def neural_prediction(self, sent):
        """
        Returns prediction for neural activity 

        Args:
            sent (int): index of sentence ID 

        Returns:
            ndarray : (layers, ch, time) Prdicted neural activity 
        """
        features = self.unroll_features(sents=sent)
        # features = np.stack([dict_feats[i] for i in range(12)], axis=0)
        return cp.asnumpy(utils.predict(features, self.B))

    #########################################    ##############################



    def simply_spikes(self, sent_s=1, sent_e=499, ch=0, delay=0, def_w=40, offset=0):
        spikes ={}
        for x,i in enumerate(range(sent_s,sent_e)):
            spikes[x] = torch.tensor(self.dataset.retrieve_spike_counts(sent=i,win=def_w,delay=delay,early_spikes=False,
                                                                        offset=offset)[ch])
        spikes = torch.cat([spikes[i] for i in range(sent_e - sent_s)], dim = 0).numpy()
        return spikes

    def all_channel_spikes(self, bin_width=40, delay=0, offset=0, sents = None):
        if sents is None:
            sents = self.sents
        spikes = []
        result = {}
        for x,i in enumerate(sents):
            spikes.append(self.dataset.retrieve_spike_counts(sent=i,win=bin_width,delay=delay,early_spikes=False,offset=offset))
        for ch in range(self.dataset.num_channels):
            result[ch] = np.concatenate([spikes[i][ch] for i in range(len(spikes))], axis=0)
        return result

    # def extract_spikes(self, bin_width=40, delay=0, offset=0, sents = None, numpy=True):
    #     if sents is None:
    #         sents = self.sents
    #     raw_spikes = {}
    #     for x,i in enumerate(sents):
    #         spikes = self.dataset.retrieve_spike_counts(sent=i,win=bin_width,delay=delay,
    #                                                     early_spikes=False,offset=offset)
    #         tmp = np.stack([spikes[ch] for ch in range(self.dataset.num_channels)], axis=1)
    #         if not numpy:
    #             tmp = cp.array(tmp)
    #         mean = np.mean(tmp, axis=0)    
    #         raw_spikes[i] = tmp #- mean
    #     return raw_spikes

    # def unroll_spikes(self, sents=None, numpy=True):
    #     """
    #     Unroll and concatenate time axis of extracted spikes.

    #     Args:
    #         sents (List): indices of sents.

    #     Returns:
            
    #     """
    #     if sents is None:
    #         sents = self.raw_spikes.keys()
    #     if numpy:
    #         spikes = np.concatenate([self.raw_spikes[sent] for sent in sents], axis=0)
    #     else:
    #         spikes = cp.concatenate([self.raw_spikes[sent] for sent in sents], axis=0)
    #     return spikes

    # def unroll_spikes(self, session, sents=mapping_set, numpy=numpy):


    def unroll_spikes_cp(self, sents=None):
        """
        Unroll and concatenate time axis of extracted spikes.

        Args:
            sents (List): indices of sents.

        Returns:
            
        """
        if sents is None:
            sents = self.raw_spikes.keys()
        spikes = cp.array(np.concatenate([self.raw_spikes[sent] for sent in sents], axis=0))
        return spikes

    def get_cc_norm_layer(self, layer, win, delay=0, sents= np.arange(1,499),normalize = False, load_features=False):
        """
        | Gives correlation coefficients for given
        | 'layer' (and all channels)
        """
        print(f"Computing correlations for layer:{layer} ...")
        sp = 1
        num_channels = self.dataset.num_channels
        train_cc_norm = np.zeros(num_channels)
        val_cc_norm = np.zeros(num_channels)
        test_cc_norm = np.zeros(num_channels)

        feats, spikes = self.get_feats_and_spikes(layer, win, delay, sents, load_features)
        if normalize:
            sp_all_channels = self.dataset.signal_power(win)
        for ch in range(num_channels):
            if normalize:
                sp = sp_all_channels[ch]
            train_cc_norm[ch], val_cc_norm[ch], test_cc_norm[ch] = self.compute_cc_norm(feats, spikes[ch], sp, normalize=normalize)
            
        return train_cc_norm, val_cc_norm, test_cc_norm

    def get_feats_and_spikes(self, layer, win, delay=0, sents= np.arange(1,499), load_features=False):
        """
        | Gives features and spikes data for given
        | 'layer' and all channels.
        """
        if load_features:
            print("Loading model layer features now...!")
            self.features = self.load_features()

        def_w, offset = self.model_extractor.def_bin_width(layer)            
        k = int(win/def_w)    # 40 is the min, bin size for 'Speech2Text' transformer model 
        feats = self.features[layer]
        spikes = self.all_channel_spikes(sents=sents, delay=delay, bin_width=def_w, offset=offset)
        if k>1:
            feats = utils.down_sample(feats, k)
            for ch in range(self.dataset.num_channels):
                spikes[ch] = utils.down_sample(spikes[ch],k)

        return feats, spikes

    def get_cc_norm(self, layer, win, channel, delay=0, normalize = False, sents= np.arange(1,499), load_features=False):
        """
        | Gives correlation coefficient for given 
        | 'layer' and 'channel' 
        """
        if load_features:
            print("Loading model layer features now...!")
            self.features = self.load_features()
        def_w, offset = self.model_extractor.def_bin_width(layer)       
        k = int(win/def_w)    # 40 is the min, bin size for 'Speech2Text' transformer model 
        feats = self.features[layer]   
        y = self.all_channel_spikes(sents=sents, delay=delay, bin_width=def_w, offset=offset)[channel]
        if k>1:
            feats = utils.down_sample(feats, k)
            y = utils.down_sample(y,k)
        if normalize:
            sp = self.dataset.signal_power(win)[channel]
        else:
            sp = 1
        r2t, r2v,r2tt = self.compute_cc_norm(feats, y, sp, normalize=normalize)
        return r2t, r2v,r2tt

    def compute_cc_norm(self, x, y, sp=1, normalize=False):
        """
        | return correlation coefficient for given 
        | data (x,y), and optional 'sp' and 'normalize' flag.
        """
        # provide 'sp' for normalized correlation coefficient...!
        r2t = np.zeros(1)
        r2v = np.zeros(1)
        r2tt = np.zeros(1)

        m = int(x.shape[0])
        n2 = int(m*0.9)
        x_test = x[n2:, :]
        y_test = y[n2:]    

        # signal power, will be used for normalization
        #sp = self.dataset.signal_power(win, channel)
        for i in range(5):
            a = int(i*0.2*n2)
            b = int((i+1)*0.2*n2)

            x_val = x[a:b, :] 
            y_val = y[a:b] 
            
            x_train = np.concatenate((x[:a,:], x[b:n2,:]), axis=0)
            y_train = np.concatenate((y[:a], y[b:n2]))
            
            # Linear Regression...!
            B = utils.regression_param(x_train, y_train)
            y_hat_train = self.predict(x_train, B)
            y_hat_val = self.predict(x_val, B)
            y_hat_test = self.predict(x_test, B)
            
            #Normalized correlation coefficient
            r2t += utils.cc_norm(y_hat_train, y_train, sp, normalize=normalize)
            r2v += utils.cc_norm(y_hat_val, y_val, sp, normalize=normalize)
            r2tt += utils.cc_norm(y_hat_test, y_test, sp, normalize=normalize)
            
        r2t /= 5
        r2v /= 5
        r2tt /= 5

        return r2t, r2v,r2tt  


    def get_Poiss_scores_layer(self, layer, win, delay=0, sents= np.arange(1,499), load_features=False):
        print(f"Computing Poisson scores for layer:{layer} ...")
        num_channels = self.dataset.num_channels
        train_scores = np.zeros(num_channels)
        val_scores = np.zeros(num_channels)
        test_scores = np.zeros(num_channels)

        feats, spikes = self.get_feats_and_spikes(layer, win, delay, sents, load_features)
        for ch in range(num_channels):
            train_scores[ch], val_scores[ch], test_scores[ch] = self.compute_poiss_scores(feats, spikes[ch])
            
        return train_scores, val_scores, test_scores  

    def compute_poiss_scores(self, x, y):
        # provide 'sp' for normalized correlation coefficient...!
        ps_t = np.zeros(1)
        ps_v = np.zeros(1)
        ps_tt = np.zeros(1)

        m = int(x.shape[0])
        n2 = int(m*0.9)
        x_test = x[n2:, :]
        y_test = y[n2:]    

        # signal power, will be used for normalization
        #sp = self.dataset.signal_power(win, channel)
        for i in range(5):
            a = int(i*0.2*n2)
            b = int((i+1)*0.2*n2)
            
            x_val = x[a:b, :] 
            y_val = y[a:b] 
            
            x_train = np.concatenate((x[:a,:], x[b:n2,:]), axis=0)
            y_train = np.concatenate((y[:a], y[b:n2]))
            
            # Poisson Regression...!
            
            poiss_model = utils.poiss_regression(x_train, y_train)


            
            #Poisson Scores
            ps_t += utils.poisson_regression_score(poiss_model, x_train, y_train)
            ps_v += utils.poisson_regression_score(poiss_model, x_val, y_val)
            ps_tt += utils.poisson_regression_score(poiss_model, x_test, y_test)
            
        ps_t /= 5
        ps_v /= 5
        ps_tt /= 5

        return ps_t, ps_v,ps_tt

    def predict(self, X, B):
        return X@B

    def r2(self, labels, predictions):
        score = 0.0
        mean = np.mean(labels)
        denom = np.sum(np.square(labels - mean))
        num = np.sum(np.square(labels - predictions))
        score = 1 - num/denom
        return score
    def regression_score(self, X,y, B):
        y_hat = self.predict(X,B)
        return self.r2(y, y_hat)


    ################################################################################################
    ##############Redundant functions....!
    #############################################################


    def demean_spikes(self, sent_s=1, sent_e=499, ch=0, w = 40):
        spikes ={}
        spk_mean = {}
        for x,i in enumerate(range(sent_s,sent_e)):
            spikes[x] = torch.tensor(self.dataset.retrieve_spike_counts(sent=i, win=w ,early_spikes=False)[ch])
            spk_mean[x] = torch.mean(spikes[x], dim = 0)
            spikes[x] = spikes[x] - spk_mean[x]
        spikes = torch.cat([spikes[i] for i in range(sent_e - sent_s)], dim = 0).numpy()
        return spikes



    def benchmark_r2_score(self, w = 40, sent = 12):
        #These sentences have repeated trials...!
        #sents = [12,13,32,43,56,163,212,218,287,308]
        r2_scores = np.zeros(self.dataset.num_channels)
        #trials = obj.dataset.get_trials(13)
        spkk = self.dataset.retrieve_spike_counts_for_all_trials(sent=sent, w=w)

        for i in range(self.dataset.num_channels):
            h1 = np.mean(spkk[i][0:6], axis=0)
            h2 = np.mean(spkk[i][6:], axis=0)
            r2_scores[i] = self.r2(h1,h2)
        return r2_scores


    def compute_r2(self, layer, win):
        k = int(win/40)    # 40 is the min, bin size for 'Speech2Text' transformer model 
        # print(f"k = {k}")
        r2t = np.zeros(self.dataset.num_channels)
        r2v = np.zeros(self.dataset.num_channels)
        pct = np.zeros(self.dataset.num_channels)
        pcv = np.zeros(self.dataset.num_channels)

        #downsamples if k>1 
        if k >1:
            feats = utils.down_sample(self.features[layer], k)
        else:
            feats = self.features[layer]

        m = int(feats.shape[0] *0.75)
        x_train = feats[0:m, :]
        x_test = feats[m:, :]

        for i in range(self.dataset.num_channels):
            y = self.simply_spikes(ch=i)
            if k>1:
                y = utils.down_sample(y,k)
            y_train = y[0:m]
            y_test = y[m:]
            B = utils.regression_param(x_train, y_train)

            r2t[i] = self.regression_score(x_train, y_train, B)
            r2v[i] = self.regression_score(x_test, y_test, B)
            pct[i] = (np.corrcoef(self.predict(x_train, B), y_train)[0,1])**2
            pcv[i] = (np.corrcoef(self.predict(x_test, B), y_test)[0,1])**2
        return r2t, r2v, pct, pcv
