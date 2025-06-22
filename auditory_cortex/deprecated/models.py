import gc
import os
import time
import yaml
import torch
import numpy as np
import pandas as pd
from scipy import linalg, signal
# import matplotlib.pyplot as plt

# from sklearn.decomposition import PCA

# local
from auditory_cortex import config
import auditory_cortex.utils as utils
# from auditory_cortex.dataset import NeuralData
from auditory_cortex.neural_data import NeuralData
import auditory_cortex.deprecated.dataloader as dataloader
from auditory_cortex import LPF_analysis_bw
# from auditory_cortex.computational_models.feature_extractors import DNNFeatureExtractor

# import GPU specific packages...
from auditory_cortex import hpc_cluster
if hpc_cluster:
    import cupy as cp

# import tensorflow.compat.v2 as tf
# tf.enable_v2_behavior()
import tensorflow_probability as tfp
tfd = tfp.distributions


class Regression():
    def __init__(
            self,
            model_name = 'speech2text',
            load_features = False,
            delay_features = False,
            audio_zeropad = False,
            resample=True,
            saved_checkpoint=None
        ):
        """
        """
        # self.data_dir = config['neural_data_dir']
        # self.dataset = NeuralData('180810')
        # self.sents = np.arange(1,500)
        # self.spike_datasets = {}

        print(f"Creating regression obj for: '{model_name}'")
        # self.model = model
        self.dataloader = dataloader.DataLoader()
        # self.model_extractor = FeatureExtractor(model_name, saved_checkpoint=saved_checkpoint)
        self.model_name = model_name
        # self.layers = self.model_extractor.layers
        # self.layer_ids = self.model_extractor.layer_ids
        # self.receptive_fields = self.model_extractor.receptive_fields
        self.features_delay_trim = None
        self.audio_padding_duration = 0
        # self.use_pca = self.model_extractor.use_pca
        # if self.use_pca:
        #     self.pca_comps = self.model_extractor.pca_comps
        #     self.pca = {}
        #     self.feature_dims = self.pca_comps
        # self.num_channels = self.dataset.num_channels
        # self.num_layers = len(self.layers)
        self.B = {}

        # if load_features:
        #     self.load_features(delay_features=delay_features, audio_zeropad=audio_zeropad,
        #                        resample=resample)





    ### Methods for the loading and accessing features..


    def unroll_features(
            self, stim_ids, bin_width=20, delay=0, numpy=True,
            return_dict=False, layer_IDs=None, third=None,
            force_reload=False, shuffled=False, mVocs=False,
            LPF=False
            ):
        """
        Unroll and concatenate time axis of extracted features.

        Args:
            stim_ids (List of int ID's): ID's of stimuli
            bin_width: int = bin width in ms.
            delay: int = delay in ms. 
            layer_IDs (list): Layer of layer indices 
            third (int):    Default=None, section of sents to compute prediction corr.
            LPF: bool = If true, low-pass-filters features to the bin width specified
                and resamples again at predefined bin-width (e.g. 10ms)

        Returns:
            dict: if return_dict=True, otherwise, return ndarray =(layers, samples, ndim)
        """
        if layer_IDs is None:
            layer_IDs = self.dataloader.get_layer_IDs(self.model_name)
        feats = {}
        all_layer_features = self.dataloader.get_resampled_DNN_features(
                    self.model_name, bin_width=bin_width, force_reload=force_reload,
                    shuffled=shuffled, mVocs=mVocs, LPF=LPF
                    )
        # Handling delay: Trimming the extra bins instead of zero padding 
        if LPF:
            bin_width = LPF_analysis_bw
        trim = int(np.ceil(delay/bin_width))
        for layer_ID in layer_IDs:

            layer_feats = all_layer_features[layer_ID]
            if third is None:
                if trim !=0:
                    feats[layer_ID] = np.concatenate([layer_feats[stim_id][:-trim] for stim_id in stim_ids], axis=0)
                else:
                    feats[layer_ID] = np.concatenate([layer_feats[stim_id] for stim_id in stim_ids], axis=0)
            else:
                pass

            if not numpy:
                feats[layer_ID] = cp.array(feats[layer_ID])
        if not return_dict:
            feats = np.stack([feats[id] for id in layer_IDs], axis=0)
        return feats
    
    def get_layer_IDs(self):
        """Retrieves layer ID for self.model_name"""
        return self.dataloader.get_layer_IDs(self.model_name)
    

    ### Methods for the getting neural data (spikes)


    def unroll_spikes(
            self, session, stim_ids, bin_width=20, delay=0, 
                numpy=False, mVocs=False, LPF=False):
        """
        Retrieves stimulus-wise neural spikes for the specified session, unrolls
        and concatenatess time axis of neural spikes.

        Args:
            session: int = ID of recording site (session)
            sents (List of int ID's): ID's of stimuli  
            bin_width: int = bin width in ms.
            delay: int = neural delay in ms.
            numpy: bool = Setting numpy=True would mean using CPU and not GPU.
            mVocs: bool = If True, returns spikes for mVocs

        Returns:
            return ndarray =(samples, n_channels)
        """
        session = str(session)
        if LPF:
            bin_width = LPF_analysis_bw

        raw_spikes = self.dataloader.get_session_spikes(
            session=session, bin_width=bin_width, delay=delay, mVocs=mVocs
            )
        # Handling delay: Trimming the extra bins instead of zero padding 
        trim = int(np.ceil(delay/bin_width))
        # Deprecated: [:-1] is used to drop the partial bins at the end of each sequence.
        if trim !=0:
            spikes = np.concatenate([raw_spikes[stim_id][:-trim] for stim_id in stim_ids], axis=0).astype(np.float32)
        else:
            spikes = np.concatenate([raw_spikes[stim_id] for stim_id in stim_ids], axis=0).astype(np.float32)

        if not numpy:
            spikes = cp.array(spikes)
        return spikes
    
    def get_test_data(
            self, session, bin_width=20, delay=0, numpy=False,
            layer_IDs=None, shuffled=False, mVocs=False,
            LPF=False,
        ):
        """
        Retrieves DNN features and all trials of neural spikes for held out sentences.

        Args:
            session: int = ID of recording site (session) 
            bin_width: int = bin width in ms.
            delay: int = neural delay in ms.
            numpy: bool = Setting numpy=True would mean using CPU and not GPU.
            mVocs: bool = If True, returns data for mVocs otherwise for timit stimuli.

        Returns:
            features: ndarray =(layers, samples, features)
            repeated_trials: ndarray =(num_trials, samples, n_channels)
        """
        features_bw = bin_width
        if LPF:
            bin_width = LPF_analysis_bw

        if mVocs:
            stim_ids = []
            mVocs_ids = self.dataloader.metadata.mVoc_test_stimIds
            for mVoc_id in mVocs_ids:
                # since we need tr id [0,780) and not mVocs id [1, 303]
                # and only taking the first tr...
                stim_ids.append(self.dataloader.metadata.get_mVoc_tr_id(mVoc_id)[0])
            spikes_all_trials = self.get_test_mVocs_spikes(session, bin_width=bin_width, delay=delay, numpy=numpy)
        else:
            stim_ids = self.dataloader.test_sent_IDs
            spikes_all_trials = self.get_test_sent_spikes(session, bin_width=bin_width, delay=delay, numpy=numpy)

        features = self.unroll_features(
            stim_ids, bin_width=features_bw, delay=delay, numpy=numpy,
            layer_IDs=layer_IDs, shuffled=shuffled, mVocs=mVocs,
            LPF=LPF
            )
        return features, spikes_all_trials

    def get_test_mVocs_spikes(
            self, session, bin_width=20, delay=0, numpy=False):
        """
        Retrieves all trials of neural spikes for held out mVocs.

        Args:
            session: int = ID of recording site (session) 
            bin_width: int = bin width in ms.
            delay: int = neural delay in ms.
            numpy: bool = Setting numpy=True would mean using CPU and not GPU.

        Returns:
            return ndarray =(num_trials, samples, n_channels)
        """
        mVocs_spikes = {}
        mVocs_ids = self.dataloader.metadata.mVoc_test_stimIds
        for mVocs in mVocs_ids:
            tr_ids = self.dataloader.metadata.get_mVoc_tr_id(mVocs)
            all_trial_spikes = []
            for tr in tr_ids:
                all_trial_spikes.append(self.unroll_spikes(
                    session, [tr], bin_width=bin_width, delay=delay, 
                    numpy=numpy, mVocs=True
                ))
            mVocs_spikes[mVocs] = np.stack(all_trial_spikes, axis=0)
        repeated_trials = np.concatenate([mVocs_spikes[mVocs] for mVocs in mVocs_ids], axis=1).astype(np.float32)
        # if not numpy:
        #     repeated_trials = cp.array(repeated_trials)
        return repeated_trials




    def get_test_sent_spikes(
            self, session, bin_width=20, delay=0, numpy=False):
        """
        Retrieves all trials of neural spikes for held out sentences.

        Args:
            session: int = ID of recording site (session) 
            bin_width: int = bin width in ms.
            delay: int = neural delay in ms.
            numpy: bool = Setting numpy=True would mean using CPU and not GPU.

        Returns:
            return ndarray =(num_trials, samples, n_channels)
        """
        sent_spikes = {}
        sent_ids = self.dataloader.test_sent_IDs
        for sent in sent_ids:
            sent_spikes[sent] = self.dataloader.get_neural_data_for_repeated_trials(
	        session, bin_width=bin_width, delay=delay, stim_ids=[sent]
            )
        trim = int(np.ceil(delay/bin_width))
        # Deprecated: [:-1] is used to drop the partial bins at the end of each sequence.
        if trim !=0:
            repeated_trials = np.concatenate([sent_spikes[sent][:,:-trim,:] for sent in sent_ids], axis=1).astype(np.float32)
        else:
            repeated_trials = np.concatenate([sent_spikes[sent] for sent in sent_ids], axis=1).astype(np.float32)
        if not numpy:
            repeated_trials = cp.array(repeated_trials)
        return repeated_trials
        
        
    def get_normalizer(self, session, sents=None, bin_width=20, delay=0, n=1000):
        """Compute dist. of normalizer for correlations (repeatability of neural
        spikes), and return median."""
        all_repeated_trials = self.dataloader.get_neural_data_for_repeated_trials(
            session, bin_width=bin_width, delay=delay
            )
        normalizer_all = Regression.inter_trial_corr(all_repeated_trials, n=n)
        normalizer_all_med = np.median(normalizer_all, axis=0)
        return normalizer_all_med


    def get_features(self, layer):
        try:
            layer = self.get_layer_index(layer)
            feats = self.sampled_features[layer]
        except AttributeError:
            raise AttributeError("Run 'load_features()' method before using hidden features...")
        return feats


    

    def regression(self, x, y, lmbda, lr, max_iterations, poisson=False):
        """Returns model coefficients for linear regression or poisson regression."""
        if poisson:
            lr = tf.constant(lr)
            model_coefficients, predicted_linear_response, is_converged, iter = tfp.glm.fit(
                    model_matrix=tf.convert_to_tensor(x),
                    response=tf.convert_to_tensor(y),
                    model=tfp.glm.Poisson(),
                    l2_regularizer=lmbda,
                    # learning_rate=lr,
                    maximum_iterations=max_iterations
                )
            return model_coefficients.numpy()
        else:
            return utils.reg(x, y, lmbda)
        


    def cross_validated_regression_LPF_features(
            self, session, bin_width=20, delay=0, num_folds=5, num_lmbdas=8,
            iterations=10, N_sents=500, return_dict=False, numpy=False,
            third=None, layer_IDs=None, poisson=False, lr=0.6,
            max_iterations=50, shuffled=False, test_trial=None, mVocs=False
        ):
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

            third (int) [1,2,3]:    Default: None, section of test sents to compute corr for.  
            poisson: bool =         Choose type of regression, poisson if True else: linear

        Returns:
            corr_coeff (3d-array):  distribution of correlations for all layers and channels (if return_dict=False)
            corr (dict):  median(corr_coeff) stored in dict, along with other details, ready to save (if return_dict=True)
        """
        if poisson:
            numpy=True

        if numpy:
            module = np
        else:
            module = cp

        if shuffled:
            print(f"Running regression for shuffled weights..")


        # if sents is None:
        #     sents = self.dataloader.sent_IDs
        # if N_sents > len(sents):
        #     N_sents = len(sents)

        # if test_sents is None:
        #     test_sents = self.dataloader.test_sent_IDs
        
        # this creates a new dataset object and extracts the spikes
        session = str(session)
        # spikes = self.unroll_spikes(session, bin_width=bin_width, delay=delay)
        raw_spikes = self.dataloader.get_session_spikes(session, bin_width=bin_width, delay=delay, mVocs=mVocs)
        num_channels = self.dataloader.get_num_channels(session, mVocs=mVocs)

        
        if mVocs:
            stim_ids = self.dataloader.metadata.mVocTrialIds
            test_ids = self.dataloader.metadata.mVoc_test_trIds

            # exclude the missing trial IDs from list of Ids
            missing_trial_ids = self.dataloader.get_dataset_object(session=session).missing_trial_ids
            stim_ids = stim_ids[np.isin(stim_ids, missing_trial_ids, invert=True)]
            test_ids = test_ids[np.isin(test_ids, missing_trial_ids, invert=True)]

        else:
            stim_ids = self.dataloader.sent_IDs
            test_ids = self.dataloader.test_sent_IDs
        N_sents = len(stim_ids)
        # _ = self.get_neural_spikes(session, bin_width=bin_width, delay=delay)
        # num_channels = self.spike_datasets[session].num_channels
        if layer_IDs is None:
            layer_IDs = self.get_layer_IDs()

        # loading features for any one sent=12, to get feature_dims..
        feature_dims = self.unroll_features(
            stim_ids=[12], bin_width=bin_width, delay=delay, layer_IDs=layer_IDs, shuffled=shuffled,
            mVocs=mVocs
            ).shape[-1]
        
        # feature_dims = self.sampled_features[0].shape[1]
        # lmbdas = module.logspace(start=-7, stop=-1, num=num_lmbdas)
        lmbdas = module.logspace(start=-10, stop=10, num=21)
        
        # lmbdas = module.logspace(start=-num_lmbdas//4, stop=num_lmbdas, num=int(1.25*num_lmbdas)+1)
        # lmbdas = module.logspace(start=1, stop=8, num=7)
        # lmbdas = module.logspace(start=5, stop=6, num=2)
        B = module.zeros((len(layer_IDs), feature_dims, num_channels))
        corr_coeff = np.zeros((iterations, num_channels, len(layer_IDs)))
        corr_coeff_mVocs = np.zeros((iterations, num_channels, len(layer_IDs)))

        # Deprecated
        # poiss_entropy = np.zeros((iterations, num_channels, len(layer_IDs)))
        # poiss_entropy_baseline = np.zeros((iterations, num_channels, len(layer_IDs)))
        # uncertainty_per_spike = np.zeros((iterations, num_channels, len(layer_IDs)))
        # bits_per_spike_NLB = np.zeros((iterations, num_channels, len(layer_IDs)))
        corr_coeff_train = np.zeros((iterations, num_channels, len(layer_IDs)))
        # stimuli = np.array(list(self.raw_features[0].keys()))

        stimuli = np.random.permutation(stim_ids)[0:N_sents]
        # mapping_sents = int(N_sents*0.7) # 70% test set...!
        # size_of_chunk = int(mapping_sents/k)
        print(f"# of iterations requested: {iterations}, \n \
                # of lambda samples per iteration: {len(lmbdas)}")
        time_itr = 0
        time_lmbda = 0
        time_map = 0
        # time_fold = 0
        for n in range(iterations): 
            print(f"Itr: {n+1}:")
            start_itr = time.time()
        
            np.random.shuffle(stimuli)
            
            # option to fix the test set..!
            mapping_set = stimuli[np.isin(stimuli, test_ids, invert=True)]
            test_set = test_ids
            
            # lmbda_loss = module.zeros(((len(lmbdas), num_channels, self.num_layers)))
            start_lmbda = time.time()
            if poisson:
                lmbda_loss = self.k_fold_CV_poisson_regression(
                        session, bin_width=bin_width, delay=delay, mapping_set=mapping_set,
                        lmbdas=lmbdas, num_folds=num_folds, layer_IDs=layer_IDs,
                        lr = lr, max_iterations=max_iterations, shuffled=shuffled
                    )
            else:
                lmbda_loss = self.k_fold_CV(
                        session, bin_width=bin_width, delay=delay, mapping_set=mapping_set,
                        lmbdas=lmbdas, num_folds=num_folds, layer_IDs=layer_IDs,
                        shuffled=shuffled, mVocs=mVocs
                    )
            end_lmbda = time.time()
            time_lmbda += end_lmbda-start_lmbda
            optimal_lmbdas = lmbdas[np.argmin(lmbda_loss, axis=0)]
            start_map = time.time()
            # Loading Mapping set...!
            mapping_x = self.unroll_features(
                stim_ids=mapping_set,
                bin_width=bin_width, delay=delay, numpy=numpy,
                layer_IDs=layer_IDs, shuffled=shuffled,
                mVocs=mVocs
                )
            mapping_y = self.unroll_spikes(
                session, stim_ids=mapping_set, 
                bin_width=bin_width, delay=delay,
                numpy=numpy, mVocs=mVocs
                )
            
            # mapping_y = self.get_neural_spikes(session, bin_width=bin_width, sents=mapping_set, numpy=numpy)
            #computing betas
            for l in range(len(layer_IDs)):
                for ch in range(num_channels):
                    B[l,:,ch] = self.regression(
                        mapping_x[l,:,:], mapping_y[:,ch], optimal_lmbdas[ch,l],
                        lr, max_iterations,
                        poisson=poisson,
                        )
                    # B[l,:,ch] = utils.reg(mapping_x[l,:,:], mapping_y[:,ch], optimal_lmbdas[ch,l])
            # self.B[session] = cp.asnumpy(self.B[session])
            
            train_pred = utils.predict(mapping_x, B)
            corr_coeff_train[n] = utils.cc_norm(mapping_y, train_pred)

            del mapping_x
            del mapping_y
            gc.collect()

            if mVocs:
                excluded_sessions = ['190726', '200213']
                if session in excluded_sessions:
                    print(f"Excluding session: {session} from testing on mVocs..")
                    corr_coeff_mVocs[n] = np.zeros_like(corr_coeff[n])
                else:
                    # testing for mVocs responses...
                    test_x, test_y_all_trials = self.get_test_data(
                        session, bin_width=bin_width, delay=delay, numpy=numpy,
                        layer_IDs=layer_IDs, shuffled=shuffled, mVocs=True,
                        )
                    test_pred = utils.predict(test_x, B)
                    corr_coeff_mVocs[n] = utils.compute_avg_test_corr(
                        test_y_all_trials, test_pred, test_trial, mVocs=True
                        )
                    corr_coeff[n] = np.zeros_like(corr_coeff_mVocs[n])
            else:
                # print(f"mVocs is True..")
                # testing for timit responses...
                test_x, test_y_all_trials = self.get_test_data(
                    session, bin_width=bin_width, delay=delay, numpy=numpy,
                    layer_IDs=layer_IDs, shuffled=shuffled, mVocs=False,
                    )
                test_pred = utils.predict(test_x, B)
                corr_coeff[n] = utils.compute_avg_test_corr(
                    test_y_all_trials, test_pred, test_trial, mVocs=False
                    )
                corr_coeff_mVocs[n] = np.zeros_like(corr_coeff[n])
            

            # 
            # ##### Deprecated
            #             # Loading test set...!
            # test_x = self.unroll_features(
            #     stim_ids=test_set,
            #     bin_width=bin_width, delay=delay, numpy=numpy,
            #     third=third, layer_IDs=layer_IDs, shuffled=shuffled,
            #     mVocs=mVocs
            #     )
            # test_y = self.unroll_spikes(
            #     session, stim_ids=test_set, bin_width=bin_width, delay=delay, numpy=numpy,
            #     mVocs=mVocs
            #     )
            # # test_y = self.get_neural_spikes(session, bin_width=bin_width, sents=test_set, numpy=numpy, third=third) 

            # test_pred = utils.predict(test_x, B)

            # poiss_entropy[n] = utils.poisson_cross_entropy(test_y, test_pred)
            
            # Nsamples, Nchannels = test_y.shape
            # data_sums = cp.asnumpy(np.sum(test_y, axis=0, keepdims=True))
            # data_means = data_sums/Nsamples
            # # using mean spikes as predicted means
            # poiss_entropy_baseline[n] = utils.poisson_cross_entropy(test_y, np.log(data_means[...,None]))
            # uncertainty_per_spike[n] = poiss_entropy[n]/(data_sums.T + 1.e-6)/np.log(2)
            # bits_per_spike_NLB[n] = (poiss_entropy_baseline[n] - poiss_entropy[n])/(data_sums.T + 1.e-6)/np.log(2)

            # if poisson:
            #     corr_coeff[n] = utils.cc_norm(test_y, np.exp(test_pred))
            # else:
            #     if mVocs:
            #         print(f"mVocs is True..")
            #         test_x, test_y_all_trials = self.get_test_data(
            #             session, bin_width=bin_width, delay=delay, numpy=numpy,
            #             layer_IDs=layer_IDs, shuffled=shuffled, mVocs=mVocs,
            #             )
            #         test_pred = utils.predict(test_x, B)
            #         corr_coeff[n] = utils.compute_avg_test_corr(
            #             test_y_all_trials, test_pred, test_trial, mVocs=mVocs
            #             )

            #     else:
            #         # computing avg test corr across all trials..
            #         test_y_all_trials = self.get_test_sent_spikes(
            #             session, bin_width=bin_width, delay=delay, numpy=numpy)
                        
            #         corr_coeff[n] = utils.compute_avg_test_corr(
            #             test_y_all_trials, test_pred, test_trial, mVocs=mVocs
            #             )
                # previous implementation...
                # corr_coeff[n] = utils.cc_norm(test_y, test_pred)
            end_map = time.time()
            end_itr = time.time()
            time_map += end_map - start_map
            time_itr += (end_itr - start_itr)
        
        #         print(f"itr-{n}: It takes {(end_itr - start_itr):.2f} seconds for all lambdas")
        # print(f"It takes (on avg.) {time_fold/(k*N*len(lmbdas)):.2f} sec for each step of cross validation (1 fold)")
        print(f"It takes (on avg.) {time_lmbda/(iterations):.2f} sec (all lmbdas). (time for {num_folds}-folds)")
        print(f"It takes (on avg.) {time_map/(iterations):.2f} sec/mapping.")
        print(f"It takes (on avg.) {time_itr/(iterations*60):.2f} minutes/iteration...!")

        # incoming: corr_coeff = (itr, ch, layers), transposed: (itr, layers, ch) 
        corr_coeff_mVocs = cp.asnumpy(corr_coeff_mVocs.transpose((0,2,1)))
        corr_coeff_mVocs = np.median(corr_coeff_mVocs, axis=0)
        corr_coeff = cp.asnumpy(corr_coeff.transpose((0,2,1)))
        corr_coeff = np.median(corr_coeff, axis=0)
        # same as corr_coeff
        # poiss_entropy = poiss_entropy.transpose((0,2,1))
        # poiss_entropy = np.median(poiss_entropy, axis=0)
        # poiss_entropy_baseline = np.median(poiss_entropy_baseline, axis=0)
        # uncertainty_per_spike = np.median(uncertainty_per_spike, axis=0).transpose((1,0))
        # bits_per_spike_NLB = np.median(bits_per_spike_NLB, axis=0).transpose((1,0))

        corr_coeff_train = cp.asnumpy(corr_coeff_train.transpose((0,2,1)))
        lmbda_loss = lmbda_loss.transpose((0,2,1))
        optimal_lmbdas = np.log10(cp.asnumpy(optimal_lmbdas.transpose((1,0))))
        if return_dict:
            # deallocate the memory of Neural data for current session, this will save memory used.
            # del self.spike_datasets[session]
            # saving results in a dictionary..
            corr_coeff_train = np.median(corr_coeff_train, axis=0)
            corr = {'test_cc_raw': corr_coeff,
                    'mVocs_test_cc_raw': corr_coeff_mVocs,
                    'train_cc_raw': corr_coeff_train,
                    'win': bin_width,
                    'delay': delay, 
                    'session': session,
                    'model': self.model_name,
                    'N_sents': N_sents,
                    'layer_ids': layer_IDs,
                    'opt_delays': None,
                    'opt_lmbdas': optimal_lmbdas,

                    #Deprecated
                    # 'poiss_entropy': poiss_entropy,
                    # 'uncertainty_per_spike': uncertainty_per_spike,
                    # 'bits_per_spike_NLB': bits_per_spike_NLB,
                    }
            return corr, optimal_lmbdas, lmbda_loss
        # return test_y, test_pred, B, optimal_lmbdas, lmbda_loss
        return corr_coeff, B, np.min(lmbda_loss, axis=0), test_set, corr_coeff_mVocs, optimal_lmbdas
    



    ### Methods for the computing correlations and grid search for optimal delay.

    def cross_validated_regression(
            self, session, bin_width=20, delay=0, num_folds=5, num_lmbdas=8,
            iterations=10, N_sents=500, return_dict=False, numpy=False,
            third=None, layer_IDs=None, poisson=False, lr=0.6,
            max_iterations=50, shuffled=False, test_trial=None,
            mVocs=False, LPF=False
        ):
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

            third (int) [1,2,3]:    Default: None, section of test sents to compute corr for.  
            poisson: bool =         Choose type of regression, poisson if True else: linear

        Returns:
            corr_coeff (3d-array):  distribution of correlations for all layers and channels (if return_dict=False)
            corr (dict):  median(corr_coeff) stored in dict, along with other details, ready to save (if return_dict=True)
        """
        if poisson:
            numpy=True

        if numpy:
            module = np
        else:
            module = cp

        if shuffled:
            print(f"Running regression for shuffled weights..")


        # if sents is None:
        #     sents = self.dataloader.sent_IDs
        # if N_sents > len(sents):
        #     N_sents = len(sents)

        # if test_sents is None:
        #     test_sents = self.dataloader.test_sent_IDs
        
        # this creates a new dataset object and extracts the spikes
        session = str(session)
        # spikes = self.unroll_spikes(session, bin_width=bin_width, delay=delay)
        raw_spikes = self.dataloader.get_session_spikes(session, bin_width=bin_width, delay=delay, mVocs=mVocs)
        num_channels = self.dataloader.get_num_channels(session, mVocs=mVocs)

        
        if mVocs:
            stim_ids = self.dataloader.metadata.mVocTrialIds
            test_ids = self.dataloader.metadata.mVoc_test_trIds

            # exclude the missing trial IDs from list of Ids
            missing_trial_ids = self.dataloader.get_dataset_object(session=session).missing_trial_ids
            stim_ids = stim_ids[np.isin(stim_ids, missing_trial_ids, invert=True)]
            test_ids = test_ids[np.isin(test_ids, missing_trial_ids, invert=True)]

        else:
            stim_ids = self.dataloader.sent_IDs
            test_ids = self.dataloader.test_sent_IDs
        N_sents = len(stim_ids)
        # _ = self.get_neural_spikes(session, bin_width=bin_width, delay=delay)
        # num_channels = self.spike_datasets[session].num_channels
        if layer_IDs is None:
            layer_IDs = self.get_layer_IDs()

        # loading features for any one sent=12, to get feature_dims..
        feature_dims = self.unroll_features(
            stim_ids=[12], bin_width=bin_width, delay=delay, layer_IDs=layer_IDs, shuffled=shuffled,
            mVocs=mVocs, LPF=LPF
            ).shape[-1]
        
        # feature_dims = self.sampled_features[0].shape[1]
        # lmbdas = module.logspace(start=-7, stop=-1, num=num_lmbdas)
        lmbdas = module.logspace(start=-10, stop=10, num=21)
        
        # lmbdas = module.logspace(start=-num_lmbdas//4, stop=num_lmbdas, num=int(1.25*num_lmbdas)+1)
        # lmbdas = module.logspace(start=1, stop=8, num=7)
        # lmbdas = module.logspace(start=5, stop=6, num=2)
        B = module.zeros((len(layer_IDs), feature_dims, num_channels))
        corr_coeff = np.zeros((iterations, num_channels, len(layer_IDs)))
        corr_coeff_mVocs = np.zeros((iterations, num_channels, len(layer_IDs)))

        # Deprecated
        # poiss_entropy = np.zeros((iterations, num_channels, len(layer_IDs)))
        # poiss_entropy_baseline = np.zeros((iterations, num_channels, len(layer_IDs)))
        # uncertainty_per_spike = np.zeros((iterations, num_channels, len(layer_IDs)))
        # bits_per_spike_NLB = np.zeros((iterations, num_channels, len(layer_IDs)))
        corr_coeff_train = np.zeros((iterations, num_channels, len(layer_IDs)))
        # stimuli = np.array(list(self.raw_features[0].keys()))

        stimuli = np.random.permutation(stim_ids)[0:N_sents]
        # mapping_sents = int(N_sents*0.7) # 70% test set...!
        # size_of_chunk = int(mapping_sents/k)
        print(f"# of iterations requested: {iterations}, \n \
                # of lambda samples per iteration: {len(lmbdas)}")
        time_itr = 0
        time_lmbda = 0
        time_map = 0
        # time_fold = 0
        for n in range(iterations): 
            print(f"Itr: {n+1}:")
            start_itr = time.time()
        
            np.random.shuffle(stimuli)
            
            # option to fix the test set..!
            mapping_set = stimuli[np.isin(stimuli, test_ids, invert=True)]
            test_set = test_ids
            
            # lmbda_loss = module.zeros(((len(lmbdas), num_channels, self.num_layers)))
            start_lmbda = time.time()
            if poisson:
                lmbda_loss = self.k_fold_CV_poisson_regression(
                        session, bin_width=bin_width, delay=delay, mapping_set=mapping_set,
                        lmbdas=lmbdas, num_folds=num_folds, layer_IDs=layer_IDs,
                        lr = lr, max_iterations=max_iterations, shuffled=shuffled
                    )
            else:
                lmbda_loss = self.k_fold_CV(
                        session, bin_width=bin_width, delay=delay, mapping_set=mapping_set,
                        lmbdas=lmbdas, num_folds=num_folds, layer_IDs=layer_IDs,
                        shuffled=shuffled, mVocs=mVocs, LPF=LPF
                    )
            end_lmbda = time.time()
            time_lmbda += end_lmbda-start_lmbda
            optimal_lmbdas = lmbdas[np.argmin(lmbda_loss, axis=0)]
            start_map = time.time()
            # Loading Mapping set...!
            mapping_x = self.unroll_features(
                stim_ids=mapping_set,
                bin_width=bin_width, delay=delay, numpy=numpy,
                layer_IDs=layer_IDs, shuffled=shuffled,
                mVocs=mVocs, LPF=LPF
                )
            mapping_y = self.unroll_spikes(
                session, stim_ids=mapping_set, 
                bin_width=bin_width, delay=delay,
                numpy=numpy, mVocs=mVocs, LPF=LPF
                )
            
            # mapping_y = self.get_neural_spikes(session, bin_width=bin_width, sents=mapping_set, numpy=numpy)
            #computing betas
            for l in range(len(layer_IDs)):
                for ch in range(num_channels):
                    B[l,:,ch] = self.regression(
                        mapping_x[l,:,:], mapping_y[:,ch], optimal_lmbdas[ch,l],
                        lr, max_iterations,
                        poisson=poisson,
                        )
                    # B[l,:,ch] = utils.reg(mapping_x[l,:,:], mapping_y[:,ch], optimal_lmbdas[ch,l])
            # self.B[session] = cp.asnumpy(self.B[session])
            
            train_pred = utils.predict(mapping_x, B)
            corr_coeff_train[n] = utils.cc_norm(mapping_y, train_pred)

            del mapping_x
            del mapping_y
            gc.collect()

            if mVocs:
                excluded_sessions = ['190726', '200213']
                if session in excluded_sessions:
                    print(f"Excluding session: {session} from testing on mVocs..")
                    corr_coeff_mVocs[n] = np.zeros_like(corr_coeff[n])
                else:
                    # testing for mVocs responses...
                    test_x, test_y_all_trials = self.get_test_data(
                        session, bin_width=bin_width, delay=delay, numpy=numpy,
                        layer_IDs=layer_IDs, shuffled=shuffled, mVocs=True,
                        LPF=LPF
                        )
                    test_pred = utils.predict(test_x, B)
                    corr_coeff_mVocs[n] = utils.compute_avg_test_corr(
                        test_y_all_trials, test_pred, test_trial, mVocs=True
                        )
                    corr_coeff[n] = np.zeros_like(corr_coeff_mVocs[n])
            else:
                # print(f"mVocs is True..")
                # testing for timit responses...
                test_x, test_y_all_trials = self.get_test_data(
                    session, bin_width=bin_width, delay=delay, numpy=numpy,
                    layer_IDs=layer_IDs, shuffled=shuffled, mVocs=False,
                    LPF=LPF,
                    )
                test_pred = utils.predict(test_x, B)
                corr_coeff[n] = utils.compute_avg_test_corr(
                    test_y_all_trials, test_pred, test_trial, mVocs=False
                    )
                corr_coeff_mVocs[n] = np.zeros_like(corr_coeff[n])
            

            # 
            # ##### Deprecated
            #             # Loading test set...!
            # test_x = self.unroll_features(
            #     stim_ids=test_set,
            #     bin_width=bin_width, delay=delay, numpy=numpy,
            #     third=third, layer_IDs=layer_IDs, shuffled=shuffled,
            #     mVocs=mVocs
            #     )
            # test_y = self.unroll_spikes(
            #     session, stim_ids=test_set, bin_width=bin_width, delay=delay, numpy=numpy,
            #     mVocs=mVocs
            #     )
            # # test_y = self.get_neural_spikes(session, bin_width=bin_width, sents=test_set, numpy=numpy, third=third) 

            # test_pred = utils.predict(test_x, B)

            # poiss_entropy[n] = utils.poisson_cross_entropy(test_y, test_pred)
            
            # Nsamples, Nchannels = test_y.shape
            # data_sums = cp.asnumpy(np.sum(test_y, axis=0, keepdims=True))
            # data_means = data_sums/Nsamples
            # # using mean spikes as predicted means
            # poiss_entropy_baseline[n] = utils.poisson_cross_entropy(test_y, np.log(data_means[...,None]))
            # uncertainty_per_spike[n] = poiss_entropy[n]/(data_sums.T + 1.e-6)/np.log(2)
            # bits_per_spike_NLB[n] = (poiss_entropy_baseline[n] - poiss_entropy[n])/(data_sums.T + 1.e-6)/np.log(2)

            # if poisson:
            #     corr_coeff[n] = utils.cc_norm(test_y, np.exp(test_pred))
            # else:
            #     if mVocs:
            #         print(f"mVocs is True..")
            #         test_x, test_y_all_trials = self.get_test_data(
            #             session, bin_width=bin_width, delay=delay, numpy=numpy,
            #             layer_IDs=layer_IDs, shuffled=shuffled, mVocs=mVocs,
            #             )
            #         test_pred = utils.predict(test_x, B)
            #         corr_coeff[n] = utils.compute_avg_test_corr(
            #             test_y_all_trials, test_pred, test_trial, mVocs=mVocs
            #             )

            #     else:
            #         # computing avg test corr across all trials..
            #         test_y_all_trials = self.get_test_sent_spikes(
            #             session, bin_width=bin_width, delay=delay, numpy=numpy)
                        
            #         corr_coeff[n] = utils.compute_avg_test_corr(
            #             test_y_all_trials, test_pred, test_trial, mVocs=mVocs
            #             )
                # previous implementation...
                # corr_coeff[n] = utils.cc_norm(test_y, test_pred)
            end_map = time.time()
            end_itr = time.time()
            time_map += end_map - start_map
            time_itr += (end_itr - start_itr)
        
        #         print(f"itr-{n}: It takes {(end_itr - start_itr):.2f} seconds for all lambdas")
        # print(f"It takes (on avg.) {time_fold/(k*N*len(lmbdas)):.2f} sec for each step of cross validation (1 fold)")
        print(f"It takes (on avg.) {time_lmbda/(iterations):.2f} sec (all lmbdas). (time for {num_folds}-folds)")
        print(f"It takes (on avg.) {time_map/(iterations):.2f} sec/mapping.")
        print(f"It takes (on avg.) {time_itr/(iterations*60):.2f} minutes/iteration...!")

        # incoming: corr_coeff = (itr, ch, layers), transposed: (itr, layers, ch) 
        corr_coeff_mVocs = cp.asnumpy(corr_coeff_mVocs.transpose((0,2,1)))
        corr_coeff_mVocs = np.median(corr_coeff_mVocs, axis=0)
        corr_coeff = cp.asnumpy(corr_coeff.transpose((0,2,1)))
        corr_coeff = np.median(corr_coeff, axis=0)
        # same as corr_coeff
        # poiss_entropy = poiss_entropy.transpose((0,2,1))
        # poiss_entropy = np.median(poiss_entropy, axis=0)
        # poiss_entropy_baseline = np.median(poiss_entropy_baseline, axis=0)
        # uncertainty_per_spike = np.median(uncertainty_per_spike, axis=0).transpose((1,0))
        # bits_per_spike_NLB = np.median(bits_per_spike_NLB, axis=0).transpose((1,0))

        corr_coeff_train = cp.asnumpy(corr_coeff_train.transpose((0,2,1)))
        lmbda_loss = lmbda_loss.transpose((0,2,1))
        optimal_lmbdas = np.log10(cp.asnumpy(optimal_lmbdas.transpose((1,0))))
        if return_dict:
            # deallocate the memory of Neural data for current session, this will save memory used.
            # del self.spike_datasets[session]
            # saving results in a dictionary..
            corr_coeff_train = np.median(corr_coeff_train, axis=0)
            corr = {'test_cc_raw': corr_coeff,
                    'mVocs_test_cc_raw': corr_coeff_mVocs,
                    'train_cc_raw': corr_coeff_train,
                    'win': bin_width,
                    'delay': delay, 
                    'session': session,
                    'model': self.model_name,
                    'N_sents': N_sents,
                    'layer_ids': layer_IDs,
                    'opt_delays': None,
                    'opt_lmbdas': optimal_lmbdas,

                    #Deprecated
                    # 'poiss_entropy': poiss_entropy,
                    # 'uncertainty_per_spike': uncertainty_per_spike,
                    # 'bits_per_spike_NLB': bits_per_spike_NLB,
                    }
            return corr, optimal_lmbdas, lmbda_loss
        # return test_y, test_pred, B, optimal_lmbdas, lmbda_loss
        return corr_coeff, B, np.min(lmbda_loss, axis=0), test_set, corr_coeff_mVocs, optimal_lmbdas
    # Deprecated
    #, poiss_entropy, uncertainty_per_spike, bits_per_spike_NLB, optimal_lmbdas 



    def k_fold_CV(
            self, session, bin_width, delay, mapping_set, lmbdas,
            layer_IDs=None, num_folds=5, use_cpu=False,
            shuffled=False, mVocs=False, LPF=False
        ):
        """Return MSE loss (avg.) for k-fold CV regression.

        Args:
            session (str): session ID 
            mapping_set (list): sent ID to be used for CV
            lmbdas (float): Range of regularization paramters to compare
            num_folds (int): # of folds
            use_cpu (bool): default 'False', use numpy or cupy? 
            mVocs: bool = If True, uses features and spikes for mVocs

        Returns:
            avg. MSE loss for validation set.     
        """
        print(f"{num_folds}_fold CV for session: {session}")
        # num_channels = self.spike_datasets[session].num_channels
        num_channels = self.dataloader.get_num_channels(session, mVocs=mVocs)
        if layer_IDs is None:
            layer_IDs = self.get_layer_IDs()

        lmbda_loss = np.zeros(((len(lmbdas), num_channels, len(layer_IDs))))
        size_of_chunk = int(len(mapping_set) / num_folds)
        # for i, lmbda in enumerate(lmbdas):
        #     loss = 0
        for r in range(num_folds):
            print(f"For fold={r}: ")
            # get the sent ids for train and validation folds...
            if r<(num_folds-1):
                val_set = mapping_set[r*size_of_chunk:(r+1)*size_of_chunk]
            else:
                val_set = mapping_set[r*size_of_chunk:]
            train_set = mapping_set[np.isin(mapping_set, val_set, invert=True)]

            # load features and spikes using the sent ids.
            train_x = self.unroll_features(
                stim_ids=train_set,
                bin_width=bin_width, delay=delay, numpy=use_cpu,
                layer_IDs=layer_IDs, shuffled=shuffled, mVocs=mVocs,
                LPF=LPF
                )
            train_y = self.unroll_spikes(
                session, stim_ids=train_set, bin_width=bin_width,
                delay=delay, numpy=use_cpu, mVocs=mVocs,
                LPF=LPF
                )

            # computing Betas for lmbdas and removing train_x, train_y to manage memory..
            Betas = {}
            for i, lmbda in enumerate(lmbdas):
                Betas[i] = utils.reg(train_x, train_y, lmbda)
            
            del train_x
            del train_y
            gc.collect()

            # train_y = self.get_neural_spikes(session, bin_width=bin_width, sents=train_set, numpy=use_cpu)

            val_x = self.unroll_features(
                stim_ids=val_set,
                bin_width=bin_width, delay=delay, numpy=use_cpu,
                layer_IDs=layer_IDs, shuffled=shuffled,
                mVocs=mVocs, LPF=LPF
                )
            val_y = self.unroll_spikes(
                session, stim_ids=val_set, bin_width=bin_width,
                delay=delay, numpy=use_cpu, mVocs=mVocs, LPF=LPF
                )

            # val_y = self.get_neural_spikes(session, bin_width=bin_width, sents=val_set, numpy=use_cpu)

            # for the current fold, compute/save validation loss for each lambda.
            for i, lmbda in enumerate(lmbdas):

                # Beta = utils.reg(train_x, train_y, lmbda)
                # val_pred = utils.predict(val_x, Beta)
                val_pred = utils.predict(val_x, Betas[i])

                loss = utils.mse_loss(val_y, val_pred)
                lmbda_loss[i] += cp.asnumpy((loss))

            # de-allocate train_x and train_y to reduce memroy utilization...
            del val_x
            del val_y
            gc.collect()
            # de-allocation of memory ends here...

        lmbda_loss /= num_folds            
        return lmbda_loss

##################################################################################
##############       GLIM: Poisson Regression
######################################################################################

    def k_fold_CV_poisson_regression(
            self, session, bin_width, delay, mapping_set, lmbdas,
            layer_IDs=None, num_folds=5, use_cpu=False, lr=0.6, max_iterations=100,
            shuffled=False
        ):
        """Return Neg. Log Likelihood (avg.) under poisson model for k-fold CV regression.

        Args:
            session (str): session ID 
            mapping_set (list): sent ID to be used for CV
            lmbdas (float): Range of regularization paramters to compare
            num_folds (int): # of folds
            use_cpu (bool): default 'False', use numpy or cupy? 

        Returns:
            avg. MSE loss for validation set.     
        """
        print(f"{num_folds}_fold CV for session: {session}")
        # num_channels = self.spike_datasets[session].num_channels
        num_channels = self.dataloader.get_num_channels(session)
        if layer_IDs is None:
            layer_IDs = self.get_layer_IDs()

        lr = tf.constant(lr)

        lmbdas = cp.asnumpy(lmbdas).astype(np.float32)
        lmbda_loss = np.zeros(((len(lmbdas), num_channels, len(layer_IDs))))
        size_of_chunk = int(len(mapping_set) / num_folds)
        # for i, lmbda in enumerate(lmbdas):
        #     loss = 0
        for r in range(num_folds):
            print(f"For fold={r}: ")
            # get the sent ids for train and validation folds...
            if r<(num_folds-1):
                val_set = mapping_set[r*size_of_chunk:(r+1)*size_of_chunk]
            else:
                val_set = mapping_set[r*size_of_chunk:]
            train_set = mapping_set[np.isin(mapping_set, val_set, invert=True)]

            # load features and spikes using the sent ids.
            train_x = self.unroll_features(
                stim_ids=train_set,
                bin_width=bin_width, delay=delay, numpy=True,
                layer_IDs=layer_IDs, shuffled=shuffled
                )
            train_y = self.unroll_spikes(
                session, stim_ids=train_set, bin_width=bin_width,
                delay=delay, numpy=True
                )
            train_x = train_x.astype(np.float32)
            train_y = train_y.astype(np.float32)

            val_x = self.unroll_features(
                stim_ids=val_set,
                bin_width=bin_width, delay=delay, numpy=True,
                layer_IDs=layer_IDs, shuffled=shuffled
                )
            val_y = self.unroll_spikes(
                session, stim_ids=val_set, bin_width=bin_width,
                delay=delay, numpy=True
                )

            val_x = val_x.astype(np.float32)
            val_y = val_y.astype(np.float32)
            # computing Betas for lmbdas and removing train_x, train_y to manage memory..
            num_layers = train_x.shape[0]
            num_channels = train_y.shape[-1]
            num_weights = train_x.shape[-1]

            # Betas = {}
            for i, lmbda in enumerate(lmbdas):
                model_weights = np.zeros((num_layers, num_weights, num_channels))
                lmbda_unconverged = np.zeros(((num_channels, len(layer_IDs))), dtype=np.int16)
                for l in range(num_layers):
                    x = train_x[l]
                    for ch in range(num_channels):
                        y = train_y[:,ch]

                        model_coefficients, predicted_linear_response, is_converged, iter = tfp.glm.fit(
                            model_matrix=tf.convert_to_tensor(x),
                            response=tf.convert_to_tensor(y),
                            model=tfp.glm.Poisson(),
                            l2_regularizer=lmbda,
                            maximum_iterations=max_iterations,
                            learning_rate=lr
                            )
                        if not is_converged.numpy():
                            lmbda_unconverged[:,l] = 1
                        model_weights[l,:,ch] = model_coefficients.numpy()

                    # x = np.expand_dims(x, axis=0)
                    # chs = np.arange(train_y.shape[1])
                    # num_channels = chs.size
                    # if num_channels > 30:
                    #     ch_list = []
                    #     select_chs = chs[:num_channels//2]
                    #     ch_list.append(select_chs)
                    #     select_chs = chs[num_channels//2:]
                    #     ch_list.append(select_chs)
                    # else:
                    #     ch_list = []
                    #     select_chs = chs[:num_channels//2]
                    #     ch_list.append(select_chs)
                    # for ch in ch_list:
                    #     y = np.transpose(train_y[:, ch])
                    #     print(x.shape)
                    #     print(y.shape)
                    #     model_coefficients, predicted_linear_response, is_converged, iter = tfp.glm.fit(
                    #             model_matrix=tf.convert_to_tensor(x),
                    #             response=tf.convert_to_tensor(y),
                    #             model=tfp.glm.Poisson(),
                    #             l2_regularizer=lmbda,
                    #             maximum_iterations=max_iterations,
                    #             # learning_rate=0.8
                    #     )

                    #     if not is_converged.numpy():
                    #         lmbda_unconverged[ch,l] = 1

                    #     # model_weights[l,:,ch] = model_coefficients.numpy()
                    #     model_weights[l,:,ch] = model_coefficients.numpy()

                val_x = tf.convert_to_tensor(val_x)
                val_y = tf.convert_to_tensor(val_y)
                model_weights = tf.convert_to_tensor(model_weights, dtype=np.float32)
                
                val_predicted_linear_resposne = tf.matmul(val_x, model_weights)
                # print(type(val_predicted_linear_resposne))
                log_likelihood = tfp.glm.Poisson().log_prob(val_y, val_predicted_linear_resposne).numpy()
                lmbda_loss[i] += -1*np.mean(log_likelihood, axis=1).transpose()
                lmbda_loss[i][np.where(lmbda_unconverged>0)] = num_folds*10000

        lmbda_loss /= num_folds            
        return lmbda_loss


##################################################################################
##############          Not sure if this is being used....so commenting out..
######################################################################################


    # def map_and_score(self, bin_width, mapping_set, test_set, optimal_lmbdas, use_cpu=False):
    #     feature_dims = self.features[0].shape[1]
    #     B = cp.zeros((self.num_layers, feature_dims, self.num_channels))
    #     corr_coeff = np.zeros((self.num_channels, self.num_layers))
    #     mapping_x = self.unroll_features(bin_width=bin_width, sents=mapping_set, numpy=use_cpu)
    #     # mapping_x = np.stack([mapping_x[i] for i in range(self.num_layers)], axis=0)
    #     mapping_y = self.unroll_spikes(sents=mapping_set, numpy=use_cpu)
        
    #     test_x = self.unroll_features(bin_width=bin_width, sents=test_set, numpy=use_cpu)
    #     # test_x = np.stack([test_x[i] for i in range(self.num_layers)], axis=0)
    #     test_y = self.unroll_spikes(sents=test_set, numpy=use_cpu) 
        
    #     for l in range(self.num_layers):
    #         for ch in range(self.num_channels):
    #             B[l,:,ch] = utils.reg(mapping_x[l,:,:], mapping_y[:,ch], optimal_lmbdas[ch,l])
        
    #     test_pred = utils.predict(test_x, B)
    #     corr_coeff = utils.cc_norm(test_y,test_pred)
    #     return corr_coeff, cp.asnumpy(B)


    def grid_search_CV(self,
            session,
            bin_width=20,
            delays=None,
            layer_IDs=None,
            num_lmbdas=10,
            N_sents=500,
            iterations=1,
            num_folds=5, 
            sents=None,
            numpy=False,
            return_dict=False,
            third=None,
            shuffled=False,
            test_trial=None,
            mVocs=False,
            LPF=False,
        ):
        
        if delays is None:
            delays = [0, 10, 20, 30] 
            
        if layer_IDs is None:
            layer_IDs = self.get_layer_IDs()   

        corr_coeffs = []
        corr_coeffs_mVocs = []
        losses = []
        opt_lmbdas_all_delays = []
        # Deprecated
        # poiss_entropies = []
        # uncertainty_per_spike_list = []
        # bits_per_spike_NLB_list = []
        session = str(session)
        if LPF:
            print(f"Grid search for low-pass-fitered features, predicting at {LPF_analysis_bw}ms")

        for i, delay in enumerate(delays):

            # force reload spikes at the desired delay...
            print(f"Loading neural spikes with delay: {delay}ms")
            # _ = self.get_neural_spikes(
            #         session, bin_width=bin_width, delay=delay, force_reload=True
            #     )

            # Deprecated
            # corr_coeff, _, loss, _, poiss_entropy, uncertainty_per_spike, bits_per_spike_NLB, opt_lmbdas
            
            corr_coeff, _, loss, _, corr_coeff_mVocs, opt_lmbdas = self.cross_validated_regression(
                session, bin_width=bin_width, delay=delay,layer_IDs=layer_IDs,
                num_folds=num_folds, iterations=iterations, num_lmbdas=num_lmbdas,
                return_dict=False, numpy=numpy, N_sents=N_sents, third=third,
                shuffled=shuffled, test_trial=test_trial, mVocs=mVocs, LPF=LPF
            )
            corr_coeffs.append(corr_coeff)
            corr_coeffs_mVocs.append(corr_coeff_mVocs)
            losses.append(loss)
            # Deprecated
            # poiss_entropies.append(poiss_entropy)
            # uncertainty_per_spike_list.append(uncertainty_per_spike)
            # bits_per_spike_NLB_list.append(bits_per_spike_NLB)
            opt_lmbdas_all_delays.append(opt_lmbdas)

        corr_coeffs = np.array(corr_coeffs)
        corr_coeffs_mVocs = np.array(corr_coeffs_mVocs)
        losses = np.array(losses)
        delays = np.array(delays)
        # Deprecated
        # poiss_entropies = np.array(poiss_entropies)
        # uncertainty_per_spike_list = np.array(uncertainty_per_spike_list)
        # bits_per_spike_NLB_list = np.array(bits_per_spike_NLB_list)
        opt_lmbdas_all_delays = np.array(opt_lmbdas_all_delays)
        # num_channels = self.spike_datasets[session].num_channels

        num_channels = self.dataloader.get_num_channels(session, mVocs=mVocs)
        opt_delay_indices = np.argmin(losses, axis=0)
        opt_delays = delays[opt_delay_indices]
        corr_coeffs_opt_delay = corr_coeffs[
                opt_delay_indices, np.arange(len(layer_IDs))[:, None], np.arange(num_channels)
            ]
        corr_coeffs_mVocs_opt_delay = corr_coeffs_mVocs[
                opt_delay_indices, np.arange(len(layer_IDs))[:, None], np.arange(num_channels)
            ]
        # Deprecated
        # poiss_entropy_opt_delay = poiss_entropies[
        #     opt_delay_indices, np.arange(len(layer_IDs))[:, None], np.arange(num_channels)
        # ]
        # uncertainty_per_spike_opt_delay = uncertainty_per_spike_list[
        #     opt_delay_indices, np.arange(len(layer_IDs))[:, None], np.arange(num_channels)
        # ]
        # bits_per_spike_NLB_opt_delay = bits_per_spike_NLB_list[
        #     opt_delay_indices, np.arange(len(layer_IDs))[:, None], np.arange(num_channels)
        # ]
        opt_lmbdas_opt_delays = opt_lmbdas_all_delays[
            opt_delay_indices, np.arange(len(layer_IDs))[:, None], np.arange(num_channels)
        ]
 
        if return_dict:
            # deallocate the memory of Neural data for current session, this will save memory used.
            # del self.spike_datasets[session]
            # saving results in a dictionary..
            corr_dict = {
                    'test_cc_raw': corr_coeffs_opt_delay,
                    'mVocs_test_cc_raw': corr_coeffs_mVocs_opt_delay,
                    'train_cc_raw': np.zeros_like(corr_coeffs_opt_delay),
                    'win': bin_width,
                    'delay': 0, 
                    'session': session,
                    'model': self.model_name,
                    'N_sents': N_sents,
                    'layer_ids': layer_IDs,
                    'opt_delays': opt_delays,
                    'opt_lmbdas': opt_lmbdas_opt_delays,
                    # Deprecated.
                    # 'poiss_entropy': poiss_entropy_opt_delay,
                    # 'uncertainty_per_spike': uncertainty_per_spike_opt_delay, 
                    # 'bits_per_spike_NLB': bits_per_spike_NLB_opt_delay,
                    }
            return corr_dict
        # optimal_delays = delays[np.argmin(losses, axis=0)]
        return corr_coeffs_opt_delay, opt_delays
    

    def grid_search_CV_LPF_features(self,
            session,
            bin_width=20,
            delays=None,
            layer_IDs=None,
            num_lmbdas=10,
            N_sents=500,
            iterations=1,
            num_folds=5, 
            sents=None,
            numpy=False,
            return_dict=False,
            third=None,
            shuffled=False,
            test_trial=None,
            mVocs=False,
        ):
        
        if delays is None:
            delays = [0, 10, 20, 30] 
            
        if layer_IDs is None:
            layer_IDs = self.get_layer_IDs()   

        corr_coeffs = []
        corr_coeffs_mVocs = []
        losses = []
        opt_lmbdas_all_delays = []
        # Deprecated
        # poiss_entropies = []
        # uncertainty_per_spike_list = []
        # bits_per_spike_NLB_list = []
        session = str(session)

        for i, delay in enumerate(delays):

            # force reload spikes at the desired delay...
            print(f"Loading neural spikes with delay: {delay}ms")
            # _ = self.get_neural_spikes(
            #         session, bin_width=bin_width, delay=delay, force_reload=True
            #     )

            # Deprecated
            # corr_coeff, _, loss, _, poiss_entropy, uncertainty_per_spike, bits_per_spike_NLB, opt_lmbdas
            
            corr_coeff, _, loss, _, corr_coeff_mVocs, opt_lmbdas = self.cross_validated_regression_LPF_features(
                session, bin_width=bin_width, delay=delay,layer_IDs=layer_IDs,
                num_folds=num_folds, iterations=iterations, num_lmbdas=num_lmbdas,
                return_dict=False, numpy=numpy, N_sents=N_sents, third=third,
                shuffled=shuffled, test_trial=test_trial, mVocs=mVocs
            )
            corr_coeffs.append(corr_coeff)
            corr_coeffs_mVocs.append(corr_coeff_mVocs)
            losses.append(loss)
            # Deprecated
            # poiss_entropies.append(poiss_entropy)
            # uncertainty_per_spike_list.append(uncertainty_per_spike)
            # bits_per_spike_NLB_list.append(bits_per_spike_NLB)
            opt_lmbdas_all_delays.append(opt_lmbdas)

        corr_coeffs = np.array(corr_coeffs)
        corr_coeffs_mVocs = np.array(corr_coeffs_mVocs)
        losses = np.array(losses)
        delays = np.array(delays)
        # Deprecated
        # poiss_entropies = np.array(poiss_entropies)
        # uncertainty_per_spike_list = np.array(uncertainty_per_spike_list)
        # bits_per_spike_NLB_list = np.array(bits_per_spike_NLB_list)
        opt_lmbdas_all_delays = np.array(opt_lmbdas_all_delays)
        # num_channels = self.spike_datasets[session].num_channels

        num_channels = self.dataloader.get_num_channels(session, mVocs=mVocs)
        opt_delay_indices = np.argmin(losses, axis=0)
        opt_delays = delays[opt_delay_indices]
        corr_coeffs_opt_delay = corr_coeffs[
                opt_delay_indices, np.arange(len(layer_IDs))[:, None], np.arange(num_channels)
            ]
        corr_coeffs_mVocs_opt_delay = corr_coeffs_mVocs[
                opt_delay_indices, np.arange(len(layer_IDs))[:, None], np.arange(num_channels)
            ]
        # Deprecated
        # poiss_entropy_opt_delay = poiss_entropies[
        #     opt_delay_indices, np.arange(len(layer_IDs))[:, None], np.arange(num_channels)
        # ]
        # uncertainty_per_spike_opt_delay = uncertainty_per_spike_list[
        #     opt_delay_indices, np.arange(len(layer_IDs))[:, None], np.arange(num_channels)
        # ]
        # bits_per_spike_NLB_opt_delay = bits_per_spike_NLB_list[
        #     opt_delay_indices, np.arange(len(layer_IDs))[:, None], np.arange(num_channels)
        # ]
        opt_lmbdas_opt_delays = opt_lmbdas_all_delays[
            opt_delay_indices, np.arange(len(layer_IDs))[:, None], np.arange(num_channels)
        ]
 
        if return_dict:
            # deallocate the memory of Neural data for current session, this will save memory used.
            # del self.spike_datasets[session]
            # saving results in a dictionary..
            corr_dict = {
                    'test_cc_raw': corr_coeffs_opt_delay,
                    'mVocs_test_cc_raw': corr_coeffs_mVocs_opt_delay,
                    'train_cc_raw': np.zeros_like(corr_coeffs_opt_delay),
                    'win': bin_width,
                    'delay': 0, 
                    'session': session,
                    'model': self.model_name,
                    'N_sents': N_sents,
                    'layer_ids': layer_IDs,
                    'opt_delays': opt_delays,
                    'opt_lmbdas': opt_lmbdas_opt_delays,
                    # Deprecated.
                    # 'poiss_entropy': poiss_entropy_opt_delay,
                    # 'uncertainty_per_spike': uncertainty_per_spike_opt_delay, 
                    # 'bits_per_spike_NLB': bits_per_spike_NLB_opt_delay,
                    }
            return corr_dict
        # optimal_delays = delays[np.argmin(losses, axis=0)]
        return corr_coeffs_opt_delay, opt_delays
    


    def get_betas(
            self, session, bin_width, delay=0, layer_IDs=None,
            use_cpu=False, force_redo = False
            ):
        """
        Returns betas for all channels and layers.,

        Args:
            session (int): session ID
        """
        session = str(int(session))
        if session not in self.B.keys() or force_redo:
            _, B, *_ = self.cross_validated_regression(
                session=session,  bin_width=bin_width, delay=delay,
                num_lmbdas=8, iterations=1, numpy=use_cpu,
                layer_IDs=layer_IDs,
            )
            self.B[session] = B
        return self.B[session]
    

    @staticmethod
    def inter_trial_corr(spikes, n=1000):
        """Compute distribution of inter-trials correlations.

        Args: 
            spikes (ndarray): (repeats, samples/time, channels)

        Returns:
            trials_corr (ndarray): (n, channels) distribution of inter-trial correlations
        """
        trials_corr = np.zeros((n, spikes.shape[2]))
        for t in range(n):
            trials = np.random.choice(np.arange(0,spikes.shape[0]), size=2, replace=False)
            trials_corr[t] = utils.cc_norm(spikes[trials[0]].squeeze(), spikes[trials[1]].squeeze())

        return trials_corr

    #####################################################################################
    ##############      Methods using contextualized features and spikes    #############
    #################################################################################

    def get_contextualized_features(self, layer_IDs, bin_width=20):
        """"Get contextualized features and resample them to get match total
        duration (approx) used here.."""
        

        # # extracting spikes...
        long_audio, total_duration, ordered_sent_IDs, duration_before_sent  = self.dataloader.get_contextualized_stim_audio()
        # total_duration = 1171.5 # seconds
        # total_duration = long_audio.size/16000
        fs = 1000/bin_width
        bin_width /= 1000 # sec
        n = int(np.ceil(round(total_duration/bin_width, 3)))
        # n = int(np.floor((total_duration*1000)/bin_width))

        # features = self.dataloader.get_DNN_obj(self.model_name).extract_features_for_audio(long_audio, total_duration)
        features = self.dataloader.get_raw_DNN_features(self.model_name, contextualized=True)
        unrolled_features = []
        for layer_id in layer_IDs:
            unrolled_features.append(
                signal.resample(features[layer_id], n, axis=0)
                )
        all_layer_features = np.stack(unrolled_features, axis=0)

        sent_wise_features = {}
        for sent, earlier_duration in duration_before_sent.items():
            sent_duration = self.dataloader.metadata.stim_duration(sent)
            
            earlier_samples = int(np.ceil(round(earlier_duration*fs, 3)))
            sent_samples = int(np.ceil(round(sent_duration*fs,3)))
            sent_wise_features[sent] = all_layer_features[:,earlier_samples:earlier_samples+sent_samples ,:]

        all_stacked_features = np.concatenate([features[:,:-1,:] for features in sent_wise_features.values()], axis=1)
        return cp.array(all_stacked_features)

        # feature_segments = []
        # dead_interval = 0.3
        
        # dead_samples = int(np.ceil(round(dead_interval/bin_width, 3)))
        # # int(np.floor((dead_interval*1000)/bin_width))
        # previous_pointer = dead_samples
        # for sent in ordered_sent_IDs[:498]:
        #     dur = self.dataloader.metadata.stim_duration(sent)
        #     n_sents = int(np.ceil(round(dur/bin_width, 3)))
        #     # int(np.floor((dur*1000)/bin_width))

        #     feature_segments.append(all_layer_features[:,previous_pointer:previous_pointer+n_sents-1,:])    # dropping last sample
        #     previous_pointer += n_sents + dead_samples

        # total = 0
        # for sent in ordered_sent_IDs:
        #     total += obj.dataloader.metadata.stim_duration(sent)
        # bin_width = 20
        # bin_width /= 1000
        # int(np.ceil(round(total/bin_width, 3)))-498
        # gives 50507
        # n = 50507
        # features = signal.resample(np.concatenate(feature_segments, axis=1), n, axis=1)
        # return cp.array(features)
            

        
        # return cp.asarray(np.stack(unrolled_features, axis=0))
    
    def get_contextualized_spikes(self, session, bin_width=20, delay=0):
        """"Get contextualized spikes and resample them to get match total
        duration (approx) used here.."""
        # total_duration = 1171.5 # seconds
        # n = int(np.floor((total_duration*1000)/bin_width))

        # spikes, total_duration = self.dataloader.get_dataset_object(session).retrieve_contextualized_spikes(
        #     bin_width, delay
        # )
        total_num_stimuli = self.dataloader.metadata.sent_IDs.size 
        session = str(session)
        raw_spikes = self.dataloader.get_session_spikes(session=session, bin_width=bin_width, delay=delay)
        ordered_sent_IDs = self.dataloader.get_dataset_object(session).ordered_sent_IDs[:total_num_stimuli]

        spikes = np.concatenate([raw_spikes[sent][:-1] for sent in ordered_sent_IDs], axis=0)
        # n = 50507
        # spikes = signal.resample(spikes, n, axis=0)
        
        return cp.array(spikes)

        # return cp.asarray(signal.resample(spikes, n, axis=0))


    def grid_search_CV_contextualized(self,
            session,
            bin_width=20,
            delays=None,
            layer_IDs=None,
            num_lmbdas=10,
            N_sents=500,
            iterations=1,
            num_folds=5, 
            sents=None,
            numpy=False,
            return_dict=False,
            third=None
        ):
        
        if delays is None:
            delays = [0, 10, 20, 30] 
            
        num_channels = self.dataloader.get_num_channels(session)
        if layer_IDs is None:
            layer_IDs = self.get_layer_IDs()   

        corr_coeffs = []
        losses = []
        session = str(session)

        ##############
        context_features = self.get_contextualized_features(layer_IDs, bin_width)


        for i, delay in enumerate(delays):

            # force reload spikes at the desired delay...
            print(f"Loading neural spikes with delay: {delay}ms")
            # _ = self.get_neural_spikes(
            #         session, bin_width=bin_width, delay=delay, force_reload=True
            #     )
            corr_coeff, _, loss, _ = self.cross_validated_regression_contextualized(
                context_features=context_features, session=session, 
                bin_width=bin_width, delay=delay,layer_IDs=layer_IDs,
                num_folds=num_folds, iterations=iterations, num_lmbdas=num_lmbdas,
                N_sents=N_sents, return_dict=False, numpy=numpy,
                )
            corr_coeffs.append(corr_coeff)
            losses.append(loss)
        
        corr_coeffs = np.array(corr_coeffs)
        losses = np.array(losses)
        delays = np.array(delays)
        # num_channels = self.spike_datasets[session].num_channels

        opt_delay_indices = np.argmin(losses, axis=0)
        opt_delays = delays[opt_delay_indices]
        corr_coeffs_opt_delay = corr_coeffs[
                opt_delay_indices, np.arange(len(layer_IDs))[:, None], np.arange(num_channels)
            ]
 
        if return_dict:
            # deallocate the memory of Neural data for current session, this will save memory used.
            # del self.spike_datasets[session]
            # saving results in a dictionary..
            corr_dict = {
                    'test_cc_raw': corr_coeffs_opt_delay,
                    'train_cc_raw': np.zeros_like(corr_coeffs_opt_delay),
                    'win': bin_width,
                    'delay': 0, 
                    'session': session,
                    'model': self.model_name,
                    'N_sents': N_sents,
                    'layer_ids': layer_IDs,
                    'opt_delays': opt_delays
                    }
            return corr_dict
        # optimal_delays = delays[np.argmin(losses, axis=0)]
        return corr_coeffs_opt_delay, opt_delays
    
    
    def cross_validated_regression_contextualized(
            self, context_features, session, bin_width=20, delay=0,
            num_folds=5, num_lmbdas=20,
            iterations=10, N_sents=500, return_dict=False, numpy=False,
            layer_IDs=None
        ):
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

            third (int) [1,2,3]:    Default: None, section of test sents to compute corr for.  

        Returns:
            corr_coeff (3d-array):  distribution of correlations for all layers and channels (if return_dict=False)
            corr (dict):  median(corr_coeff) stored in dict, along with other details, ready to save (if return_dict=True)
        """
        if numpy:
            module = np
        else:
            module = cp

        session = str(session)
        print(f"Loading spikes for session: {session}")
        context_spikes = self.get_contextualized_spikes(session, bin_width, delay)

        num_layers, num_samples, feature_dims = context_features.shape
        num_channels = context_spikes.shape[-1]
       
        # feature_dims = self.sampled_features[0].shape[1]
        lmbdas = module.logspace(start=-4, stop=-1, num=num_lmbdas)
        B = module.zeros((num_layers, feature_dims, num_channels))
        corr_coeff = np.zeros((iterations, num_channels, num_layers))
        corr_coeff_train = np.zeros((iterations, num_channels, num_layers))
        # stimuli = np.array(list(self.raw_features[0].keys()))

        sample_set = np.arange(num_samples)
        
        test_set = sample_set[int(0.9*num_samples):]
        mapping_set = sample_set[:int(0.9*num_samples)]

        # size_of_chunk = int(mapping_sents/k)
        print(f"# of iterations requested: {iterations}, \n \
                # of lambda samples per iteration: {len(lmbdas)}")
        time_itr = 0
        time_lmbda = 0
        time_map = 0
        # time_fold = 0
        for n in range(iterations): 
            print(f"Itr: {n+1}:")
            start_itr = time.time()

            # lmbda_loss = module.zeros(((len(lmbdas), num_channels, self.num_layers)))
            start_lmbda = time.time()

            mapping_x = context_features[:,mapping_set, :]
            mapping_y = context_spikes[mapping_set, :]
            lmbda_loss = self.k_fold_CV_contextualized(
                features=mapping_x,
                spikes=mapping_y,
                lmbdas=lmbdas,
                num_folds=num_folds
            )
            
            end_lmbda = time.time()
            time_lmbda += end_lmbda-start_lmbda
            optimal_lmbdas = lmbdas[np.argmin(lmbda_loss, axis=0)]
            start_map = time.time()

            # mapping_y = self.get_neural_spikes(session, bin_width=bin_width, sents=mapping_set, numpy=numpy)
            #computing betas
            for l in range(num_layers):
                for ch in range(num_channels):
                    B[l,:,ch] = utils.reg(mapping_x[l,:,:], mapping_y[:,ch], optimal_lmbdas[ch,l])
            # self.B[session] = cp.asnumpy(self.B[session])

            train_pred = utils.predict(mapping_x, B)
            corr_coeff_train[n] = utils.cc_norm(mapping_y, train_pred)

            del mapping_x
            del mapping_y
            gc.collect()

            test_x = context_features[:,test_set, :]
            test_y = context_spikes[test_set, :]

            test_pred = utils.predict(test_x, B)
            corr_coeff[n] = utils.cc_norm(test_y, test_pred)
            end_map = time.time()
            end_itr = time.time()
            time_map += end_map - start_map
            time_itr += (end_itr - start_itr)
        
        #         print(f"itr-{n}: It takes {(end_itr - start_itr):.2f} seconds for all lambdas")
        # print(f"It takes (on avg.) {time_fold/(k*N*len(lmbdas)):.2f} sec for each step of cross validation (1 fold)")
        print(f"It takes (on avg.) {time_lmbda/(iterations):.2f} sec (all lmbdas). (time for {num_folds}-folds)")
        print(f"It takes (on avg.) {time_map/(iterations):.2f} sec/mapping.")
        print(f"It takes (on avg.) {time_itr/(iterations*60):.2f} minutes/iteration...!")
        corr_coeff = cp.asnumpy(corr_coeff.transpose((0,2,1)))
        corr_coeff = np.median(corr_coeff, axis=0)
        corr_coeff_train = cp.asnumpy(corr_coeff_train.transpose((0,2,1)))
        lmbda_loss = lmbda_loss.transpose((0,2,1))
        if return_dict:
            # deallocate the memory of Neural data for current session, this will save memory used.
            # del self.spike_datasets[session]
            # saving results in a dictionary..
            corr_coeff_train = np.median(corr_coeff_train, axis=0)
            corr = {'test_cc_raw': corr_coeff,
                    'train_cc_raw': corr_coeff_train,
                    'win': bin_width,
                    'delay': delay, 
                    'session': session,
                    'model': self.model_name,
                    'N_sents': N_sents,
                    'layer_ids': layer_IDs,
                    'opt_delays': None
                    }
            return corr
        return corr_coeff, B, np.min(lmbda_loss, axis=0), test_set


    def k_fold_CV_contextualized(
            self, features, spikes, lmbdas, num_folds=5
        ):
        """Return MSE loss (avg.) for k-fold CV regression.

        Args:
            session (str): session ID 
            mapping_set (list): sent ID to be used for CV
            lmbdas (float): Range of regularization paramters to compare
            num_folds (int): # of folds
            use_cpu (bool): default 'False', use numpy or cupy? 

        Returns:
            avg. MSE loss for validation set.     
        """
        num_layers, num_samples, _ = features.shape
        num_channels = spikes.shape[-1]
        lmbda_loss = np.zeros(((len(lmbdas), num_channels, num_layers)))

        size_of_chunk = int(num_samples / num_folds)
        mapping_set = np.arange(num_samples)
        # for i, lmbda in enumerate(lmbdas):
        #     loss = 0
        for r in range(num_folds):
            print(f"For fold={r}: ")
            # get the sent ids for train and validation folds...
            if r<(num_folds-1):
                val_set = mapping_set[r*size_of_chunk:(r+1)*size_of_chunk]
            else:
                val_set = mapping_set[r*size_of_chunk:]
            train_set = mapping_set[np.isin(mapping_set, val_set, invert=True)]

            train_x = features[:,train_set,:]
            train_y = spikes[train_set,:]

            # computing Betas for lmbdas and removing train_x, train_y to manage memory..
            Betas = {}
            for i, lmbda in enumerate(lmbdas):
                Betas[i] = utils.reg(train_x, train_y, lmbda)
            
            del train_x
            del train_y
            gc.collect()

            # train_y = self.get_neural_spikes(session, bin_width=bin_width, sents=train_set, numpy=use_cpu)

            val_x = features[:,val_set,:]
            val_y = spikes[val_set,:]

            # for the current fold, compute/save validation loss for each lambda.
            for i, lmbda in enumerate(lmbdas):

                # Beta = utils.reg(train_x, train_y, lmbda)
                # val_pred = utils.predict(val_x, Beta)
                val_pred = utils.predict(val_x, Betas[i])

                loss = utils.mse_loss(val_y, val_pred)
                lmbda_loss[i] += cp.asnumpy((loss))

            # de-allocate train_x and train_y to reduce memroy utilization...
            del val_x
            del val_y
            gc.collect()
            # de-allocation of memory ends here...

        lmbda_loss /= num_folds            
        return lmbda_loss
    


    def compute_context_dependent_normalizer_variance(
            self, layer_IDs, bin_width=20, n_iterations=10000
        ):
        """Computes context dependent variance of the normalizer,
        using features from layers provided (for the self.model_name)
        
        Args:
            layer_IDs: list = list of layer IDs
            bin_width: float = bin_width in ms.
            n_iterations: float = number of pairwise comparisons.

        """
        long_audio, total_duration, ordered_sent_IDs, duration_before_sent  = self.dataloader.get_contextualized_stim_audio(
            include_repeated_trials=True
            )

        fs = 1000/bin_width
        bin_width /= 1000 # sec
        n = int(np.ceil(round(total_duration/bin_width, 3)))

        features = self.dataloader.get_raw_DNN_features(self.model_name, contextualized=True)
        unrolled_features = []
        for layer_id in layer_IDs:
            unrolled_features.append(
                signal.resample(features[layer_id], n, axis=0)
                )
        all_layer_features = np.stack(unrolled_features, axis=0)

        ### picking out features for repeated sents only...
        all_repeated_trials = {tr:[] for tr in range(11)}
        for sent, times in duration_before_sent.items():
            if len(times) > 1:  # picks repeated sents only..
                for trial, earlier_duration in enumerate(times):
                    sent_duration = self.dataloader.metadata.stim_duration(sent)
            
                    earlier_samples = int(np.ceil(round(earlier_duration*fs, 3)))
                    sent_samples = int(np.ceil(round(sent_duration*fs,3)))
                    features_slice = all_layer_features[:,earlier_samples:earlier_samples+sent_samples ,:]
                    all_repeated_trials[trial].append(features_slice)

        all_features_combined = []
        for trial, sequences in all_repeated_trials.items():  
            all_features_combined.append(np.concatenate(sequences, axis=1))
        all_features_combined = np.stack(all_features_combined, axis=1)

        # computing corr (avg. across neurons of each layer)
        layer_wise_corr = {}
        for l, layer_ID in enumerate(layer_IDs):
            corr_all_channels = self.inter_trial_corr(all_features_combined[l], n=n_iterations)
            layer_wise_corr[layer_ID] = np.mean(corr_all_channels)

        return layer_wise_corr









        



    ### Useful methods (Used in the past)


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


    # def get_betas(self, session, use_cpu=False):
    #     """
    #     Returns betas for all channels of the layer,

    #     Args:
    #         layer (int): index of the layer
    #     """
    #     features = self.unroll_features(numpy=use_cpu)
    #     spikes = self.get_neural_spikes(session, numpy=use_cpu)
    #     print("loading features and spikes for beta calculation in regression class...")
    #     # print(type(features))
    #     # print(type(spikes))
    #     B = utils.reg(features, spikes)
    #     return B
    
    def neural_prediction(
            self, session, bin_width: int, delay: int,
            sents: list, layer_IDs: list=None,
            force_reload: bool=False, shuffled: bool=False,
            force_redo: bool=True,
        ):
        """
        Returns prediction for neural activity 

        Args:
            sent (list int): index of sentence ID 

        Returns:
            ndarray : (time, ch, layers) Prdicted neural activity 
        """
        features = self.unroll_features(
            stim_ids=sents,
            bin_width=bin_width, delay=delay, 
            layer_IDs=layer_IDs, force_reload=force_reload, shuffled=shuffled
            )
        # features = np.stack([dict_feats[i] for i in range(12)], axis=0)
        beta = self.get_betas(
            session, bin_width=bin_width, delay=delay,
            layer_IDs=layer_IDs, force_redo=force_redo)
        beta = cp.asnumpy(beta)
        return utils.predict(features, beta)

    #########################################    ##############################


    ################################################################################################
    ##############Redundant functions....!
    #############################################################


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
