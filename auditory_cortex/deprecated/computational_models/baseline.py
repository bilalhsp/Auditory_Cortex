import math
import numpy as np
from scipy.signal import resample

import naplib as nl
from auditory_cortex import config
# from auditory_cortex.neural_data import dataset
from auditory_cortex.deprecated.dataloader import DataLoader
from auditory_cortex import utils
from sklearn.linear_model import RidgeCV, ElasticNet, Ridge, PoissonRegressor
from transformers import Speech2TextProcessor


class STRF:
    def __init__(self, num_freqs=80, mel_spectrogram=False):
        """
        Args:
            num_freqs (int): Number of frequency channels on spectrogram
        """       
        self.model_name = 'strf_model'
        data_dir = config['neural_data_dir']
        # self.dataset = NeuralData(data_dir, session)
        # self.dataset.extract_spikes(bin_width=self.bin_width, delay=0)
        self.mel_spectrogram = mel_spectrogram
        if self.mel_spectrogram:
            print(F"Using mel-spectrogram for STRF.")
        else:
            print(F"Using wavelet-spectrogram for STRF.")

        self.processor = Speech2TextProcessor.from_pretrained("facebook/s2t-large-librispeech-asr")
        # self.fs = self.dataset.fs
        self.dataloader = DataLoader()
        self.num_freqs = num_freqs # num_freqs in the spectrogram
        self.session_bw_cache = {}

    def get_spectrogram(self, aud, sampling_rate):
        """Transforms the given audio into the spectrogram"""
        # Getting the spectrogram at 10 ms and then resample to match the bin_width
        if self.mel_spectrogram:
            n_new = int(aud.size*16000/sampling_rate)
            aud = resample(aud, n_new)
            spect = self.processor(aud, padding=True, sampling_rate=16000).input_features[0]
        else:
            spect = nl.features.auditory_spectrogram(aud, sampling_rate, frame_len=10)
        return spect
    def get_training_stim_ids(self, session=None, mVocs=False):
        """Returns the stim ids for training set.
        
        Args:
            session: int = session ID, needed ONLY if mVocs=True.
            mVocs: bool = If True, returns ids for mVocs,
                otherwise for timit stimuli.
        """
        if mVocs:
            stim_ids = self.dataloader.metadata.mVocTrialIds
            # exclude the missing trial IDs from list of Ids
            missing_trial_ids = self.dataloader.get_dataset_object(session=session).missing_trial_ids
            stim_ids = stim_ids[np.isin(stim_ids, missing_trial_ids, invert=True)]
            test_ids = self.dataloader.metadata.mVoc_test_trIds
            training_stim_ids = stim_ids[np.isin(stim_ids, test_ids, invert=True)]
        else:
            sent_IDs = self.dataloader.sent_IDs
            testing_sent_ids = self.dataloader.test_sent_IDs
            training_stim_ids = sent_IDs[np.isin(sent_IDs, testing_sent_ids, invert=True)]
        return training_stim_ids
    
    def get_testing_stim_ids(self, mVocs=False):
        """Returns the stim ids for testing set.
        
        Args:
            session: int = session ID, needed ONLY if mVocs=True.
            mVocs: bool = If True, returns ids for mVocs,
                otherwise for timit stimuli.
        """
        if mVocs:
            testing_stim_ids = self.dataloader.metadata.mVoc_test_stimIds
        else:
            testing_stim_ids = self.dataloader.test_sent_IDs
        return testing_stim_ids

    def load_sent_wise_data(self, session, bin_width, mVocs=False):
        """Reads data for all the sents, and caches the spectrogram and spike pairs,
        saves using session-bin_width key.
        """
        print(f"Loading data for session-{session} at bin_width-{bin_width}ms.")
        session = str(session)
        raw_spikes = self.dataloader.get_session_spikes(
            session=session,
            bin_width=bin_width,
            delay=0,
            mVocs=mVocs
            )
        
        training_stim_ids = self.get_training_stim_ids(session=session, mVocs=mVocs)
        sampling_rate = self.dataloader.metadata.get_sampling_rate(mVocs=mVocs)
        stim_wise_spects = {}
        stim_wise_spikes = {} 
        for stim in training_stim_ids:
            aud = self.dataloader.get_stim_aud(stim, mVocs=mVocs)
            spect = self.get_spectrogram(aud, sampling_rate)
            # Getting the spectrogram at 10 ms and then resample to match the bin_width
            # spect = nl.features.auditory_spectrogram(aud, sampling_rate, frame_len=10)
            
            spikes = raw_spikes[stim]
            num_bins = spikes.shape[0]
            spect = resample(spect, num_bins, axis=0)
            spect = resample(spect, self.num_freqs, axis=1)

            stim_wise_spects[stim] = spect
            stim_wise_spikes[stim] = spikes

        cache_data = {
            'spects': stim_wise_spects,
            'spikes': stim_wise_spikes,
        }

        cache_key = str(int(session))+'_'+str(int(bin_width))
        if mVocs:
            cache_key = 'mVocs_'+cache_key
        self.session_bw_cache[cache_key] = cache_data
        print(f"Done.")

    def get_data(self, session, bin_width, stim_ids=None, mVocs=False):
        """Returns spectral-features, spikes pair for the 
        given session ID and sent IDs. If test=True,
        returns repeated trials data.
        """
        cache_key = str(int(session))+'_'+str(int(bin_width))
        if mVocs:
            cache_key = 'mVocs_'+cache_key
        # print(f"key: {cache_key}")
        
        if cache_key not in self.session_bw_cache.keys():
            print(f"loading again..")
            self.load_sent_wise_data(session, bin_width, mVocs=mVocs)
        # else:
        #     print(self.session_bw_cache.keys())

        if stim_ids is None:
            stim_ids = self.get_training_stim_ids(session, mVocs)
        
        spects = self.session_bw_cache[cache_key]['spects']
        spikes = self.session_bw_cache[cache_key]['spikes']
        spikes_list = []
        spects_list = []
        for stim in stim_ids:
            spects_list.append(spects[stim])
            spikes_list.append(spikes[stim])
        
        return spects_list, spikes_list




        # session = str(session)
        # raw_spikes = self.dataloader.get_session_spikes(
        #     session=session,
        #     bin_width=bin_width,
        #     delay=0
        #     )
        # spikes_list = []
        # spect_list = []
        # for sent in sent_IDs:
        #     aud = self.dataloader.metadata.stim_audio(sent)
        #     # Getting the spectrogram at 10 ms and then resample to match the bin_width
        #     spect = nl.features.auditory_spectrogram(aud, self.fs, frame_len=10)
            
        #     spikes = raw_spikes[sent]
        #     num_bins = spikes.shape[0]
        #     spect = resample(spect, num_bins, axis=0)
        #     spect = resample(spect, self.num_freqs, axis=1)
        #     spikes_list.append(spikes)
        #     spect_list.append(spect)

        # return spect_list, spikes_list
    
    def get_test_data(self, session, bin_width, mVocs = False):
        """Returns spectral-features, spikes (all trials)
        for the test sent IDs, given session.

        Args:
            session: int = session ID
            bin_width: int = bin width in ms
            
        Returns:
            spect_list: list = each entry of list is a spect 
                for a sent audio.
            repeated_spikes_list: list of lists = 11 lists 
                corresponding to 11 trials and each one having
                entries equal to number of sentences in test set.

        """
        session = str(session)
        spect_list = []
        # mVocs = True
        # repeated_spikes_list = {i: [] for i in range(11)}
        all_sent_spikes = []
        testing_stim_ids = self.get_testing_stim_ids(mVocs=mVocs)
        sampling_rate = self.dataloader.metadata.get_sampling_rate(mVocs=mVocs)

        for stim in testing_stim_ids:
            if mVocs:
                tr_id = self.dataloader.metadata.get_mVoc_tr_id(stim)[0]
                aud = self.dataloader.get_stim_aud(tr_id, mVocs=mVocs)
            else:
                aud = self.dataloader.get_stim_aud(stim, mVocs=mVocs)
            # Getting the spectrogram at 10 ms and then resample to match the bin_width
            # spect = nl.features.auditory_spectrogram(aud, sampling_rate, frame_len=10)
            spect = self.get_spectrogram(aud, sampling_rate)

            repeated_spikes = self.dataloader.get_neural_data_for_repeated_trials(
                session=session,
                bin_width=bin_width,
                delay=0,
                stim_ids=[stim],
                mVocs=mVocs,
                )

            num_bins = repeated_spikes.shape[1]
            spect = resample(spect, num_bins, axis=0)
            spect = resample(spect, self.num_freqs, axis=1)
            spect_list.append(spect)
            all_sent_spikes.append(repeated_spikes)
            # for i in range(11):
            #     repeated_spikes_list[i].append(repeated_spikes[i])
        all_sent_spikes = np.concatenate(all_sent_spikes, axis=1)
        return spect_list, all_sent_spikes

    # Deprecated
    # def get_test_data_timit(self, session, bin_width):
    #     """Returns spectral-features, spikes (all trials)
    #     for the test sent IDs, given session.

    #     Args:
    #         session: int = session ID
    #         bin_width: int = bin width in ms
            
    #     Returns:
    #         spect_list: list = each entry of list is a spect 
    #             for a sent audio.
    #         repeated_spikes_list: list of lists = 11 lists 
    #             corresponding to 11 trials and each one having
    #             entries equal to number of sentences in test set.

    #     """
    #     session = str(session)
    #     spect_list = []
    #     # repeated_spikes_list = {i: [] for i in range(11)}
    #     all_sent_spikes = []

    #     for sent in self.testing_sent_ids:
    #         aud = self.dataloader.metadata.stim_audio(sent)
    #         # Getting the spectrogram at 10 ms and then resample to match the bin_width
    #         spect = nl.features.auditory_spectrogram(aud, self.fs, frame_len=10)

    #         repeated_spikes = self.dataloader.get_neural_data_for_repeated_trials(
    #             session=session,
    #             bin_width=bin_width,
    #             delay=0,
    #             sent_IDs=[sent]
    #             )

    #         num_bins = repeated_spikes.shape[1]
    #         spect = resample(spect, num_bins, axis=0)
    #         spect = resample(spect, self.num_freqs, axis=1)
    #         spect_list.append(spect)
    #         all_sent_spikes.append(repeated_spikes)
    #         # for i in range(11):
    #         #     repeated_spikes_list[i].append(repeated_spikes[i])
    #     all_sent_spikes = np.concatenate(all_sent_spikes, axis=1)
    #     return spect_list, all_sent_spikes


    def evaluate(
            self, strf_model, session, bin_width,
            test_trial=None, mVocs=False
        ):
        """Computes correlation on trials of test set for the model provided.
        
        Args:
            strf_model: naplib model = trained model
            session: int = session ID
            bin_width: int = bin width in ms
            test_trial: int = trial ID to be tested on. Default=None, 
                in which case, it tests on all the trials [0--10]
                and returns averaged correlations. 
        Return:
            ndarray: (num_channels,)   
        """

        test_spect_list, all_test_spikes = self.get_test_data(
            session, bin_width, mVocs=mVocs)
        predicted_response = strf_model.predict(X=test_spect_list)
        predicted_response = np.concatenate(predicted_response, axis=0)

        corr = utils.compute_avg_test_corr(
            all_test_spikes, predicted_response, test_trial, mVocs=mVocs)
        return corr

    def cross_validted_fit(
            self, session, bin_width, tmax = 50,
            tmin=0, num_workers=1,
            num_lmbdas=8, num_folds=3,
            use_nonlinearity = False,
            mVocs=False,
        ):
        """Computes score for the given lag (tmax) using cross-validated fit.
        
        Args:
            session: int = session ID
            bin_width: int = bin width in ms
            tmax: int = lag (window width) in ms
            tmin: int = min lag start of window in ms
            num_workers: int = number of workers used by naplib.
            num_lmbdas: int  = number of regularization parameters (lmbdas)
            num_folds: int = number of folds of cross-validation
            use_nonlinearity: bool = using non-linearity with the linear model or not.
                Default = False.
            mVocs: bool = if True, uses mVocs as set of stimuli.
        
        """
        
        tmin = tmin/1000
        tmax = tmax/1000
        sfreq = 1000/bin_width
        # lmbdas = np.logspace(-2, 5, num_lmbdas)
        lmbdas = np.logspace(-2, 10, 13)
        session = str(int(session))
        # load session spikes, if not already done so far..
        cache_key = session+'_'+str(int(bin_width))
        if mVocs:
            cache_key = 'mVocs_'+cache_key
        if cache_key not in self.session_bw_cache.keys():
            self.load_sent_wise_data(session, bin_width, mVocs=mVocs)

        num_channels = self.dataloader.get_num_channels(session)
        lmbda_score = np.zeros(((len(lmbdas), num_channels)))
        # val_score = np.zeros(num_channels)
        # mapping_set = self.training_sent_ids
        mapping_set = self.get_training_stim_ids(session, mVocs)
        np.random.shuffle(mapping_set)
        size_of_chunk = int(len(mapping_set) / num_folds)

        for r in range(num_folds):
            print(f"\n For fold={r}: ")
            if r<(num_folds-1):
                val_set = mapping_set[r*size_of_chunk:(r+1)*size_of_chunk]
            else:
                val_set = mapping_set[r*size_of_chunk:]
            train_set = mapping_set[np.isin(mapping_set, val_set, invert=True)]

            train_x, train_y = self.get_data(
                session, bin_width, stim_ids=train_set, mVocs=mVocs
            )

            val_x, val_y = self.get_data(
                session, bin_width, stim_ids=val_set, mVocs=mVocs
            )
        # Deprecated
        #     # changing Ridge to RidgeCV
        #     alphas = np.logspace(-2, 5, 6)
        #     estimator = RidgeCV(alphas=alphas, cv=5)
        #     strf = nl.encoding.TRF(
        #                 tmin, tmax, sfreq, estimator=estimator,
        #                 n_jobs=num_workers, show_progress=True
        #                 )
        #     strf.fit(X=train_x, y=train_y)
        #     val_score += strf.corr(X=val_x, y=val_y)

        # val_score /= num_folds
        # return np.mean(val_score)
    
        #changing to Ridge
            for i, lmbda in enumerate(lmbdas):

                if use_nonlinearity:
                    estimator = PoissonRegressor(alpha=lmbda)
                else:
                    estimator = Ridge(alpha=lmbda)
            
                strf = nl.encoding.TRF(
                        tmin, tmax, sfreq, estimator=estimator,
                        n_jobs=num_workers, show_progress=True
                        )
                strf.fit(X=train_x, y=train_y)

                # save validation score for lmbda..
                lmbda_score[i] += strf.score(X=val_x, y=val_y)

        lmbda_score /= num_folds

        avg_lmbda_score = np.mean(lmbda_score, axis=1)
        max_lmbda_score = np.max(avg_lmbda_score)
        opt_lmbda = lmbdas[np.argmax(avg_lmbda_score)]
        return max_lmbda_score, opt_lmbda



    def grid_search_CV(
            self,
            session: int,
            bin_width: int,  
            lags: list = None,      
            tmin = 0,
            num_workers=1, 
            num_lmbdas = 8, 
            num_folds = 3, 
            use_nonlinearity = False, 
            test_trial=None,
            mVocs=False,
        ):
        """Fits the linear model (with or without non-linearity) 
        by searching for optimal lag (max window lag) using cross-
        validation.

        Args:
            session: int = session ID
            bin_width: int = bin width in ms
            lags: list = lags (window width) in ms
            tmin: int = min lag start of window in ms
            num_workers: int = number of workers used by naplib.
            num_lmbdas: int  = number of regularization parameters (lmbdas)
            num_folds: int = number of folds of cross-validation
            use_nonlinearity: bool = using non-linearity with the linear model or not.
                Default = False.
            test_trial: int = trial ID to be tested on. Default=None, 
                in which case, it tests on all the trials [0--10]
                and returns averaged correlations. 
            mVocs: bool = if True, uses mVocs as set of stimuli.
        Return:
            ndarray: (num_channels,) 
            optimal_lag: int 

        """



        if lags is None:
            lags = [5, 10, 20, 40, 80, 160, 320]

        lag_scores = []
        opt_lmbdas = []
        for lag in lags:

            print(f"\n Running for max lag={lag} ms")
            score, lmbda = self.cross_validted_fit(
                session, bin_width,
                tmin=tmin, tmax=lag, 
                num_workers=num_workers,
                num_lmbdas=num_lmbdas, num_folds=num_folds,
                use_nonlinearity=use_nonlinearity,
                mVocs=mVocs
                )
            lag_scores.append(score)
            opt_lmbdas.append(lmbda)

        max_score_ind = np.argmax(lag_scores)
        opt_lag = lags[max_score_ind]

        #### for using RidgeCV ####
        # print(f"\n Opt lag for session-{session} = {opt_lag} ms")
        # sfreq = 1000/bin_width
        # tmax = opt_lag/1000 # convert seconds to ms
        # alphas = np.logspace(-2, 5, 6)
        # estimator = RidgeCV(alphas=alphas, cv=5)
        # strf = nl.encoding.TRF(
        #             tmin, tmax, sfreq, estimator=estimator,
        #             n_jobs=num_workers, show_progress=True
        #             )
        # # get mapping data..
        # mapping_x, mapping_y = self.get_data(
        #                 session, bin_width
        #             )
        # strf.fit(X=mapping_x, y=mapping_y)
        # corr = self.evaluate(strf, session, bin_width, test_trial)

        # return corr, opt_lag, 0.01


        ### for using RidgeCV ####
        opt_lmbda = opt_lmbdas[max_score_ind]

        # get mapping data..
        mapping_x, mapping_y = self.get_data(
                        session, bin_width, mVocs=mVocs
                    )

        # fit strf using opt lag and lmbda
        if use_nonlinearity:
            estimator = PoissonRegressor(alpha=opt_lmbda)
        else:
            estimator = Ridge(alpha=opt_lmbda)
        sfreq = 1000/bin_width
        tmax = opt_lag/1000 # convert seconds to ms
        strf = nl.encoding.TRF(
                tmin, tmax, sfreq, estimator=estimator,
                n_jobs=num_workers, show_progress=True
                )
        strf.fit(X=mapping_x, y=mapping_y)

        corr = self.evaluate(strf, session, bin_width, test_trial, mVocs=mVocs)

        return corr, opt_lag, opt_lmbda

# # Old implementations

# class STRF:
#     def __init__(self, session, estimator=None, num_workers=1, num_freqs=80, 
#             tmin=0, tmax = 0.3, bin_width=20, train_dataset_size=0.8):
#         """
#         Args:
#             num_freqs (int): Number of frequency channels on spectrogram
#             tmin: receptive field begins at (ms)
#             tmax: receptive field ends at (ms)
#             sfreq: sampling frequency of data (Hz)

#             train_dataset_size: [0.0, 1.0]= fraction of total dataset 
#             to be used as training/val split. Defalt=0.8
#                 (excluding the test split only)
#         """
#         self.model_name = 'strf_model'
#         session = str(int(session))
#         self.bin_width = bin_width
#         data_dir = config['neural_data_dir']
#         # self.dataset = NeuralData(data_dir, session)
#         # self.dataset.extract_spikes(bin_width=self.bin_width, delay=0)
#         # self.fs = self.dataset.fs
#         self.dataloader = DataLoader()
#         self.session_spikes = self.dataloader.get_session_spikes(
#             session=session, bin_width=bin_width, delay=0
#             )
#         self.fs = self.dataloader.metadata.get_sampling_rate()
#         self.num_freqs = num_freqs # num_freqs in the spectrogram

#         if estimator is None:
#             alphas = np.logspace(-2, 5, 6)
#             estimator = RidgeCV(alphas=alphas, cv=5)


#         # creating a STRF model...
#         sfreq = 1000/bin_width # 50 (since bin_width is in ms)
#         self.strf_model = nl.encoding.TRF(
#             tmin, tmax, sfreq, estimator=estimator,
#             n_jobs=num_workers, show_progress=True
#             )

#         # sents = np.arange(1,499)
#         # self.random_sent_ids = np.random.permutation(sents)
#         # self.size_training_dataset = int(train_dataset_size*(sents.size))
#         sent_IDs = self.dataloader.sent_IDs
#         self.testing_sent_ids = self.dataloader.test_sent_IDs
#         self.training_sent_ids = sent_IDs[np.isin(sent_IDs, self.testing_sent_ids, invert=True)]

#     def get_sample(self, sent, third=None):

#         # spikes = self.dataset.unroll_spikes([sent], third=third).astype(np.float32)
#         # aud = self.dataset.audio(sent)
    
#         spikes = self.session_spikes[sent]
#         num_bins = spikes[0]
#         aud = self.dataloader.metadata.stim_audio(sent)
#         # Getting the spectrogram at 10 ms and then resample to match the bin_width
#         spect = nl.features.auditory_spectrogram(aud, self.fs, frame_len=10)

#         if third is not None:
#             # bin_width = self.bin_width/1000.0
#             # n = int(np.ceil(round(self.dataset.duration(sent)/bin_width, 3)))
            
#             # # store boundaries of sent thirds...
#             # one_third = int(n/3)
#             # two_third = int(2*n/3)
#             # sent_sections = [0, one_third, two_third, n]

#             # spect = resample(spect, n, axis=0)
#             # # print(spect.shape)
#             # spect = spect[sent_sections[third-1]: sent_sections[third]]
#             # print(spect.shape)
#             ...
            
#         else:
#             spect = resample(spect, num_bins, axis=0)
#         # read spikes for the sent id, as (time, channel)
#         # spikes = self.spikes[sent].astype(np.float32)

#         # get spectrogram for the audio inputs, as (time, freq) 
#         # spect = nl.features.auditory_spectrogram(aud, self.fs)
#         # resample spect to get same # number of time samples as in spikes..
#         # spect = resample(spect, spikes.shape[0], axis=0)
#         # spect = resample(spect, samples, axis=0)

#         # spect has 128 channels at this point, we can reduce the channels 
#         # for easy training, to match speech2text spectrogram 80 for now...
#         spect = resample(spect, self.num_freqs, axis=1)
#         return spect, spikes#np.expand_dims(spikes[:,ch], axis=1)


	

        

    
#     def fit(self, third=None):
#         """

#         """
#         # training_sent_ids = self.random_sent_ids[:self.size_training_dataset]
        
#         # collecting data after pre-processing...
#         train_spect_list = []
#         train_spikes_list = []
#         for sent in self.training_sent_ids:
#             spect, spikes = self.get_sample(sent, third=third)
#             train_spect_list.append(spect)
#             train_spikes_list.append(spikes)
        
#         self.strf_model.fit(X=train_spect_list, y=train_spikes_list)
#         corr = self.evaluate(third=third)
#         return corr
    
#     def evaluate(self, third=None):
#         # testing_sent_ids = self.random_sent_ids[self.size_training_dataset:]
    
#         # collecting data after pre-processing...
#         test_spect_list = []
#         test_spikes_list = []
#         for sent in self.testing_sent_ids:
#             spect, spikes = self.get_sample(sent, third=third)
#             test_spect_list.append(spect)
#             test_spikes_list.append(spikes)
        
#         corr = self.strf_model.corr(X=test_spect_list, y=test_spikes_list)
#         return corr
    
#     def get_coefficients(self):
#         """Returns the coefficients of the linear map from STRF to 
#         neural responses.
        
#         Returns:
#             ndarray = strf_model coefficients (num_ch, n_features_X, n_lags)
#         """
#         return self.strf_model.coef_




