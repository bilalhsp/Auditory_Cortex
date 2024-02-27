import math
import numpy as np
from scipy.signal import resample

import naplib as nl
from auditory_cortex import config
# from auditory_cortex.neural_data import dataset
from auditory_cortex.dataloader import DataLoader
from sklearn.linear_model import Ridge, ElasticNet



class STRF:
    def __init__(self, session, estimator, num_workers=1, num_freqs=32, 
            tmin=0, tmax = 0.3, bin_width=20, train_dataset_size=0.8):
        """
        Args:
            num_freqs (int): Number of frequency channels on spectrogram
            tmin: receptive field begins at (ms)
            tmax: receptive field ends at (ms)
            sfreq: sampling frequency of data (Hz)

            train_dataset_size: [0.0, 1.0]= fraction of total dataset 
            to be used as training/val split. Defalt=0.8
                (excluding the test split only)
        """
        self.model_name = 'strf_model'
        session = str(session)
        self.bin_width = bin_width
        data_dir = config['neural_data_dir']
        # self.dataset = NeuralData(data_dir, session)
        # self.dataset.extract_spikes(bin_width=self.bin_width, delay=0)
        # self.fs = self.dataset.fs
        self.dataloader = DataLoader()
        self.session_spikes = self.dataloader.get_session_spikes(session)
        self.fs = self.dataloader.metadata.get_sampling_rate()
        self.num_freqs = num_freqs # num_freqs in the spectrogram    

        # creating a STRF model...
        sfreq = 1000/bin_width # 50 (since bin_width is in ms)
        self.strf_model = nl.encoding.TRF(
            tmin, tmax, sfreq, estimator=estimator,
            n_jobs=num_workers, show_progress=True
            )

        # sents = np.arange(1,499)
        # self.random_sent_ids = np.random.permutation(sents)
        # self.size_training_dataset = int(train_dataset_size*(sents.size))
        sent_IDs = self.dataloader.sent_IDs
        self.testing_sent_ids = self.dataloader.test_sent_IDs
        self.training_sent_ids = sent_IDs[np.isin(sent_IDs, self.testing_sent_ids, invert=True)]

    def get_sample(self, sent, third=None):

        # spikes = self.dataset.unroll_spikes([sent], third=third).astype(np.float32)
        # aud = self.dataset.audio(sent)
        spikes = self.session_spikes[sent]
        aud = self.dataloader.metadata.stim_audio(sent)
        spect = nl.features.auditory_spectrogram(aud, self.fs)

        if third is not None:
            # bin_width = self.bin_width/1000.0
            # n = int(np.ceil(round(self.dataset.duration(sent)/bin_width, 3)))
            
            # # store boundaries of sent thirds...
            # one_third = int(n/3)
            # two_third = int(2*n/3)
            # sent_sections = [0, one_third, two_third, n]

            # spect = resample(spect, n, axis=0)
            # # print(spect.shape)
            # spect = spect[sent_sections[third-1]: sent_sections[third]]
            # print(spect.shape)
            ...
            
        else:
            spect = resample(spect, spikes.shape[0], axis=0)
        # read spikes for the sent id, as (time, channel)
        # spikes = self.spikes[sent].astype(np.float32)

        # get spectrogram for the audio inputs, as (time, freq) 
        # spect = nl.features.auditory_spectrogram(aud, self.fs)
        # resample spect to get same # number of time samples as in spikes..
        # spect = resample(spect, spikes.shape[0], axis=0)
        # spect = resample(spect, samples, axis=0)

        # spect has 128 channels at this point, we can reduce the channels 
        # for easy training, following examples lets reduce to 32 for now...
        spect_32 = resample(spect, self.num_freqs, axis=1)

        return spect_32, spikes#np.expand_dims(spikes[:,ch], axis=1)
    
    def fit(self, third=None):
        """

        """
        # training_sent_ids = self.random_sent_ids[:self.size_training_dataset]
        
        # collecting data after pre-processing...
        train_spect_list = []
        train_spikes_list = []
        for sent in self.training_sent_ids:
            spect, spikes = self.get_sample(sent, third=third)
            train_spect_list.append(spect)
            train_spikes_list.append(spikes)
        
        # # building a strf model...
        # tmin = 0 # receptive field begins at time=0
        # tmax = 0.3 # receptive field ends at a lag of 0.4 seconds
        # sfreq = 100 # sampling frequency of data

        # estimator = Ridge(10, max_iter=max_itr)
        # setting show_progress=False would disable the progress bar
        

        # strf_model.fit(X=spects, y=spikess[:,32])
        # strf_model.fit(X=inp, y=out)
        self.strf_model.fit(X=train_spect_list, y=train_spikes_list)
        corr = self.evaluate(third=third)
        return corr
    
    def evaluate(self, third=None):
        # testing_sent_ids = self.random_sent_ids[self.size_training_dataset:]
    
        # collecting data after pre-processing...
        test_spect_list = []
        test_spikes_list = []
        for sent in self.testing_sent_ids:
            spect, spikes = self.get_sample(sent, third=third)
            test_spect_list.append(spect)
            test_spikes_list.append(spikes)
        
        corr = self.strf_model.corr(X=test_spect_list, y=test_spikes_list)
        return corr
    
    def get_coefficients(self):
        """Returns the coefficients of the linear map from STRF to 
        neural responses.
        
        Returns:
            ndarray = strf_model coefficients (num_ch, n_features_X, n_lags)
        """
        return self.strf_model.coef_




