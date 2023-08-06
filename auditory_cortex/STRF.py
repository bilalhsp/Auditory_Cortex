import math
import numpy as np
from scipy.signal import resample

import naplib as nl
from auditory_cortex import dataset, config
from sklearn.linear_model import Ridge, ElasticNet



class STRF:
    def __init__(self, session, bin_width=20):
        session = str(session)

        self.bin_width = bin_width

        data_dir = config['neural_data_dir']
        self.dataset = dataset.NeuralData(data_dir, session)
        self.dataset.extract_spikes(bin_width=self.bin_width, delay=0)
        self.fs = self.dataset.fs



        sents = np.arange(1,499)
        self.random_sent_ids = np.random.permutation(sents)

    def get_sample(self, sent, num_freqs=32, third=None):

        spikes = self.dataset.unroll_spikes([sent], third=third).astype(np.float32)

        aud = self.dataset.audio(sent)
        spect = nl.features.auditory_spectrogram(aud, self.fs)

        if third is not None:
            bin_width = self.bin_width/1000.0
            n = int(np.ceil(round(self.dataset.duration(sent)/bin_width, 3)))
            
            # store boundaries of sent thirds...
            one_third = int(n/3)
            two_third = int(2*n/3)
            sent_sections = [0, one_third, two_third, n]

            spect = resample(spect, n, axis=0)
            # print(spect.shape)
            spect = spect[sent_sections[third-1]: sent_sections[third]]
            # print(spect.shape)
            
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
        spect_32 = resample(spect, num_freqs, axis=1)

        return spect_32, spikes#np.expand_dims(spikes[:,ch], axis=1)
    
    def fit(self, estimator, num_workers=1, num_freqs=32, 
            tmin=0, tmax = 0.3, sfreq = 100, third=None):
        """
        Args:
            num_freqs (int): Number of frequency channels on spectrogram
            tmin: receptive field begins at (ms)
            tmax: receptive field ends at (ms)
            sfreq: sampling frequency of data (Hz)
        """

        training_sent_ids = self.random_sent_ids[:350]
        
        # collecting data after pre-processing...
        train_spect_list = []
        train_spikes_list = []
        for sent in training_sent_ids:
            spect, spikes = self.get_sample(sent, num_freqs=num_freqs)
            train_spect_list.append(spect)
            train_spikes_list.append(spikes)
        
        # # building a strf model...
        # tmin = 0 # receptive field begins at time=0
        # tmax = 0.3 # receptive field ends at a lag of 0.4 seconds
        # sfreq = 100 # sampling frequency of data

        # estimator = Ridge(10, max_iter=max_itr)
        # setting show_progress=False would disable the progress bar
        strf_model = nl.encoding.TRF(tmin, tmax, sfreq, estimator=estimator,
                                    n_jobs=num_workers ,show_progress=True)

        # strf_model.fit(X=spects, y=spikess[:,32])
        # strf_model.fit(X=inp, y=out)
        strf_model.fit(X=train_spect_list, y=train_spikes_list)
        corr = self.evaluate(strf_model, third=third)
        return strf_model, corr
    
    def evaluate(self, model, num_freqs=32, third=None):
        testing_sent_ids = self.random_sent_ids[350:]
    
        # collecting data after pre-processing...
        test_spect_list = []
        test_spikes_list = []
        for sent in testing_sent_ids:
            spect, spikes = self.get_sample(sent, num_freqs=num_freqs, third=third)
            test_spect_list.append(spect)
            test_spikes_list.append(spikes)
        
        corr = model.corr(X=test_spect_list, y=test_spikes_list)
        return corr
