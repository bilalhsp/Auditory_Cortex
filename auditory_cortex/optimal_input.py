import numpy as np
import torch
import os
import pandas as pd
from scipy import linalg, signal
# from transformers import Speech2TextForConditionalGeneration, Speech2TextProcessor,Wav2Vec2Processor, Wav2Vec2ForCTC

# from auditory_cortex.dataset import Neural_Data
# from auditory_cortex.feature_extractors import Feature_Extractor_S2T,Feature_Extractor_GRU,FeatureExtractorW2L


# local
import auditory_cortex.regression as Reg
import auditory_cortex.utils as utils

#from sklearn.decomposition import PCA
# import rnn_model.speech_recognition as speech_recognition
import matplotlib.pyplot as plt
import torchaudio

class optimal_input():
    def __init__(self, model, load_features = True):
   
    # def __init__(self, dir, subject, model, load_features = True):

        self.linear_model = Reg.transformer_regression(model, load_features=load_features)
        self.num_layers = self.linear_model.num_layers
        self.num_channels = {}





        # self.dir = os.path.join(dir, subject)
        # print("Creating dataset and other objects...")
        # self.dataset = Neural_Data(dir, subject)
        # print(f"Creating opt_input obj for: '{model.model_name}'")
        # self.model = model
        # self.model_extractor = FeatureExtractorW2L(self.model)
        # self.model_name = model.model_name
        # self.num_channels = self.dataset.num_channels
        # self.layers = self.linear_model.model_extractor.layers
        self.B = {}

        for param in self.linear_model.model_extractor.extractor.model.parameters():
            param.requires_grad = False

    # def load_features_and_spikes(self, bin_width=40, delay=0, offset=0, sents=None):
    #     if sents == None:
    #         sents = np.arange(1,499)
        
    #     self.raw_features = self.extract_features(sents=sents)
    #     self.features = self.unroll_time(self.resample(self.raw_features, bin_width))
    #     self.spikes = self.all_channel_spikes(bin_width=bin_width, delay=delay, offset=offset, sents=sents)

    # def extract_features(self, sents = np.arange(1,499), grad=False):
    #     """
    #     Returns all layer features for given 'sents'
        
    #     Args:
    #         sents (list, optional): List of sentence ID's to get the features for. 

    #     Returns:
    #         List of dict: List index corresponds to layer number carrying 
    #                     dict of extracted features for all sentences. 
    #     """
    #     # if sents == None:
    #     #     sents = np.arange(1,499)
    #     features = [{} for _ in range(len(self.layers))]
    #     for x, i in enumerate(sents):
    #         self.model_extractor.translate(self.dataset.audio(i), grad = grad)
    #         for j, l in enumerate(self.layers):
    #             features[j][i] = self.model_extractor.features[l]
    #             if self.model_name=='wav2vec':
    #                     features[j][i] = features[j][x][:self.seq_lengths[i]]
    #     return features

    # def resample(self, features, bin_width):
    #     """
    #     resample all layer features to specific bin_width

    #     Args:
    #         bin_width (float): width of data samples in ms (1000/sampling_rate).

    #     Returns:
    #         List of dict: all layer features (resampled at required sampling_rate).
    #     """
    #     resampled_features = [{} for _ in range(len(self.layers))]
    #     bin_width = bin_width/1000 # ms
    #     for sent in features[0].keys():
    #         n = int(np.ceil(round(self.dataset.duration(sent)/bin_width, 3)))
    #         for j, l in enumerate(self.layers):
    #             resampled_features[j][sent] = signal.resample(features[j][sent],n, axis=0)
    #     return resampled_features

    # def unroll_time(self, features):
    #     """
    #     Unroll and concatenate time axis of extracted features.

    #     Args:
    #         features (List of dict): features for all layers.
        
    #     Returns:
    #         dict: 
    #     """
    #     feats = {}
    #     for j, l in enumerate(self.layers):
    #         feats[j] = np.concatenate([features[j][sent] for sent in features[j].keys()], axis=0)
    #     return feats
    # def all_channel_spikes(self, bin_width=40, delay=0, offset=0, sents = np.arange(1,499)):
    #     spikes = []
    #     result = {}
    #     for x,i in enumerate(sents):
    #         spikes.append(self.dataset.retrieve_spike_counts(sent=i,win=bin_width,delay=delay,early_spikes=False,offset=offset))
    #     for ch in range(self.dataset.num_channels):
    #         result[ch] = np.concatenate([spikes[i][ch] for i in range(len(spikes))], axis=0)

    #     return result
    def load_dataset(self, session):
        """Creates dataset and extracts features"""
        _ = self.linear_model.get_neural_spikes(session)
        self.num_channels[session] = self.linear_model.num_channels[session]

    def get_num_channels(self, session):
        if session in self.num_channels.keys():
            return self.num_channels[session]
        else:
            print(f"Session information not loaded.")
            return -1


    def get_betas(self, session, use_cpu=False):
        """
        Returns betas for all layers and channels 

        Args:
            session: ID of session 
        """
        # check if self.B holds result for current session,
        # if not compute B's and remove for all other sessions.
        if session not in self.B.keys():    
            self.B = {}
            self.B[session] = torch.tensor(
                self.linear_model.get_betas(session, use_cpu=use_cpu),
                dtype=torch.float32
                )
        # n_channels = self.dataset.num_channels
        # x = self.features[layer]
        # y = np.stack([self.spikes[i] for i in range(n_channels)], axis=1)
        # # regression_param replaced with reg()
        # #self.B[layer] = torch.tensor(utils.regression_param(x, y), dtype=torch.float32)
        # self.B[layer] = torch.tensor(utils.reg(x, y), dtype=torch.float32)
        
    # def get_input(self, sent=12, random=False):

    #     if random:
    #         inp = torch.randn(16000, dtype=torch.float32)
    #     else:
    #         inp = torch.tensor(self.dataset.audio(sent), dtype=torch.float32)
    #     inp = inp.unsqueeze(dim=0)
    #     return inp

    def optimize(self, session, layer, ch, starting_sent=0, input_duration=1000, epochs=500, lr=0.5, w1=1, w2=100):
        fs = 16000
        samples = int(fs*input_duration/1000)
        if starting_sent == 0:
            inp = torch.randn(samples, dtype=torch.float32)
        else:
            inp = torch.tensor(self.linear_model.dataset.audio(starting_sent), dtype=torch.float32)
        inp = inp.unsqueeze(dim=0)
        inp.requires_grad = True
        self.get_betas(session, use_cpu=True)      
        opt = torch.optim.Adam([inp], lr=lr)
        loss_history = []
        inps_history = []
        basic_loss_history = []
        TVloss_history = []
        grads = []
        inps_history.append(inp.clone().detach())
        for i in range(epochs):
            # fwd pass
            opt.zero_grad()
            pred = self.fwd_pass(inp, layer, ch, session)
            loss = -pred.mean()
            basic_loss_history.append(loss.item())            
            TVloss = torch.nn.functional.mse_loss(inp[:,1:], inp[:,:-1])
            TVloss_history.append(TVloss.item())

            # print(f'Loss: {loss}')
            loss = w1*loss + w2*TVloss
            loss.backward(inputs=inp)
            ### Normalize grad by the 'global norm'
            var, mean = torch.var_mean(inp.grad, unbiased=False)
            inp.grad = inp.grad / (torch.sqrt(var) + 1e-8)
            grads.append(inp.grad.clone().detach().numpy())
            
            opt.step()
            
            ### Clip input values at -1 and 1 (after update)
            with torch.no_grad():
                inp[inp > 1] = 1
                inp[inp<-1] = -1
            # grads.append(np.zeros(16000))
            loss_history.append(loss.item())
            inps_history.append(inp.clone().detach())
        return inps_history, loss_history, basic_loss_history, TVloss_history, grads

    def fwd_pass(self, input, layer, ch, session):
            self.linear_model.model_extractor.translate(input, grad=True)
            feats = self.linear_model.model_extractor.get_features(layer)
            # pred = utils.predict(feats, self.B[layer,:,ch])
            pred = feats@ self.B[session][layer,:,ch]
            # self.model.eval()
            # self.model(input)            
            # feats = self.model_extractor.features[self.layers[layer]]
            # pred = feats @ self.B[layer][:,ch]
            return pred
            
def normalize(x):
    """
    ONLY USED FOR VISUALIZING THE SPECTROGRAM...!
    Normalizes the spectrogram (obtained using kaldi transform),
    done to match the spectrogram exactly to the Speec2Text transform
    """
    mean = x.mean(axis=0)
    square_sums = (x ** 2).sum(axis=0)
    x = np.subtract(x, mean)
    var = square_sums / x.shape[0] - mean ** 2
    std = np.sqrt(np.maximum(var, 1e-10))
    x = np.divide(x, std)

    return x

def plot_spect(waveform, ax, cmap='turbo'):
    waveform = waveform * (2 ** 15)
    kaldi = torchaudio.compliance.kaldi.fbank(waveform, num_mel_bins=80, window_type='hanning')
    kaldi = normalize(kaldi)
    x_ticks = np.arange(0,kaldi.shape[0],20)
    data = ax.imshow(kaldi.transpose(1,0), cmap=cmap, origin='lower')
    ax.set_xticks(x_ticks, 10*x_ticks)
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('mel filters')
    return data
