import torch
import torchaudio
import numpy as np

# import os
# import pandas as pd
# from scipy import linalg, signal

# local
import auditory_cortex.models as models
import auditory_cortex.utils as utils

import matplotlib.pyplot as plt

class OptimalInput():
    def __init__(self, model_name, load_features = True):
        self.linear_model = models.Regression(model_name, load_features=load_features)
        self.num_layers = self.linear_model.num_layers
        self.num_channels = {}
        self.B = {}

        for param in self.linear_model.model_extractor.extractor.model.parameters():
            param.requires_grad = False


    # def load_dataset(self, session):
    #     """Creates dataset and extracts features"""
    #     _ = self.linear_model.get_neural_spikes(session)
    #     self.num_channels[session] = self.linear_model.num_channels[session]

    # def get_num_channels(self, session):
    #     if session in self.num_channels.keys():
    #         return self.num_channels[session]
    #     else:
    #         print(f"Session information not loaded.")
    #         return -1


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


    def get_optimal_input(
            self, session, layer, ch, starting_sent=0, input_duration=1000,
            epochs=500, lr=0.5, w1=1, w2=100, use_cpu=False
            ):
        
        session = str(int(session))
        layer = int(layer)
        ch = int(ch)
        fs = 16000
        samples = int(fs*input_duration/1000)
        if starting_sent == 0:
            inp = torch.randn(samples, dtype=torch.float32)
        else:
            inp = torch.tensor(self.linear_model.dataset.audio(starting_sent), dtype=torch.float32)
        inp = inp.unsqueeze(dim=0)
        inp.requires_grad = True
        self.get_betas(session, use_cpu=use_cpu)      
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

def plot_spect(waveform, ax, cmap='viridis'):
    waveform = waveform * (2 ** 15)
    kaldi = torchaudio.compliance.kaldi.fbank(waveform, num_mel_bins=80, window_type='hanning')
    kaldi = normalize(kaldi)
    x_ticks = np.arange(0,kaldi.shape[0],20)
    data = ax.imshow(kaldi.transpose(1,0), cmap=cmap, origin='lower')
    ax.set_xticks(x_ticks, 10*x_ticks)
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('mel filters')
    return data
