# import os
import torch
# import torchaudio
import numpy as np

# import os
# import pandas as pd
# from scipy import linalg, signal

# local
import auditory_cortex.deprecated.models as models
from auditory_cortex.utils import SyntheticInputUtils

# import matplotlib.pyplot as plt

class OptimalInput():
    def __init__(self, model_name, load_features = True):
        self.linear_model = models.Regression(model_name, load_features=load_features)
        self.model_name = model_name
        self.num_layers = self.linear_model.num_layers
        self.num_channels = {}
        self.B = {}

        for param in self.linear_model.model_extractor.extractor.model.parameters():
            param.requires_grad = False

    def get_betas(self, session, use_cpu=False, force_redo=False):
        """
        Returns betas for all layers and channels 

        Args:
            session: ID of session 
        """
        # check if self.B holds result for current session,
        # if not compute B's and remove for all other sessions.
        if session not in self.B.keys() or force_redo: 
               
            # self.B = {}
            # print(f"Initializing with random betas...")
            # self.B[session] = torch.randn((12,250, 64))
            print(f"Computing betas...")
            self.B[session] = torch.tensor(
                self.linear_model.get_betas(session, use_cpu=use_cpu, force_redo=force_redo),
                dtype=torch.float32
                )
        return self.B[session]


    def get_optimal_input(
            self, session, layer, ch, starting_sent=0, input_duration=1000,
            epochs=500, lr=0.5, w1=1, w2=100, use_cpu=False, force_redo=False
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

        if self.model_name == 'speech2text':
            inp = SyntheticInputUtils.get_spectrogram(inp)
        elif self.model_name == 'deepspeech2':
            inp = self.linear_model.model_extractor.extractor.get_spectrogram(inp)
        
        inp = inp.unsqueeze(dim=0)
        inp.requires_grad = True
        self.get_betas(session, use_cpu=use_cpu, force_redo=force_redo)      
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
            # basic_loss_history.append(loss.item())            
            # TVloss = torch.nn.functional.mse_loss(inp[:,1:], inp[:,:-1])
            # TVloss_history.append(TVloss.item())

            # print(f'Loss: {loss}')
            # loss = w1*loss + w2*TVloss
            loss.backward(inputs=inp)
            ### Normalize grad by the 'global norm'
            # var, mean = torch.var_mean(inp.grad, unbiased=False)
            # inp.grad = inp.grad / (torch.sqrt(var) + 1e-8)
            # grads.append(inp.grad.clone().detach().numpy())
            
            opt.step()
            
            ### Clip input values at -1 and 1 (after update)
            # with torch.no_grad():
            #     inp[inp > 1] = 1
            #     inp[inp<-1] = -1
            # grads.append(np.zeros(16000))
            loss_history.append(loss.item())
            inps_history.append(inp.clone().detach())
        return inps_history, loss_history, basic_loss_history, TVloss_history, grads

    def fwd_pass(self, input, layer_id, ch, session):
            layer_idx = self.get_layer_index(layer_id=layer_id)
            self.linear_model.model_extractor.translate(input, grad=True)
            feats = self.linear_model.model_extractor.get_features(layer_idx)
            # pred = utils.predict(feats, self.B[layer,:,ch])
            betas = self.get_betas(session)
            pred = feats@ betas[layer_idx,:,ch]
            return pred
            
    def get_layer_index(self, layer_id):
        """Returns layer index for layer ID (defined in config).
        Calls instance method of 'self.linear_model.model_extractor'
        """
        return self.linear_model.model_extractor.get_layer_index(layer_id)




