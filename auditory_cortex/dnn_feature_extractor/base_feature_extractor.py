import os
import yaml
import numpy as np
from scipy.signal import resample
from abc import ABC, ABCMeta, abstractmethod, abstractproperty
import torch
import gc
from transformers import ClapModel, ClapProcessor
from memory_profiler import profile
from auditory_cortex import config_dir, results_dir, aux_dir, cache_dir

import logging
logger = logging.getLogger(__name__)



class BaseFeatureExtractor(ABC):
    def __init__(self, model, config, shuffled=False, sampling_rate=16000) -> None:
        super().__init__()
        self.model = model
        self.config = config
        self.shuffled = shuffled
        self.sampling_rate = sampling_rate
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.layer_names, self.layer_ids, self.layer_types, self.receptive_fields = self.get_config_details()
        self.num_layers = len(self.layer_names)
        self.features = {}
        self.register_hooks()
        

        if self.shuffled:
            self.shuffle_weights()
            # layers = self.reset_model_parameters()


            # if self.scale_factor is not None:
            # 	self.scale_weights()
            # self.randomly_reinitialize_weights(uniform=True)

    @abstractmethod
    def fwd_pass(self, aud):
        """DNN specific forward pass method."""
        pass


    def reset_model_parameters(self):
        """Reset weights of all the layers of the model.
        """
        logger.info(f"Randomly 'resetting' the network parameters...")
        layer_names = []
        named_modules = dict([*self.model.named_modules()])
        for name, layer in named_modules.items():
            if hasattr(layer, 'reset_parameters'):
                # print(f"{layer.__name__}")
                layer.reset_parameters()
                layer_names.append(name)
        return layer_names

            # if len(param.size()) > 1: #check if param is a weight tensor

    def shuffle_weights(self):
        """Shuffle weights of all the layers of the model.
        """
        logger.info(f"Randomly 'permuting' the network parameters...")
        for param in self.model.parameters():
            # flatten the parameter tensor and apply a random permutation...
            flattened_param = param.data.view(-1)
            shuffled_param = flattened_param[np.random.permutation(flattened_param.size(0))]

            # Reshape the shuffled_param back to original shape
            param.data = shuffled_param.view(param.size())

    def randomly_reinitialize_weights(self, uniform=False):
        """Randomly initialize weights of all the layers of the model.
        """
        logger.info(f"Initializing 'random weights' the network parameters...")
        with torch.no_grad():
            for param in self.extractor.model.parameters():
                if uniform:
                    param.data = torch.rand_like(param)
                else:
                    param.data = torch.randn_like(param)

    def scale_weights(self, scale_factor):
        """Randomly initialize weights of all the layers of the model.
        """
        logger.info(f"Scalling weights by factor: {scale_factor}")
        with torch.no_grad():
            for param in self.extractor.model.parameters():
                param.data = param.data*self.scale_factor


    # @profile
    def extract_features(self, stim_audios, sampling_rate, stim_durations=None, pad_time=None):
        """
        Returns raw features for all layers of the DNN..!

        Args:
            stim_audios (dict): dictionary of audio inputs for each sentence.
                {stim_id: audio}
            sampling_rate (int): sampling rate of the audio inputs.
            stim_durations (dict): dictionary of sentence durations.
                {stim_id: duration}
            pad_time (float): amount of padding time in seconds.

        Returns:
            dict of dict: read this as features[layer_id][stim_id]
        """
        features = {id:{} for id in self.layer_ids}
        for stim_id, audio in stim_audios.items():

            if sampling_rate != self.sampling_rate:
                n_samples = int(audio.size*self.sampling_rate/sampling_rate)
                audio = resample(audio, n_samples)
            
            if pad_time is not None:
                # print(f"Padding audio by {pad_time} seconds...")
                pad = int(pad_time*self.sampling_rate)
                padding = np.zeros((pad, ))
                audio = np.concatenate([padding, audio])

            # # needed only for Whisper...!
            if 'whisper' in self.model_name:
                bin_width = 20/1000.0   #20 ms for all layers except the very first...
                sent_duration = stim_durations[stim_id]
                if pad_time is not None:
                    sent_duration += pad_time
                # sent_samples = int(np.ceil(round(sent_duration/bin_width, 3)))
                sent_samples = int((sent_duration + bin_width/2)/bin_width)

            # self.translate(audio, grad=False)
            # _ = self.fwd_pass(audio)
            stim_features = self.get_features(audio)
            for layer_id in self.layer_ids:
                # features[layer_id][stim_id] = self.get_features(layer_id)
                layer_name = self.get_layer_name(layer_id)
                features[layer_id][stim_id] = stim_features[layer_name]
                if 'whisper' in self.model_name:
                    ## whisper networks gives features for 30s long clip,
                    ## extracting only the true initial samples...
                    layer_name = self.get_layer_name(layer_id)
                    if layer_name == 'model.encoder.conv1':
                        # sampling rate is 100 Hz for very first layer
                        # and 50 Hz for all the other layers...
                        feature_samples = 2*sent_samples
                    else:
                        feature_samples = sent_samples
                    features[layer_id][stim_id] = features[layer_id][stim_id][:feature_samples]
            del stim_features
            collected = gc.collect()
            # logger.debug(f"Garbage collector: collected {collected} objects.")
        return features


    def register_hooks(self):
        """Registers hooks for all the layers in the model."""
        
        # self.hooks = []
        for layer_name in self.layer_names:
            layer = dict([*self.model.named_modules()])[layer_name]
            layer.__name__ = layer_name
            hook = layer.register_forward_hook(self.create_hooks())
        # 	self.hooks.append(hook)
        # return self.hooks


    def create_hooks(self):
        """Creates hooks for all the layers in the model."""
        def fn(layer, inp, output):
            if 'rnn' in layer.__name__:
                features = output[0].data
                # output = output[1][0][1].squeeze()  # reading the 2nd half of data (only backward RNNs)
            else:
                output = output.squeeze()
                if 'conv' in layer.__name__:
                    if output.ndim > 2:
                        output = output.reshape(output.shape[0]*output.shape[1], -1)
                    output = output.transpose(0,1)
                elif 'coch' in self.model_name:
                    if output.ndim > 2:
                        output = output.reshape(output.shape[0]*output.shape[1], -1)
                    output = output.transpose(0,1)      # (time, features) format
                    if output.shape[1] > 2000:
                        # restrict the output to max 2000 features
                        output = output[:, :2000]   
                features = output
            self.features[layer.__name__] = features
        return fn


    def get_config_details(self):
        """
        Reads the model-specific configuration file and
        and details of the layers to be analysed.
        """
        
        num_layers = len(self.config['layers'])
    
        layer_names = []
        layer_ids = []
        layer_types = []
        receptive_fields = []
        for i in range(num_layers):
            layer_names.append(self.config['layers'][i]['layer_name'])
            layer_ids.append(self.config['layers'][i]['layer_id'])
            layer_types.append(self.config['layers'][i]['layer_type'])
            receptive_fields.append(self.config['layers'][i]['RF'])
        return layer_names, layer_ids, layer_types, receptive_fields

    def get_layer_names(self):
        return self.layer_names
    
    def get_layer_ids(self):
        return self.layer_ids
    
    def get_layer_name(self, layer_id):
        """Returns layer_name corresponidng to layer_ID"""
        ind = self.layer_ids.index(layer_id)
        return self.layer_names[ind]
    
    def get_features(self, audio):
        """Returns features for all layers of the DNN..!"""
        _ = self.fwd_pass(audio)
        # if 'rnn' in layer_name:
            # return self.features[layer_name].cpu()
        #	return self.features[layer].data[:,1024:] # only using fwd features (first half of concatenatation)
        features = {layer_name:feat.cpu() for layer_name, feat in self.features.items()}
        self.features = {}
        return features


    # def get_features(self, layer_id):
    # 	'''
    # 	Use to extract features for specific layer after calling 'translate()' method 
    # 	for given audio input.

    # 	Args:
    # 		layer_ID (int): layer identifier, assigned in config.

    # 	returns:
    # 		(dim, time) features extracted for layer at 'layer_ID'
    # 	'''
    # 	layer_name = self.get_layer_name(layer_id)
    # 	if 'rnn' in layer_name:
    # 		return self.features[layer_name].cpu()
    # 	#	return self.features[layer].data[:,1024:] # only using fwd features (first half of concatenatation)
    # 	else:
    # 		return self.features[layer_name].cpu()
    
    def translate(self, aud, grad=False):
        if grad:
            input = self.fwd_pass_tensor(aud)
        else:
            with torch.no_grad():
                input = self.fwd_pass(aud)
        return input
    
    @staticmethod
    def read_config_file(file_name):
        """Reads the configuration file for the model."""
        config_file = os.path.join(aux_dir, file_name)
        with open(config_file, 'r') as f:
            config = yaml.load(f, yaml.FullLoader)
        return config