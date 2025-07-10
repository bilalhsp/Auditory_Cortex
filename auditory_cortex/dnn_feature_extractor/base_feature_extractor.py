import os
import gc
import yaml
import torch
import numpy as np
from scipy.signal import resample
from abc import ABC, abstractmethod

from memory_profiler import profile
from auditory_cortex import aux_dir

import logging
logger = logging.getLogger(__name__)

FEATURE_EXTRACTOR_REGISTRY  = {}

def register_feature_extractor(model_name: str):
    """
    Decorator to register a feature extractor class.
    
    Args:
        model_name (str): name of the model to be used.
    
    Returns:
        function: returns the decorated class.
    """
    def decorator(cls):
        if model_name in FEATURE_EXTRACTOR_REGISTRY :
            raise ValueError(f"Model {model_name} is already defined!")
        FEATURE_EXTRACTOR_REGISTRY[model_name] = cls
        return cls
    return decorator

def create_feature_extractor(model_name, shuffled=False, **kwargs):
    if model_name not in FEATURE_EXTRACTOR_REGISTRY :
        raise ValueError(f"Model {model_name} is not defined!")
    return FEATURE_EXTRACTOR_REGISTRY[model_name](shuffled, **kwargs)

def list_dnn_models():
    """Returns the list of available feature extractors."""
    return list(FEATURE_EXTRACTOR_REGISTRY.keys())


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
                layer.reset_parameters()
                layer_names.append(name)
        return layer_names

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
                pad = int(pad_time*self.sampling_rate)
                padding = np.zeros((pad, ))
                audio = np.concatenate([padding, audio])

            # # needed only for Whisper...!
            if 'whisper' in self.model_name:
                bin_width = 20/1000.0   #20 ms for all layers except the very first...
                sent_duration = stim_durations[stim_id]
                if pad_time is not None:
                    sent_duration += pad_time
                sent_samples = int((sent_duration + bin_width/2)/bin_width)

            stim_features = self.get_features(audio)
            for layer_id in self.layer_ids:
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
        return features


    def register_hooks(self):
        """Registers hooks for all the layers in the model."""
        # Not saving the hooks as they are not needed for the analysis.
        for layer_name in self.layer_names:
            layer = dict([*self.model.named_modules()])[layer_name]
            layer.__name__ = layer_name
            hook = layer.register_forward_hook(self.create_hooks())


    def create_hooks(self):
        """Creates hooks for all the layers in the model."""
        def fn(layer, inp, output):
            if 'rnn' in layer.__name__:
                features = output[0].data
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
        features = {layer_name:feat.cpu() for layer_name, feat in self.features.items()}
        self.features = {}
        return features

    
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