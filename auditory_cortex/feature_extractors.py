import os
import yaml
import torch
import cupy as cp
import numpy as np
from torch import nn, Tensor
from abc import ABC, abstractmethod
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from transformers import Speech2TextForConditionalGeneration, Speech2TextProcessor
from transformers import AutoProcessor, WhisperForConditionalGeneration

from deepspeech_pytorch.model import DeepSpeech
import deepspeech_pytorch.loader.data_loader as data_loader
from deepspeech_pytorch.configs.train_config import SpectConfig

# local
from auditory_cortex import config_dir
from wav2letter.models import Wav2LetterRF


class FeatureExtractor():
    def __init__(self, model_name = 'wave2letter_modified'):
        # super(FeatureExtractor, self).__init__()
        self.layers = []
        self.layer_ids = []
        self.layer_types = []
        self.receptive_fields = []
        self.features_delay = []        # features delay needed to compensate for the RFs
        # self.bin_widths = []
        # self.offsets = []
        self.features = {}
        # self.model = model
            
        # read yaml config file
        config_file = os.path.join(config_dir, f"{model_name}_config.yml")
        with open(config_file, 'r') as f:
            self.config = yaml.load(f, yaml.FullLoader)
        self.num_layers = len(self.config['layers'])
        self.use_pca = self.config['use_pca']
        if self.use_pca:
            self.pca_comps = self.config['pca_comps']
        
        # create feature extractor as per model_name
        if model_name == 'wave2letter_modified':
            self.extractor = FeatureExtractorW2L(self.config['saved_checkpoint'])
        elif model_name == 'wave2vec2':
            self.extractor = FeatureExtractorW2V2()
        elif model_name == 'speech2text':
            self.extractor = FeatureExtractorS2T()
        elif model_name == 'whisper':
            self.extractor = FeatureExtractorWhisper()
        elif model_name == 'deepspeech2':
            self.extractor = FeatureExtractorDeepSpeech2(self.config['saved_checkpoint'])
        else:
            raise NotImplementedError(f"FeatureExtractor class does not support '{model_name}'")

        for i in range(self.num_layers):
            self.layers.append(self.config['layers'][i]['layer_name'])
            self.layer_ids.append(self.config['layers'][i]['layer_id'])
            self.layer_types.append(self.config['layers'][i]['layer_type'])
            self.receptive_fields.append(self.config['layers'][i]['RF'])

            # self.bin_widths.append(self.config['layers'][i]['bin_width'])
            # self.offsets.append(self.config['layers'][i]['offset'])

        # Register fwd hooks for the given layers
        for layer_name in self.layers:
            layer = dict([*self.extractor.model.named_modules()])[layer_name]
            layer.__name__ = layer_name
            layer.register_forward_hook(self.create_hooks())
            
    def create_hooks(self):
        def fn(layer, _, output):
            if 'rnn' in layer.__name__:
            #    output = output[0]
                output = output[1][0][1].squeeze()  # reading the 2nd half of data (only backward RNNs)
            else:
                output = output.squeeze()
                if 'conv' in layer.__name__:
                    if output.ndim > 2:
                        output = output.reshape(output.shape[0]*output.shape[1], -1)
                    output = output.transpose(0,1)
            self.features[layer.__name__] = output
        return fn

    def get_features(self, layer_index):
        '''
        Use to extract features for specific layer after calling 'translate()' method 
        for given audio input.

        Args:
        
            layer_index (int): layer index in the range [0, Total_Layers)

        returns:
            (dim, time) features extracted for layer at 'layer_index' location 
        '''
        layer = self.layers[layer_index]
        if 'rnn' in layer:
           return self.features[layer]
        #    return self.features[layer].data[:,1024:] # only using fwd features (first half of concatenatation)
        else:
            return self.features[layer]

    def def_bin_width(self, layer):
        def_w = self.bin_widths[layer]
        offset = self.offsets[layer]
        return def_w, offset

    def translate(self, aud, grad=False):
        if grad:
            input = self.extractor.fwd_pass_tensor(aud)
        else:
            with torch.no_grad():
                input = self.extractor.fwd_pass(aud)
        return input


class FeatureExtractorW2L():
    def __init__(self, checkpoint):		
        self.model = Wav2LetterRF.load_from_checkpoint(checkpoint)
        print(f"Loading from checkpoint: {checkpoint}")
    

    def fwd_pass(self, aud):
        """
        Forward passes audio input through the model and captures 
        the features in the 'self.features' dict.

        Args:
            aud (ndarray): single 'wav' input of shape (t,) 
        
        Returns:
            input (torch.Tensor): returns the torch Tensor of the input sent passed through the model.
        """
        input = torch.tensor(aud, dtype=torch.float32)#, requires_grad=True)
        input = input.unsqueeze(dim=0)
        input.requires_grad=True
        self.model.eval()
        out = self.model(input)
        return input
    
    def fwd_pass_tensor(self, aud):
        """
        Forward passes audio input through the model and captures 
        the features in the 'self.features' dict.

        Args:
            aud (tensor): input tensor 'wav' input of shape (1, t) 
        
        Returns:
            input (torch.Tensor): returns the torch Tensor of the input sent passed through the model.
        """
        self.model.eval()
        out = self.model(aud)
        return input

class FeatureExtractorW2V2():
    def __init__(self):
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        # self.model = model 
        ########################## NEED TO MAKE THIS CONSISTENT>>>##################
    def fwd_pass(self, aud):
        """
        Forward passes audio input through the model and captures 
        the features in the 'self.features' dict.

        Args:
            aud (ndarray): single 'wav' input of shape (t,) 
        
        Returns:
            input (torch.Tensor): returns the torch Tensor of the input sent passed through the model.
        """
        input = aud.astype(np.float64)
        input_values = self.processor(input, sampling_rate=16000, return_tensors="pt", padding="longest").input_values  # Batch size 1
        self.model.eval()
        # with torch.no_grad():
        logits = self.model(input_values).logits

        # input = torch.tensor(aud, dtype=torch.float32)#, requires_grad=True)
        # input = input.unsqueeze(dim=0)
        # input.requires_grad=True
        # self.model.eval()
        # out = self.model(input)
        return input

class FeatureExtractorS2T():
    def __init__(self):
        self.model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-large-librispeech-asr")
        self.processor = Speech2TextProcessor.from_pretrained("facebook/s2t-large-librispeech-asr")
                
    def fwd_pass(self, aud):
        input_features = self.processor(aud,padding=True, sampling_rate=16000, return_tensors="pt").input_features
        # with torch.no_grad():
        self.model.eval()
        generated_ids = self.model.generate(input_features, max_new_tokens=200)
        return generated_ids
    
    
class FeatureExtractorWhisper():
    def __init__(self):
        self.processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
        self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
				
    def fwd_pass(self, aud):
        input_features = self.processor(aud, sampling_rate=16000, return_tensors="pt").input_features
        # with torch.no_grad():
        self.model.eval()
        generated_ids = self.model.generate(inputs=input_features, max_new_tokens=400)
            # transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_ids
    
class FeatureExtractorDeepSpeech2():
    def __init__(self, checkpoint):

        audio_config = SpectConfig()
        self.parser = data_loader.AudioParser(audio_config, normalize=True)
        self.model = DeepSpeech.load_from_checkpoint(checkpoint_path=checkpoint)
        # self.processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
        # self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
				
    def fwd_pass(self, aud):
        spect = self.parser.compute_spectrogram(aud)
        spect = spect.unsqueeze(dim=0).unsqueeze(dim=0)

        # length of the spect along time
        lengths = torch.tensor([spect.shape[-1]], dtype=torch.int64)
        out = self.model(spect, lengths)
        return out






        # pass
        # input_features = self.processor(aud, sampling_rate=16000, return_tensors="pt").input_features
        # # with torch.no_grad():
        # self.model.eval()
        # generated_ids = self.model.generate(inputs=input_features, max_new_tokens=400)
        #     # transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # return generated_ids
    


# class FeatureExtractorW2L(FeatureExtractor):
# 	def __init__(self, model):
# 		super(FeatureExtractorW2L, self).__init__(model, f'{model.model_name}.yml')
		
# 	def fwd_pass(self, aud):
# 		"""
# 		Forward passes audio input through the model and captures 
# 		the features in the 'self.features' dict.

# 		Args:
# 			aud (ndarray): single 'wav' input of shape (t,) 
		
# 		Returns:
# 			input (torch.Tensor): returns the torch Tensor of the input sent passed through the model.
# 		"""
# 		input = torch.tensor(aud, dtype=torch.float32)#, requires_grad=True)
# 		input = input.unsqueeze(dim=0)
# 		input.requires_grad=True
# 		self.model.eval()
# 		out = self.model(input)
# 		return input

# class FeatureExtractorW2V2(FeatureExtractor):
# 	def __init__(self, model):
# 		super(FeatureExtractorW2V2, self).__init__(model, f'wav2vec2.yml')
# 		self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
# 		# self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
# 		self.model = model 
# 		########################## NEED TO MAKE THIS CONSISTENT>>>##################
# 	def fwd_pass(self, aud):
# 		"""
# 		Forward passes audio input through the model and captures 
# 		the features in the 'self.features' dict.

# 		Args:
# 			aud (ndarray): single 'wav' input of shape (t,) 
		
# 		Returns:
# 			input (torch.Tensor): returns the torch Tensor of the input sent passed through the model.
# 		"""
# 		input = aud.astype(np.float64)
# 		input_values = self.processor(input, sampling_rate=16000, return_tensors="pt", padding="longest").input_values  # Batch size 1
# 		self.model.eval()
# 		with torch.no_grad():
# 			logits = self.model(input_values).logits
		
# 		# input = torch.tensor(aud, dtype=torch.float32)#, requires_grad=True)
# 		# input = input.unsqueeze(dim=0)
# 		# input.requires_grad=True
# 		# self.model.eval()
# 		# out = self.model(input)
# 		return input



# class Feature_Extractor_S2T(FeatureExtractor):
# 	def __init__(self, model, processor):
# 		super(Feature_Extractor_S2T, self).__init__(model, 'speech2text.yml')
# 		self.processor = processor 

# 	def fwd_pass(self, aud):
# 		inputs_features = self.processor(aud,padding=True, sampling_rate=16000, return_tensors="pt").input_features
# 		with torch.no_grad():
# 			self.model.eval()
# 		generated_ids = self.model.generate(inputs_features)
# 		return generated_ids

############ Temporarily commented out on May 4, 2023 
#########################################################################################################



####################################################3
######## below this line, everything was commented out long ago and is not  used

# class FeatureExtractorW2L(nn.Module):
#   def __init__(self, model):
#     super(FeatureExtractorW2L, self).__init__()
#     self.model = model
#     config_file = os.path.join(
#       os.path.dirname(os.path.abspath(__file__)),
#       'conf',
#       f'{model.model_name}.yml'
#       )
#     with open(config_file, 'r') as f:
#       self.config = yaml.load(f, yaml.FullLoader)
#     self.num_layers = len(self.config)
#     self.layers = []
#     self.bin_widths = []
#     self.offsets = []
#     for i in range(self.num_layers):
#       self.layers.append(self.config[i]['layer_name'])
#       self.bin_widths.append(self.config[i]['bin_width'])
#       self.offsets.append(self.config[i]['offset'])

#     # self.layers = ["conv1","conv2","conv3","conv4","conv5","conv6","conv7","conv8","conv9",
#     #               "conv10","conv11","conv12","conv13","conv14","conv15",
#     #               #"conv16","conv17","conv18",
#     #               ]
#     # 
#     self.features = {}
#     # Register a hook for the given layer
#     for layer_name in self.layers:
#       layer = dict([*self.model.named_modules()])[layer_name]
#       layer.__name__ = layer_name
#       layer.register_forward_hook(self.create_hooks())
          
#   def create_hooks(self):
#     def fn(layer, _, output):
#       output = output.squeeze()
#       if 'conv' in layer.__name__:
#             output = output.transpose(0,1)
#       self.features[layer.__name__] = output
#     return fn

#   def get_features(self, index):
#     '''
#     arguments:
#     - index: (int) layer index in the range [0, 18)
#     returns:
#     - (dim, time) features extracted for layer at 'index' location 
#     '''
#     return self.features[self.layers[index]]

#   def def_bin_width(self, layer):
#     def_w = self.bin_widths[layer]
#     offset = self.offsets[layer]
#     return def_w, offset

#   @torch.no_grad()
#   def forward(self, input):
#     # input: 'ndarray' of shape (t,) 
#     #(single audio input in the form of ndarray)
#     input = torch.tensor(input, dtype=torch.float32)
#     input = input.unsqueeze(dim=0)
#     self.model.eval()
#     return self.model(input)

#   def translate(self, aud, fs = 16000):
#     inputs_features = aud
#     generated_ids = self.forward(inputs_features)


# class Feature_Extractor_S2T(nn.Module):
#   def __init__(self, model, processor, layers):
#     super(Feature_Extractor_S2T, self).__init__()
#     self.model = model
#     self.processor = processor
#     self.features = {}
#     print("S2T feature extractor created...!")
    
#     # Register a hook for the given layer
#     for layer_name in layers:
#       layer = dict([*self.model.named_modules()])[layer_name]
#       layer.__name__ = layer_name
#       layer.register_forward_hook(self.create_hooks())
          
#   def create_hooks(self):
#     def fn(layer, _, output):
#       output = output.squeeze()
#       if 'conv' in layer.__name__:
#             output = output.transpose(0,1)
#       # if layer.__name__ in ["model.encoder.conv.conv_layers.0","model.encoder.conv.conv_layers.1"]:
#       #       output = output.transpose(0,1)
#       self.features[layer.__name__] = output
#     return fn
    
#   def forward(self, input):
#     return self.model.generate(input)
  
#   def translate(self, aud, fs = 16000):
#     inputs_features = self.processor(aud,padding=True, sampling_rate=fs, return_tensors="pt").input_features
#     generated_ids = self.forward(inputs_features)
  
#   def def_bin_width(self, layer):
#     if layer <1:
#       def_w = 20
#       offset = -0.25 
#     else:
#       def_w = 40
#       offset = 0.39
#     return def_w, offset

class Feature_Extractor_GRU(nn.Module):
  def __init__(self, model: nn.Module, layers):
    super(Feature_Extractor_GRU, self).__init__()
    self.model = model
    self.features = {}
    
    # Register a hook for the given layer
    for layer_name in layers:
      layer = dict([*self.model.named_modules()])[layer_name]
      layer.__name__ = layer_name
      layer.register_forward_hook(self.create_hooks())
          
  def create_hooks(self):
    def fn(layer, _, output):
      self.features[layer.__name__] = output[0].squeeze()
    return fn
    
  def forward(self, input):
    return self.model(input)
  
  def prepare_GRU_input(self, aud):
    aud = aud.astype(np.float32) 
    input = torch.tensor(aud)
    aud_spect = self.spect(input)
    aud_spect = aud_spect.unsqueeze(dim=0)
    aud_spect = aud_spect.unsqueeze(dim=0).type(torch.float32)
    return aud_spect
  
  def translate(self, aud, fs = 16000):
    inputs_features = self.prepare_GRU_input(aud)
    generated_ids = self.forward(inputs_features)
  
  def def_bin_width(self, layer):
    # Same for all layers...!
    # Need to verify these...!
    def_w = 25
    offset = 0.001 
    return def_w, offset

class feature_extractor_wav2vec(nn.Module):
  def __init__(self, model,  processor, layers):
    super(feature_extractor_wav2vec, self).__init__()
    self.model = model
    self.processor = processor
    self.factor = {layer:i for i, layer in enumerate(reversed(layers))}
    self.features = {}
    
    # Register a hook for the given layer
    for layer_name in layers:
      layer = dict([*self.model.named_modules()])[layer_name]
      layer.__name__ = layer_name
      layer.register_forward_hook(self.create_hooks())
          
  def create_hooks(self):
    def fn(layer, _, output):
      output = output.squeeze().transpose(0,1).detach().numpy()
      #print("Before down_sample in extractor...!")
      #print(output.shape)
      output = utils.down_sample(output, 2**self.factor[layer.__name__]) 
      #print("After down_sample in extractor...!")
      #print(output.shape)
      self.features[layer.__name__] = output
    return fn
    
  def forward(self, input):
    return self.model(input)
  
  def prepare_wav2vec_input(self, aud, fs):
    input_ids = self.processor(aud, padding=True, sampling_rate=fs, return_tensor="pt").input_values[0][None,:]
    input_ids = torch.tensor(input_ids.astype(np.float32))
    return input_ids
  
  def translate(self, aud, fs = 16000):
    inputs_features = self.prepare_wav2vec_input(aud, fs)
    generated_ids = self.forward(inputs_features)
  
  def transcribe(self, aud, fs):
    input_ids = self.processor(aud, padding=True, sampling_rate=fs, return_tensor="pt").input_values[0][None,:]
    input_ids = torch.tensor(input_ids.astype(np.float32)) 
    logits = self.model(input_ids).logits
    tokens = logits.argmax(axis=-1)
    out = self.processor.decode(tokens.squeeze())
    return out 

  def def_bin_width(self, layer):
    # Same for all layers...!
    # Need to verify these...!
    def_w = 20
    offset = -0.25
    return def_w, offset