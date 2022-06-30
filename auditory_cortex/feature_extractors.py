import os
import yaml
import torch
from torch import nn, Tensor

class FeatureExtractorW2L(nn.Module):
  def __init__(self, model: nn.Module):
    super(FeatureExtractorW2L, self).__init__()
    self.model = model
    config_file = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      'conf',
      f'{model.model_name}.yaml'
      )
    with open(config_file, 'r') as f:
      self.config = yaml.load(f, yaml.FullLoader)
    self.num_layers = len(self.config)
    self.layers = []
    for i in range(self.num_layers):
      self.layers.append(self.config[i]['layer_name'])

    self.bin_widths = []
    self.offsets = []
    for i in range(self.num_layers):
      self.bin_widths.append(self.config[i]['bin_width'])
      self.offsets.append(0)

    # self.layers = ["conv1","conv2","conv3","conv4","conv5","conv6","conv7","conv8","conv9",
    #               "conv10","conv11","conv12","conv13","conv14","conv15",
    #               #"conv16","conv17","conv18",
    #               ]
    # 
    self.features = {}
    # Register a hook for the given layer
    for layer_name in self.layers:
      layer = dict([*self.model.named_modules()])[layer_name]
      layer.__name__ = layer_name
      layer.register_forward_hook(self.create_hooks())
          
  def create_hooks(self):
    def fn(layer, _, output):
      output = output.squeeze()
      self.features[layer.__name__] = output.transpose(0,1)
      # print("here:", output)
    return fn

  @torch.no_grad()
  def forward(self, input):
    # input: 'ndarray' of shape (t,) 
    #(single audio input in the form of ndarray)
    input = torch.tensor(input, dtype=torch.float32)
    input = input.unsqueeze(dim=0)
    self.model.eval()
    return self.model(input)

  def translate(self, aud, fs = 16000):
    inputs_features = aud
    generated_ids = self.forward(inputs_features)

  def get_features(self, index):
    '''
    arguments:
    - index: (int) layer index in the range [0, 18)
    returns:
    - (dim, time) features extracted for layer at 'index' location 
    '''
    return self.features[self.layers[index]]

  def def_bin_width(self, layer):
    # Same for all layers incase of w2l (puzzlelib version)...!
    # def_w = 20
    # offset = 1.014
    def_w = self.bin_widths[layer]
    offset = self.offsets[layer]
    return def_w, offset


class Feature_Extractor_S2T(nn.Module):
  def __init__(self, model, processor, layers):
    super(Feature_Extractor_S2T, self).__init__()
    self.model = model
    self.processor = processor
    self.features = {}
    print("S2T feature extractor created...!")
    
    # Register a hook for the given layer
    for layer_name in layers:
      layer = dict([*self.model.named_modules()])[layer_name]
      layer.__name__ = layer_name
      layer.register_forward_hook(self.create_hooks())
          
  def create_hooks(self):
    def fn(layer, _, output):
      output = output.squeeze()
      if layer.__name__ in ["model.encoder.conv.conv_layers.0","model.encoder.conv.conv_layers.1"]:
            output = output.transpose(0,1)
      # print(output.shape)
      self.features[layer.__name__] = output
    return fn
    
  def forward(self, input):
    return self.model.generate(input)
  
  def translate(self, aud, fs = 16000):
    inputs_features = self.processor(aud,padding=True, sampling_rate=fs, return_tensors="pt").input_features
    generated_ids = self.forward(inputs_features)
  
  def def_bin_width(self, layer):
    if layer <1:
      def_w = 20
      offset = -0.25 
    else:
      def_w = 40
      offset = 0.39
    return def_w, offset

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