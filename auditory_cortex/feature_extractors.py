import torch
from torch import nn, Tensor

class FeatureExtractorW2L(nn.Module):
  def __init__(self, model: nn.Module):
    super(FeatureExtractorW2L, self).__init__()
    self.model = model
    self.layers = ["conv1","conv2","conv3","conv4","conv5","conv6","conv7","conv8","conv9","conv10",
                  "conv11","conv12","conv13","conv14","conv15","conv16","conv17","conv18",]
    self.features = {}
    
    # Register a hook for the given layer
    for layer_name in self.layers:
      layer = dict([*self.model.named_modules()])[layer_name].conv
      layer.__name__ = layer_name
      layer.register_forward_hook(self.create_hooks())
          
  def create_hooks(self):
    def fn(layer, _, output):
      self.features[layer.__name__] = output.squeeze()
      # print("here:", output)
    return fn
  def forward(self, input):
    # input: 'ndarray' of shape (t,) 
    #(single audio input in the form of ndarray)
    input = torch.tensor(input, dtype=torch.float32)
    input = input.unsqueeze(dim=0)
    return self.model(input)
  def get_features(self, index):
    '''
    arguments:
    - index: (int) layer index in the range [0, 18)
    returns:
    - (dim, time) features extracted for layer at 'index' location 
    '''
    return self.features[self.layers[index]]

class Feature_Extractor_S2T(nn.Module):
  def __init__(self, model: nn.Module, layers):
    super(Feature_Extractor_S2T, self).__init__()
    self.model = model
    self.features = {}
    
    # Register a hook for the given layer
    for layer_name in layers:
      layer = dict([*self.model.named_modules()])[layer_name]
      layer.__name__ = layer_name
      layer.register_forward_hook(self.create_hooks())
          
  def create_hooks(self):
    def fn(layer, _, output):
      self.features[layer.__name__] = output.squeeze()
      # print("here:", output)
    return fn
    
  def forward(self, input):
    return self.model.generate(input)

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