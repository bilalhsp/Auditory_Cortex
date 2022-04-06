from torch import nn, Tensor

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