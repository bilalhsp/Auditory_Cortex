import torch

class Hook():
    def __init__(self, model, layer_name, backward=False):

        layer = dict([*model.named_modules()])[layer_name]
        layer.__name__ = layer_name

        self.hook_fwd = layer.register_forward_hook(self.hook_fn_f)
        # self.hook_bwd = layer.register_backward_hook(self.hook_fn_b)

    def hook_fn_f(self, layer, input, output):
        self.input_f = input
        self.output_f = output#.requires_grad_(True)

    def hook_fn_b(self, layer, input, output):
        self.input_b = input
        self.output_b = output
    
    def close(self):
        self.hook.remove()