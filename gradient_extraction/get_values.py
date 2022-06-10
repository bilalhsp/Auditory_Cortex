import torch
from gradient_extraction.hook import Hook


class GetValues:
    def __init__(self, model):
        self.model = model 

    def get_layer_output(self, spect, layer):
        self.hook = Hook(self.model, layer, backward=True)
        decoder_input_ids = torch.tensor([[1, 1]]) * self.model.config.decoder_start_token_id
        
        net_out = self.model(spect, decoder_input_ids=decoder_input_ids)

        return self.hook.output_f