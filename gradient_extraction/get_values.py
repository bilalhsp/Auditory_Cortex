import torch
from gradient_extraction.hook import Hook


class GetValues:
    def __init__(self, model, feature_extractor):
        self.model = model 
        self.feature_extractor = feature_extractor

    def get_layer_output(self, aud, layer):
        self.hook = Hook(self.model, layer, backward=True)
        decoder_input_ids = torch.tensor([[1, 1]]) * self.model.config.decoder_start_token_id

        self.spect = self.feature_extractor(aud, sampling_rate=16000, return_tensors="pt").input_features
        
        net_out = self.model(self.spect, decoder_input_ids=decoder_input_ids)
        # print(f"shape: {self.hook.output_f.shape}")
        # chosen_feature = self.hook.output_f[0, unit, :]

                