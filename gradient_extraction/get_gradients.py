import torch
from gradient_extraction.hook import Hook


class GetGradient:
    def __init__(self, model, feature_extractor):
        self.model = model 
        self.feature_extractor = feature_extractor

    def back_prop_step(self, unit):     
        if self.layer_ID < 2:
            chosen_feature = self.hook.output_f[0, unit, :]
        else: 
            chosen_feature = self.hook.output_f[0, :, unit]
            
        loss = -chosen_feature.mean()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def get_opt_input(self, aud, layer_ID, layer, iterations=10):
        self.layer_ID = layer_ID
        self.hook = Hook(self.model, layer, backward=True)
        decoder_input_ids = torch.tensor([[1, 1]]) * self.model.config.decoder_start_token_id

        for param in self.model.parameters():
            assert param.requires_grad == False, "Model parameters not frozen"

        self.loss_list = []
        self.spect_list = []
        self.og_spect = []

        for unit in range(10):
            self.spect = self.feature_extractor(aud, sampling_rate=16000, return_tensors="pt").input_features
            self.og_spect.append(self.spect) 
            self.spect.requires_grad = True
            self.optimizer = torch.optim.Adam([self.spect], lr=10)
            loss_list = []

            self.optimizer.zero_grad()
            for i in range(iterations):
                net_out = self.model(self.spect, decoder_input_ids=decoder_input_ids)
                l = self.back_prop_step(unit)
                loss_list.append(l)

            self.loss_list.append(loss_list)
            self.spect_list.append(self.spect)

