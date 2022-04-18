import torch
from gradient_extraction.hook import Hook


class GetGradient():
    def __init__(self, model, optimizer):
        self.model = model 
        self.optimizer = optimizer
        self.sig = torch.nn.Sigmoid()

    def back_prop_step(self):     
        
        self.optimizer.zero_grad()
        
        
        net_out = self.model(self.spect, decoder_input_ids=self.decoder_input_ids)
        # print(hook.output_f.shape)
        beta_z = self.sig(torch.matmul(self.hook.output_f, self.betas))
        # print(beta_z[0].shape)
        loss = (1 - self.sig(beta_z[0])).mean()
        # print("loss:", loss.item())
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def get_opt_input(self, spect, betas, layer, iterations=10):
        self.spect = spect
        self.spect.requires_grad = True

        self.betas = betas
        self.hook = Hook(self.model, layer, backward=True)
        self.decoder_input_ids = torch.tensor([[1, 1]]) * self.model.config.decoder_start_token_id

        self.loss_list = []

        for i in range(iterations):
            l = self.back_prop_step()
            self.loss_list.append(l)         
