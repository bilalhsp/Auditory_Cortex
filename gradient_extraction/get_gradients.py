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
        # add 
        # print("hf_shape:", self.hook.output_f.shape)
        beta = torch.zeros(self.hook.output_f.shape[1], 1)
        beta[beta.shape[0]//2, :] = 1
        self.betas = beta
        beta_z = torch.matmul(self.betas.T, self.hook.output_f)
        # beta_z = torch.zeros_like(self.hook.output_f)
        # beta_z[:,:,0] = self.hook.output_f[:, :, 0]
        # self.beta_z = beta_z
        loss = -beta_z[0].mean()
        # print("loss:", loss.item())
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def get_opt_input(self, spect, layer, iterations=10):
        self.spect = spect
        self.spect.requires_grad = True

        self.hook = Hook(self.model, layer, backward=True)
        self.decoder_input_ids = torch.tensor([[1, 1]]) * self.model.config.decoder_start_token_id

        self.loss_list = []

        for i in range(iterations):
            l = self.back_prop_step()
            self.loss_list.append(l)         
