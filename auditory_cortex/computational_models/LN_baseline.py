import torch.nn as nn



class LNBaseline(nn.Module):
    def __init__(self, in_channels, tmin, tmax, sfreq):
        super(LNBaseline, self).__init__()
        past_samples = int(sfreq*tmax)
        future_samples = int(sfreq*tmin) # zero for causal models...
        kernel_size = past_samples + future_samples
        padding = past_samples-1
        self.extra_samples = padding - future_samples

        self.conv = nn.Conv1d(
            in_channels=in_channels, out_channels=1,
            kernel_size=kernel_size, padding=padding, stride=1
        )


        # self.batch_norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #print("Conv forward.....")
        out = self.conv(x)[...,:-self.extra_samples]
        # out = self.batch_norm(out)
        # out = torch.clamp(out, min=0.0, max=20.0)
        out = self.activation(out)
        # out = self.dropout(out)
        #print(out.shape)
        return out
