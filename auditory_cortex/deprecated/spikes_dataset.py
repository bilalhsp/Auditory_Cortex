import torch
import torch.nn as nn 
import torch.nn.functional as F
import torchaudio
import numpy as np
from scipy.signal import resample

import naplib as nl
from auditory_cortex.deprecated.dataloader import DataLoader


class SpikesData(torch.utils.data.Dataset):
    def __init__(self, validation=False, test=False, pretrained=False, 
                 pretrained_features=None, neural_area='all'
        ):
        super(SpikesData, self).__init__()
        self.pretrained = pretrained
        self.neural_dataloader = DataLoader()
        self.spikes = self.neural_dataloader.get_all_neural_spikes(
            bin_width=20, threshold=0.061, area=neural_area
        )
        self.feat_channels = 128

        self.spike_channels = next(iter(self.spikes.values())).shape[-1]
        if self.pretrained:
            self.pretrained_features = pretrained_features
            self.feat_channels = next(iter(self.pretrained_features.values())).shape[1]
        
        self.fs = self.neural_dataloader.metadata.get_sampling_rate()
        self.test_IDs = self.neural_dataloader.test_sent_IDs
        if test:
            self.sent_IDs = self.test_IDs
        else:
            self.sent_IDs = self.neural_dataloader.sent_IDs
            self.sent_IDs = self.sent_IDs[
                                    np.isin(self.sent_IDs, self.test_IDs, invert=True)
                                ]
            if validation:
                self.sent_IDs = self.sent_IDs[:10]
            else:
                self.sent_IDs = self.sent_IDs[10:]

                
    def __len__(self):
        return len(self.sent_IDs)
    
    def __getitem__(self, index):
        sent_ID = self.sent_IDs[index]

        spikes = torch.tensor(self.spikes[sent_ID], dtype=torch.float32)
        
        if self.pretrained:
            feats = self.pretrained_features[sent_ID]
            feats = torch.tensor(feats, dtype=torch.float32)
            return feats, spikes
        
        else:
            audio = self.neural_dataloader.metadata.stim_audio(sent_ID)

            spect = nl.features.auditory_spectrogram(audio, self.fs)
            spect = resample(spect, spikes.shape[0], axis=0)
            spect = torch.tensor(spect, dtype=torch.float32)
            return spect, spikes
    
    @staticmethod
    def collate_fn(data_items):
        audio_list = []
        spikes_list = []

        for audio, spikes in data_items:
            audio_list.append(audio)
            spikes_list.append(spikes)
        
        audio_stim = torch.concat(audio_list, dim=0)
        neural_spikes = torch.concat(spikes_list, dim=0)

        return torch.unsqueeze(audio_stim, dim=0), torch.unsqueeze(neural_spikes, dim=0)
    

def get_dataloaders(batch_size, pretrained=False, pretrained_features=None,
                    neural_area = 'all'):
    
    if pretrained:
        train_data = SpikesData(
            pretrained=pretrained, pretrained_features=pretrained_features,
            neural_area=neural_area 
        )
        test_data = SpikesData(
            validation=True, pretrained=pretrained,
            pretrained_features=pretrained_features, neural_area=neural_area 
        )     

    else:
        train_data = SpikesData(neural_area=neural_area)
        test_data = SpikesData(validation=True, neural_area=neural_area)

    channels = train_data.feat_channels
    spike_channels = train_data.spike_channels
    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size = batch_size, shuffle=True, collate_fn=SpikesData.collate_fn
    )
    val_dataloader = torch.utils.data.DataLoader(
        test_data, batch_size = len(test_data), shuffle=True, collate_fn=SpikesData.collate_fn
    )
    return train_dataloader, val_dataloader, channels, spike_channels


class Network(nn.Module):

    def __init__(
            self, num_layers, num_units, kernel_sizes,
            in_channels, out_channels
        ):
        super(Network, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(nn.BatchNorm1d(in_channels))
            layers.append(nn.Conv1d(in_channels, out_channels=num_units[i],
                                kernel_size=kernel_sizes[i], padding='same'))
            layers.append(nn.ReLU())
            in_channels = num_units[i]

        self.conv_layers = nn.Sequential(*layers)
        self.linear = nn.Linear(in_channels, out_channels)
        self.out_channels = out_channels

    def forward(self, x):
        # (1, time, channels)
        x = x.transpose(1,2)
        # (1, channels, time)
        x = self.conv_layers(x)     

        # x = x.view(-1, x.shape[0]) 
        x = x.transpose(1,2)  
        # (1, time, channels)
        x = self.linear(x)
        out = F.relu(x)
        return out


def correlation_score(Y_hat, Y):

    Y = Y.squeeze()
    Y_hat = Y_hat.squeeze()

    N = Y.shape[1]

    Y_std = torch.std(Y, dim=0)
    Y_hat_std = torch.std(Y_hat, dim=0)
    inners = torch.matmul((Y - torch.mean(Y, dim=0)).T, (Y_hat - torch.mean(Y_hat, dim=0)))/N
    corr = torch.diag(inners)/torch.sqrt(Y_std*Y_hat_std)
    corr = torch.clip(corr, min=0, max=1)

    return torch.mean(corr)





def training_epoch(
        model, optimizer, train_dataloader, device
    ):
    model.train()
    training_loss = 0
    for spect, spikes in train_dataloader:
        spect = spect.to(device)
        spikes = spikes.to(device)
        predicted_spikes = model(spect)

        optimizer.zero_grad()
        loss = F.mse_loss(predicted_spikes, spikes) - correlation_score(predicted_spikes, spikes)
        loss.backward()
        optimizer.step()

        training_loss += loss.item()
    return training_loss


def evaluation_epoch(
        model, test_dataloader, device
    ):
    model.eval()
    with torch.no_grad():
        # Test loss (Needs to be minimized..)
        test_score = 0
        for spect, spikes in test_dataloader:
            spect = spect.to(device)
            spikes = spikes.to(device)
            predicted_spikes = model(spect)

            score = correlation_score(predicted_spikes, spikes)
            # loss = F.mse_loss(predicted_spikes, spikes)
            test_score += score.item()
    return test_score



# def ll
