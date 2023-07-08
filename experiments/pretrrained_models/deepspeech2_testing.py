import os
import torch

import auditory_cortex.models as Reg
from deepspeech_pytorch.model import DeepSpeech
import deepspeech_pytorch.loader.data_loader as data_loader
from deepspeech_pytorch.configs.train_config import SpectConfig
# checkpoint_path = '/depot/jgmakin/data/auditory_cortex/pretrained_weights/deepspeech2/librispeech_pretrained_v3.ckpt'

model = 'deepspeech2'
obj = Reg.Regression(model, load_features=False)

sent = 12
aud = obj.dataset.audio(sent=sent)

audio_config = SpectConfig()
parser = data_loader.AudioParser(audio_config, normalize=True)
spect = parser.compute_spectrogram(aud)

spect = spect.unsqueeze(dim=0)
spect = spect.unsqueeze(dim=0)

lengths = torch.tensor([spect.shape[-1]], dtype=torch.int64)

checkpoint_path = '/depot/jgmakin/data/auditory_cortex/pretrained_weights/deepspeech2/librispeech_pretrained_v3.ckpt'
model = DeepSpeech.load_from_checkpoint(checkpoint_path=checkpoint_path)

print("About to forward pass thru the network...")

out = model(spect, lengths)

print("Done with forward pass thru the network")

print("Do nothing, just print this...!")