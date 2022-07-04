from transformers import Speech2TextFeatureExtractor, Speech2TextModel
import torch
import numpy as np
from scipy.io.wavfile import read

from gradient_extraction.get_optimal_input import GetOptInput

# initialise and load pretrained models

dir = '/scratch/gilbreth/akamsali/Research/Makin/Auditory_Cortex/notebooks/opt_inputs/'
model = Speech2TextModel.from_pretrained("facebook/s2t-small-librispeech-asr")
feature_extractor = Speech2TextFeatureExtractor.from_pretrained("facebook/s2t-small-librispeech-asr")

for param in model.parameters():
    param.requires_grad = False

layers = ["encoder.conv.conv_layers.0","encoder.conv.conv_layers.1","encoder.layers.0.fc2",
			"encoder.layers.1.fc2","encoder.layers.2.fc2","encoder.layers.3.fc2",
			"encoder.layers.4.fc2","encoder.layers.5.fc2","encoder.layers.6.fc2",
			"encoder.layers.7.fc2","encoder.layers.8.fc2","encoder.layers.9.fc2"]


get_opt_input = GetOptInput(model, feature_extractor)

n=0
l=layers[n]

for i in range(1):
    sr, aud = read("/scratch/gilbreth/akamsali/Research/Makin/Auditory_Cortex/notebooks/audio_data/sent_" + str(i) +".wav")
    get_opt_input.get_opt_input(aud, n, l, iterations=100)

    np.save(dir + f'conv_layer_0_sent_{i}+'.npy', get_opt_input.spect_list, allow_pickle=True)
    # np.save('/scratch/gilbreth/akamsali/Research/Makin/Auditory_Cortex/notebooks/opt_inputs/conv_layer_0_sent_'+str(i)+'.npy', get_opt_input.lo_list, allow_pickle=True)