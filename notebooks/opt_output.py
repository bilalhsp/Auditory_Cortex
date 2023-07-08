from transformers import Speech2TextFeatureExtractor, Speech2TextModel
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read, write


from gradient_extraction.get_gradients import GetGradient


# initialise and load pretrained models
model = Speech2TextModel.from_pretrained("facebook/s2t-small-librispeech-asr")
feature_extractor = Speech2TextFeatureExtractor.from_pretrained("facebook/s2t-small-librispeech-asr")
# reg = Regression(dir, subject)

for param in model.parameters():
    param.requires_grad = False

layers = ["encoder.conv.conv_layers.0","encoder.conv.conv_layers.1","encoder.layers.0.fc2",
			"encoder.layers.1.fc2","encoder.layers.2.fc2","encoder.layers.3.fc2",
			"encoder.layers.4.fc2","encoder.layers.5.fc2","encoder.layers.6.fc2",
			"encoder.layers.7.fc2","encoder.layers.8.fc2","encoder.layers.9.fc2"]


get_opt_input = GetGradient(model, feature_extractor)

n = 1
l = layers[1]

for i in range(1):
    sr, aud = read("/scratch/gilbreth/akamsali/Research/Makin/Auditory_Cortex/notebooks/audio_data/sent_" + str(i) +".wav")
# for n, l in enumerate(layers[:1]):
    # print(n, l)
    get_opt_input.get_opt_input(aud, n, l, iterations=100)
    
    # spect_lists.append(get_opt_input.spect_list)
    # loss_lists.append(get_opt_input.loss_lists)
    np.save(f"/scratch/gilbreth/akamsali/Research/Makin/Auditory_Cortex/notebooks/opt_inputs/conv_layer_{n}_sent_{i}.npy", get_opt_input.spect_list, allow_pickle=True)

