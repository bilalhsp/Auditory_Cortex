import os
import yaml
import torch
import numpy as np
from torch import nn, Tensor
from abc import ABC, abstractmethod

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from transformers import Speech2TextForConditionalGeneration, Speech2TextProcessor
from transformers import AutoProcessor, WhisperForConditionalGeneration

# local
from auditory_cortex import config_dir, results_dir, aux_dir
from wav2letter.models import Wav2LetterRF

# import GPU specific packages...
from auditory_cortex import hpc_cluster
if hpc_cluster:
    import cupy as cp
    import fairseq
    from deepspeech_pytorch.model import DeepSpeech
    import deepspeech_pytorch.loader.data_loader as data_loader
    from deepspeech_pytorch.configs.train_config import SpectConfig


class FeatureExtractor():
    def __init__(self, model_name = 'wave2letter_modified'):
        
        self.layers = []
        self.layer_ids = []
        self.layer_types = []
        self.receptive_fields = []
        self.features_delay = []        # features delay needed to compensate for the RFs
        self.features = {}
        # self.model = model
            
        # read yaml config file
        config_file = os.path.join(aux_dir, f"{model_name}_config.yml")
        with open(config_file, 'r') as f:
            self.config = yaml.load(f, yaml.FullLoader)
        self.num_layers = len(self.config['layers'])
        self.use_pca = self.config['use_pca']
        if self.use_pca:
            self.pca_comps = self.config['pca_comps']
        
        # create feature extractor as per model_name
        if model_name == 'wave2letter_modified':
            pretrained = self.config['pretrained']
            checkpoint = os.path.join(results_dir, 'pretrained_weights', model_name, self.config['saved_checkpoint'])
            self.extractor = FeatureExtractorW2L(checkpoint, pretrained)
        elif model_name == 'wave2vec2':
            self.extractor = FeatureExtractorW2V2()
        elif model_name == 'speech2text':
            self.extractor = FeatureExtractorS2T()
        elif model_name == 'whisper':
            self.extractor = FeatureExtractorWhisper()
        elif model_name == 'deepspeech2':
            checkpoint = os.path.join(results_dir, 'pretrained_weights', model_name, self.config['saved_checkpoint'])
            self.extractor = FeatureExtractorDeepSpeech2(checkpoint)
        elif model_name == 'wave2vec':
            checkpoint = os.path.join(results_dir, 'pretrained_weights', model_name, self.config['saved_checkpoint'])
            self.extractor = FeatureExtractorW2V(checkpoint)
        else:
            raise NotImplementedError(f"FeatureExtractor class does not support '{model_name}'")

        for i in range(self.num_layers):
            self.layers.append(self.config['layers'][i]['layer_name'])
            self.layer_ids.append(self.config['layers'][i]['layer_id'])
            self.layer_types.append(self.config['layers'][i]['layer_type'])
            self.receptive_fields.append(self.config['layers'][i]['RF'])

            # self.bin_widths.append(self.config['layers'][i]['bin_width'])
            # self.offsets.append(self.config['layers'][i]['offset'])

        # Register fwd hooks for the given layers
        for layer_name in self.layers:
            layer = dict([*self.extractor.model.named_modules()])[layer_name]
            layer.__name__ = layer_name
            layer.register_forward_hook(self.create_hooks())
            
    def create_hooks(self):
        def fn(layer, _, output):
            if 'rnn' in layer.__name__:
               output = output[0].data
                # output = output[1][0][1].squeeze()  # reading the 2nd half of data (only backward RNNs)
            else:
                output = output.squeeze()
                if 'conv' in layer.__name__:
                    if output.ndim > 2:
                        output = output.reshape(output.shape[0]*output.shape[1], -1)
                    output = output.transpose(0,1)
            self.features[layer.__name__] = output
        return fn
    
    def get_layer_index(self, layer_id):
        """Returns index for the layer_id (assigned in model specific config file),
        """
        try: 
            return self.layer_ids.index(layer_id)
        except:
            raise ValueError(f"Layer ID '{layer_id}' is not included in the network FE configuration.")


    def get_features(self, layer_index):
        '''
        Use to extract features for specific layer after calling 'translate()' method 
        for given audio input.

        Args:
        
            layer_index (int): layer index in the range [0, Total_Layers)

        returns:
            (dim, time) features extracted for layer at 'layer_index' location 
        '''
        layer = self.layers[layer_index]
        if 'rnn' in layer:
           return self.features[layer]
        #    return self.features[layer].data[:,1024:] # only using fwd features (first half of concatenatation)
        else:
            return self.features[layer]

    def def_bin_width(self, layer):
        def_w = self.bin_widths[layer]
        offset = self.offsets[layer]
        return def_w, offset

    def translate(self, aud, grad=False):
        if grad:
            input = self.extractor.fwd_pass_tensor(aud)
        else:
            with torch.no_grad():
                input = self.extractor.fwd_pass(aud)
        return input


class FeatureExtractorW2L():
    def __init__(self, checkpoint, pretrained):
        if pretrained:		
            self.model = Wav2LetterRF.load_from_checkpoint(checkpoint)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Loading from checkpoint: {checkpoint}")
    
        else:
            self.model = Wav2LetterRF()
            print(f"Creating untrained network...!")

    def fwd_pass(self, aud):
        """
        Forward passes audio input through the model and captures 
        the features in the 'self.features' dict.

        Args:
            aud (ndarray): single 'wav' input of shape (t,) 
        
        Returns:
            input (torch.Tensor): returns the torch Tensor of the input sent passed through the model.
        """
        
        input = torch.tensor(aud, dtype=torch.float32, device=self.device)#, requires_grad=True)
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # input = input.to(device)


        input = input.unsqueeze(dim=0)
        # input.requires_grad=True
        self.model.eval()
        out = self.model(input)
        return input
    
    def fwd_pass_tensor(self, aud):
        """
        Forward passes audio input through the model and captures 
        the features in the 'self.features' dict.

        Args:
            aud (tensor): input tensor 'wav' input of shape (1, t) 
        
        Returns:
            input (torch.Tensor): returns the torch Tensor of the input sent passed through the model.
        """
        self.model.eval()
        out = self.model(aud)
        return input

class FeatureExtractorW2V2():
    def __init__(self):
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        # self.model = model 
        ########################## NEED TO MAKE THIS CONSISTENT>>>##################
    def fwd_pass(self, aud):
        """
        Forward passes audio input through the model and captures 
        the features in the 'self.features' dict.

        Args:
            aud (ndarray): single 'wav' input of shape (t,) 
        
        Returns:
            input (torch.Tensor): returns the torch Tensor of the input sent passed through the model.
        """
        input = aud.astype(np.float64)
        input_values = self.processor(input, sampling_rate=16000, return_tensors="pt", padding="longest").input_values  # Batch size 1
        self.model.eval()
        # with torch.no_grad():
        logits = self.model(input_values).logits

        # input = torch.tensor(aud, dtype=torch.float32)#, requires_grad=True)
        # input = input.unsqueeze(dim=0)
        # input.requires_grad=True
        # self.model.eval()
        # out = self.model(input)
        return input
    
    def fwd_pass_tensor(self, aud_tensor):
        """
        Forward passes audio input through the model and captures 
        the features in the 'self.features' dict.

        Args:
            aud (tensor): input tensor 'wav' input of shape (1, t) 
        
        Returns:
            input (torch.Tensor): returns the torch Tensor of the input sent passed through the model.
        """
        self.model.eval()
        logits = self.model(aud_tensor).logits
        return logits

class FeatureExtractorS2T():
    def __init__(self):
        self.model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-large-librispeech-asr")
        self.processor = Speech2TextProcessor.from_pretrained("facebook/s2t-large-librispeech-asr")
                
    def fwd_pass(self, aud):
        input_features = self.processor(aud,padding=True, sampling_rate=16000, return_tensors="pt").input_features
        # with torch.no_grad():
        self.model.eval()
        generated_ids = self.model.generate(input_features, max_new_tokens=200)
        return generated_ids

    def fwd_pass_tensor(self, aud_spect):
        """
        Forward passes spectrogram of audio input through the model and captures 
        the features in the 'self.features' dict.

        Args:
            aud_spect (tensor): spectrogram of input tensor, shape (1, t, 80) 
        
        Returns:
            input (torch.Tensor): returns the torch Tensor of the input sent passed through the model.
        """
        self.model.eval()
        # feeding decoder the start token...!
        decoder_input_ids = torch.tensor([[1, 1]]) * self.model.config.decoder_start_token_id
        out = self.model(aud_spect, decoder_input_ids=decoder_input_ids)
        return out

    
    
class FeatureExtractorWhisper():
    def __init__(self):
        self.processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
        self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
				
    def fwd_pass(self, aud):
        input_features = self.processor(aud, sampling_rate=16000, return_tensors="pt").input_features
        # with torch.no_grad():
        self.model.eval()
        generated_ids = self.model.generate(inputs=input_features, max_new_tokens=400)
            # transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_ids
    
class FeatureExtractorDeepSpeech2():
    def __init__(self, checkpoint):

        audio_config = SpectConfig()
        self.parser = data_loader.AudioParser(audio_config, normalize=True)
        self.model = DeepSpeech.load_from_checkpoint(checkpoint_path=checkpoint)
        # self.processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
        # self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
	
    def get_spectrogram(self, aud):
        """Gives spectrogram of audio input."""
        if torch.is_tensor(aud):
            aud = aud.cpu().numpy()
        return self.parser.compute_spectrogram(aud)

    def fwd_pass(self, aud):
        
        # spect = self.parser.compute_spectrogram(aud)
        spect = self.get_spectrogram(aud)
        spect = spect.unsqueeze(dim=0).unsqueeze(dim=0)

        # length of the spect along time
        lengths = torch.tensor([spect.shape[-1]], dtype=torch.int64)
        out = self.model(spect, lengths)
        return out
    
    def fwd_pass_tensor(self, aud_spect):
        """
        Forward passes spectrogram of audio input through the model and captures 
        the features in the 'self.features' dict.

        Args:
            aud_spect (tensor): spectrogram of input tensor, shape (1, t, 80) 
        
        Returns:
            output (torch.Tensor): returns the torch Tensor of the input sent passed through the model.
        """
        aud_spect = aud_spect.unsqueeze(dim=0)
        lengths = torch.tensor([aud_spect.shape[-1]], dtype=torch.int64)
        self.model.eval()
        out = self.model(aud_spect, lengths)
        return out
    
class FeatureExtractorW2V():
    def __init__(self, checkpoint):

        # cp_path = os.path.join(pretrained_dir, 'wave2vec', 'wav2vec_large.pt')
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([checkpoint])
        self.model = model[0]
				
    def fwd_pass(self, aud):
        aud_tensor = torch.tensor(aud, dtype=torch.float32)
        aud_tensor = aud_tensor.unsqueeze(dim=0)
        self.model.eval()
        with torch.no_grad():
            z = self.model.feature_extractor(aud_tensor)
            c = self.model.feature_aggregator(z)
        return c

    def fwd_pass_tensor(self, aud_tensor):
        """
        Forward passes audio input through the model and captures 
        the features in the 'self.features' dict.

        Args:
            aud (tensor): input tensor 'wav' input of shape (1, t) 
        
        Returns:
            input (torch.Tensor): returns the torch Tensor of the input sent passed through the model.
        """
        self.model.eval()
        z = self.model.feature_extractor(aud_tensor)
        c = self.model.feature_aggregator(z)
       
        # # gives us translated sentence
        # predicted_ids = torch.argmax(logits, dim=-1)
        # # transcribe speech
        # transcription = self.processor.batch_decode(predicted_ids)
        return c

