import os
import yaml
import torch
import numpy as np
import naplib as nl
from scipy.signal import resample
from torch import nn, Tensor
from abc import ABC, abstractmethod

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from transformers import Speech2TextForConditionalGeneration, Speech2TextProcessor
from transformers import AutoProcessor, WhisperForConditionalGeneration
from transformers import AutoProcessor, AutoModelForPreTraining
from transformers import ClapModel, ClapProcessor
from transformers import AutoModel, Wav2Vec2FeatureExtractor

from transformers import Wav2Vec2ForPreTraining
from transformers import Wav2Vec2Config
import json

# local
from auditory_cortex.neural_data import NeuralMetaData
from auditory_cortex import config_dir, results_dir, aux_dir, cache_dir
from wav2letter.models import Wav2LetterRF, Wav2LetterSpect

# import GPU specific packages...
from auditory_cortex import hpc_cluster
if hpc_cluster:
    # import cupy as cp
    import fairseq
    from deepspeech_pytorch.model import DeepSpeech
    import deepspeech_pytorch.loader.data_loader as data_loader
    from deepspeech_pytorch.configs.train_config import SpectConfig


class DNNFeatureExtractor():
    def __init__(self, model_name = 'wav2letter_modified',
                saved_checkpoint=None,
                shuffled=False,
                scale_factor=None
                ):
        
        self.metadata = NeuralMetaData()
        self.model_name = model_name
        # self.layers = []
        self.layer_names = []
        self.layer_IDs = []
        self.layer_types = []
        self.receptive_fields = []
        self.features_delay = []        # features delay needed to compensate for the RFs
        self.features = {}
        self.shuffled = shuffled
        self.scale_factor = scale_factor
        # self.model = model


        #############################################################
        ###########      Need to clean this part            #########
        ###########          Starting here....!               #######
        #############################################################
            
        # read yaml config file
        config_file = os.path.join(aux_dir, f"{model_name}_config.yml")
        with open(config_file, 'r') as f:
            self.config = yaml.load(f, yaml.FullLoader)
        self.num_layers = len(self.config['layers'])
        self.use_pca = self.config['use_pca']
        if self.use_pca:
            self.pca_comps = self.config['pca_comps']

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Model on device: {self.device}")
        
        # create feature extractor as per model_name
        if model_name == 'wav2letter_modified':
            if saved_checkpoint is None:
                saved_checkpoint = self.config['saved_checkpoint']
            pretrained = self.config['pretrained']
            checkpoint = os.path.join(results_dir, 'pretrained_weights', model_name, saved_checkpoint)
            self.extractor = FeatureExtractorW2L(checkpoint, pretrained, self.device)
        elif model_name == 'wav2vec2':
            self.extractor = FeatureExtractorW2V2(self.device)
        elif model_name == 'w2v2_audioset':
            self.extractor = FeatureExtractorW2V2Audioset(self.device)
        elif model_name == 'w2v2_generic':
            self.extractor = FeatureExtractorW2V2Generic(self.device)
        elif model_name == 'spect2vec':
            self.extractor = FeatureExtractorS2V(self.device)
        elif model_name == 'MERT':
            self.extractor = FeatureExtractorMERT(self.device)
        elif model_name == 'CLAP':
            self.extractor = FeatureExtractorCLAP(self.device)
        elif model_name == 'speech2text':
            self.extractor = FeatureExtractorS2T(self.device)
        elif 'whisper' in model_name:
            if saved_checkpoint is None:
                saved_checkpoint = self.config['saved_checkpoint']
            self.extractor = FeatureExtractorWhisper(saved_checkpoint, self.device)
        elif model_name == 'deepspeech2':
            checkpoint = os.path.join(results_dir, 'pretrained_weights', model_name, self.config['saved_checkpoint'])
            self.extractor = FeatureExtractorDeepSpeech2(checkpoint, self.device)
        elif model_name == 'wav2vec':
            checkpoint = os.path.join(results_dir, 'pretrained_weights', model_name, self.config['saved_checkpoint'])
            self.extractor = FeatureExtractorW2V(checkpoint)
        elif model_name == 'wav2letter_spect':
            checkpoint = None
            pretrained = False
            self.extractor = FeatureExtractorW2LSpect(checkpoint, pretrained, self.device)
        else:
            raise NotImplementedError(f"FeatureExtractor class does not support '{model_name}'")

        for i in range(self.num_layers):
            # self.layers.append(self.config['layers'][i]['layer_name'])
            self.layer_names.append(self.config['layers'][i]['layer_name'])
            self.layer_IDs.append(self.config['layers'][i]['layer_id'])
            self.layer_types.append(self.config['layers'][i]['layer_type'])
            self.receptive_fields.append(self.config['layers'][i]['RF'])

            # self.bin_widths.append(self.config['layers'][i]['bin_width'])
            # self.offsets.append(self.config['layers'][i]['offset'])

        # Register fwd hooks for the given layers
        for layer_name in self.layer_names:
            layer = dict([*self.extractor.model.named_modules()])[layer_name]
            layer.__name__ = layer_name
            layer.register_forward_hook(self.create_hooks())

        
        # using text normalizer of whisper_tiny to compute WER..
        processor = AutoProcessor.from_pretrained('openai/whisper-tiny')
        self.label_normalizer = processor.tokenizer._normalize

        # self.shuffled = self.config['shuffle_weights']
        if self.shuffled:
            self.shuffle_weights()
            # layers = self.reset_model_parameters()
            if self.scale_factor is not None:
                self.scale_weights()
            # self.randomly_reinitialize_weights(uniform=True)

        #############################################################
        ###########      Need to clean this part            #########
        ###########          Ending here....!               #########
        #############################################################
    def get_labels_normalizer(self):
        """Normalizes or formats the labels the same way as the predictions.."""
        return self.label_normalizer
        
        # if 'whisper' in self.model_name:
        #     return self.extractor.processor.tokenizer._normalize
        # else:
        #     return None

    def create_hooks(self):
        def fn(layer, inp, output):
            if 'rnn' in layer.__name__:
               features = output[0].data
                # output = output[1][0][1].squeeze()  # reading the 2nd half of data (only backward RNNs)
            elif 'audio' in layer.__name__:
                features = output[0]
            else:
                output = output.squeeze()
                if 'conv' in layer.__name__:
                    if output.ndim > 2:
                        output = output.reshape(output.shape[0]*output.shape[1], -1)
                    output = output.transpose(0,1)
                features = output
                # Deprecated...Not needed anymore...!
                # print(inp[0].shape)
                # inp = inp[0].squeeze()
                # if 'conv' in layer.__name__:
                #     if inp.ndim > 2:
                #         inp = inp.reshape(inp.shape[0]*inp.shape[1], -1)
                #     features = inp.transpose(0,1)
                # else:
                #     features = output.squeeze()
                
            self.features[layer.__name__] = features
        return fn
    
    # def get_layer_index(self, layer_id):
    #     """Returns index for the layer_id (assigned in model specific config file),
    #     """
    #     try: 
    #         return self.layer_ids.index(layer_id)
    #     except:
    #         raise ValueError(f"Layer ID '{layer_id}' is not included in the network FE configuration.")

    def get_layer_name(self, layer_ID):
        """Returns layer_name corresponidng to layer_ID"""
        ind = self.layer_IDs.index(layer_ID)
        return self.layer_names[ind]

    def get_features(self, layer_ID):
        '''
        Use to extract features for specific layer after calling 'translate()' method 
        for given audio input.

        Args:
            layer_ID (int): layer identifier, assigned in config.

        returns:
            (dim, time) features extracted for layer at 'layer_ID'
        '''
        layer_name = self.get_layer_name(layer_ID)
        if 'rnn' in layer_name:
           return self.features[layer_name].cpu()
        #    return self.features[layer].data[:,1024:] # only using fwd features (first half of concatenatation)
        else:
            return self.features[layer_name].cpu()

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

    def reset_model_parameters(self):
        """Reset weights of all the layers of the model.
        """
        print(f"Randomly 'resetting' the network parameters...")
        layer_names = []
        named_modules = dict([*self.extractor.model.named_modules()])
        for name, layer in named_modules.items():
            if hasattr(layer, 'reset_parameters'):
                # print(f"{layer.__name__}")
                layer.reset_parameters()
                layer_names.append(name)
        return layer_names

            # if len(param.size()) > 1: #check if param is a weight tensor

    def shuffle_weights(self):
        """Shuffle weights of all the layers of the model.
        """
        print(f"Randomly 'shuffling' the network parameters...")
        for param in self.extractor.model.parameters():

            # flatten the parameter tensor and apply a random permutation...
            flattened_param = param.data.view(-1)
            shuffled_param = flattened_param[np.random.permutation(flattened_param.size(0))]

            # Reshape the shuffled_param back to original shape
            param.data = shuffled_param.view(param.size())

    def randomly_reinitialize_weights(self, uniform=False):
        """Randomly initialize weights of all the layers of the model.
        """
        print(f"Initializing 'random weights' the network parameters...")
        with torch.no_grad():
            for param in self.extractor.model.parameters():
                if uniform:
                    param.data = torch.rand_like(param)
                else:
                    param.data = torch.randn_like(param)

    def scale_weights(self):
        """Randomly initialize weights of all the layers of the model.
        """
        print(f"Scalling weights by factor: {self.scale_factor}")
        with torch.no_grad():
            for param in self.extractor.model.parameters():
                param.data = param.data*self.scale_factor

    def save_state_dist(self):
        """Saves the state_dict of the model to cache dir."""
        if self.scale_factor is None:
            weights_factor = 1 
        else:
            weights_factor = self.scale_factor
        state_dict_path = os.path.join(
            cache_dir, self.model_name, 'shuffled', f'shuffled_weights_factor_{weights_factor}.pth')
        state_dict = self.extractor.model.state_dict()
        for k,v in state_dict.items():
            state_dict[k] = v.cpu().numpy()
        torch.save(state_dict, state_dict_path)
        print(f"State_dict saved at: {state_dict_path}, with scale_factor: {weights_factor}")
                
    #############################################################
    ###########      Moved from Regression to here      #########
    ###########          Starts here....!               #########
    #############################################################

    def extract_DNN_features_for_mVocs(self):
        """
        Returns raw features for all layers of the DNN for mVoc stimuli
        (NOT TIMIT stimuli)!

        Returns:
            dict of dict: read this as features[layer_ID][sent_ID]
        """
        print(f"Extracting raw features for {self.model_name}...!")
        features = {id:{} for id in self.layer_IDs}
        # self.audio_padding_duration = 0 # incase of no padding, 
        sampling_rate = self.metadata.get_sampling_rate(mVocs=True)
        for tr_id in self.metadata.mVocTrialIds:
            audio_input = self.metadata.get_mVoc_aud(tr_id)
            
            if self.model_name == 'MERT':
                if sampling_rate != 24000:
                    n_samples = int(audio_input.size*24000/sampling_rate)
                    audio_input = resample(audio_input, n_samples)
            elif self.model_name == 'w2v2_generic' or self.model_name == 'spect2vec':
                if sampling_rate != 48000:
                    n_samples = int(audio_input.size*48000/sampling_rate)
                    audio_input = resample(audio_input, n_samples)
            elif self.model_name == 'CLAP':
                if sampling_rate != 48000:
                    n_samples = int(audio_input.size*48000/sampling_rate)
                    audio_input = resample(audio_input, n_samples)
            else:
                if sampling_rate != 16000:
                    n_samples = int(audio_input.size*16000/sampling_rate)
                    audio_input = resample(audio_input, n_samples)

            
            # needed only for Whisper...!
            if 'whisper' in self.model_name:
                bin_width = 20/1000.0   #20 ms for all layers except the very first...
                audio_duration = self.metadata.get_mVoc_dur(tr_id)
                sent_samples = int(np.ceil(round(audio_duration/bin_width, 3)))

            self.translate(audio_input, grad = False)
            for layer_ID in self.layer_IDs:
                features[layer_ID][tr_id] = self.get_features(layer_ID)
                if 'whisper' in self.model_name:
                    ## whisper networks gives features for 30s long clip,
                    ## extracting only the true initial samples...
                    layer_name = self.get_layer_name(layer_ID)
                    if layer_name == 'model.encoder.conv1':
                        # sampling rate is 100 Hz for very first layer
                        # and 50 Hz for all the other layers...
                        feature_samples = 2*sent_samples
                    else:
                        feature_samples = sent_samples
                    features[layer_ID][tr_id] = features[layer_ID][tr_id][:feature_samples]
        return features


    def extract_DNN_features(self):
        """
        Returns raw features for all layers of the DNN..!

        Returns:
            dict of dict: read this as features[layer_ID][sent_ID]
        """
        print(f"Extracting raw features for {self.model_name}...!")
        features = {id:{} for id in self.layer_IDs}
        # self.audio_padding_duration = 0 # incase of no padding, 
        sampling_rate = self.metadata.get_sampling_rate(mVocs=False)
        for sent_ID in self.metadata.sent_IDs:
            audio_input = self.metadata.stim_audio(sent_ID)
            if self.model_name == 'MERT':
                if sampling_rate != 24000:
                    n_samples = int(audio_input.size*24000/sampling_rate)
                    audio_input = resample(audio_input, n_samples)
            elif self.model_name == 'CLAP':
                if sampling_rate != 48000:
                    n_samples = int(audio_input.size*48000/sampling_rate)
                    audio_input = resample(audio_input, n_samples)
            else:
                if sampling_rate != 16000:
                    n_samples = int(audio_input.size*16000/sampling_rate)
                    audio_input = resample(audio_input, n_samples)
            
            # needed only for Whisper...!
            if 'whisper' in self.model_name:
                bin_width = 20/1000.0   #20 ms for all layers except the very first...
                sent_duration = self.metadata.stim_duration(sent_ID)
                sent_samples = int(np.ceil(round(sent_duration/bin_width, 3)))

            self.translate(audio_input, grad = False)
            for layer_ID in self.layer_IDs:
                features[layer_ID][sent_ID] = self.get_features(layer_ID)
                if 'whisper' in self.model_name:
                    ## whisper networks gives features for 30s long clip,
                    ## extracting only the true initial samples...
                    layer_name = self.get_layer_name(layer_ID)
                    if layer_name == 'model.encoder.conv1':
                        # sampling rate is 100 Hz for very first layer
                        # and 50 Hz for all the other layers...
                        feature_samples = 2*sent_samples
                    else:
                        feature_samples = sent_samples
                    features[layer_ID][sent_ID] = features[layer_ID][sent_ID][:feature_samples]
        return features
    
    def extract_features_for_audio(self, audio_input, sent_duration):
        """Extracts features for given audio input.

        Args:
            audio_input (ndarray): (num_samples,) audio stimulus as np array.
            sent_duration (float): duration of audio input in sec.

        Returns:
            Dict of features, with layer ID being the keys.

        """
        print(f"Extracting features for duration: {sent_duration} sec")
        features = {}
        # needed only for Whisper...!
        if 'whisper' in self.model_name:
            bin_width = 20/1000.0   #20 ms for all layers except the very first...
            # sent_duration = feature_extractor.metadata.stim_duration(sent_ID)
            sent_samples = int(np.ceil(round(sent_duration/bin_width, 3)))

        self.translate(audio_input, grad = False)
        for layer_ID in self.layer_IDs:
            features[layer_ID] = self.get_features(layer_ID).cpu()
            if 'whisper' in self.model_name:
                ## whisper networks gives features for 30s long clip,
                ## extracting only the true initial samples...
                layer_name = self.get_layer_name(layer_ID)
                if layer_name == 'model.encoder.conv1':
                    # sampling rate is 100 Hz for very first layer
                    # and 50 Hz for all the other layers...
                    feature_samples = 2*sent_samples
                else:
                    feature_samples = sent_samples
                # features[layer_ID] = features[layer_ID][:feature_samples]
                # features[layer_ID] = features[layer_ID][-feature_samples:]
        return features
    

    def batch_predictions(self, audio_batch):
        """Returns prediction for the batch of audio tensors."""
        # audio_batch = audio_batch.to(self.device)
        return self.extractor.batch_predictions(audio_batch, self.label_normalizer)
        
    #############################################################
    ###########      Moved from Regression to here      #########
    ###########          Ends here....!               #########
    #############################################################

class FeatureExtractorW2LSpect():
    def __init__(self, checkpoint, pretrained, device):
        
        if pretrained:		
            self.model = Wav2LetterRF.load_from_checkpoint(checkpoint)
            print(f"Loading from checkpoint: {checkpoint}")
    
        else:
            self.model = Wav2LetterSpect()
            print(f"Creating untrained network...!")
        self.processor = Speech2TextProcessor.from_pretrained("facebook/s2t-large-librispeech-asr")
        self.device = device
        self.model = self.model.to(self.device)

    def fwd_pass(self, aud):
        """
        Forward passes audio input through the model and captures 
        the features in the 'self.features' dict.

        Args:
            aud (ndarray): single 'wav' input of shape (t,) 
        
        Returns:
            input (torch.Tensor): returns the torch Tensor of the input sent passed through the model.
        """
        # input_features = self.processor(aud,padding=True, sampling_rate=16000, return_tensors="pt").input_features
        
        spect = nl.features.auditory_spectrogram(aud, 16000, frame_len=10)
        spect = resample(spect, 80, axis=1)

        input_features = torch.tensor(spect, dtype=torch.float32).unsqueeze(dim=0)
        # with torch.no_grad():
        self.model.eval()
        input_features = input_features.to(self.device).transpose(1,2)
        # print(input_features.shape)
        out = self.model(input_features)
        return out
    
    # def fwd_pass_tensor(self, aud):
    #     """
    #     Forward passes audio input through the model and captures 
    #     the features in the 'self.features' dict.

    #     Args:
    #         aud (tensor): input tensor 'wav' input of shape (1, t) 
        
    #     Returns:
    #         input (torch.Tensor): returns the torch Tensor of the input sent passed through the model.
    #     """
    #     self.model.eval()
    #     out = self.model(aud)
    #     return input
    

    # def batch_predictions(self, audio_batch, label_normalizer):
    #     """Returns prediction for the batch of audio tensors."""
    #     predictions = []
    #     with torch.no_grad():
    #         self.model.eval()
    #         # audio, _, target_lens = batch
    #         for audio in audio_batch:
    #             audio = audio.to(self.device)
    #             predictions.append(label_normalizer(self.model.decode(audio)[0]))# 
    #     return predictions


class FeatureExtractorW2L():
    def __init__(self, checkpoint, pretrained, device):
        if pretrained:		
            self.model = Wav2LetterRF.load_from_checkpoint(checkpoint)
            print(f"Loading from checkpoint: {checkpoint}")
    
        else:
            self.model = Wav2LetterRF()
            print(f"Creating untrained network...!")
        self.device = device
        self.model = self.model.to(self.device)

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
        return out
    
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
    

    def batch_predictions(self, audio_batch, label_normalizer):
        """Returns prediction for the batch of audio tensors."""
        predictions = []
        with torch.no_grad():
            self.model.eval()
            # audio, _, target_lens = batch
            for audio in audio_batch:
                audio = audio.to(self.device)
                predictions.append(label_normalizer(self.model.decode(audio)[0]))# 
        return predictions
    
class FeatureExtractorW2V2Generic():
    def __init__(self, device):
        # Replace with your model's repository and the specific revision (e.g., checkpoint)
        # cache_dir = '/scratch/gilbreth/ahmedb/cache/huggingface/models/'
        # # repo_id = 'wav2vec2-audioset-natual-sounds-v59'
        # repo_id = 'wav2vec2-48KHz-audioset-natual-sounds-v1'
        # revision = 'checkpoint-24860'
        # local_dir = os.path.join(cache_dir, repo_id, revision)

        cache_dir = '/scratch/gilbreth/ahmedb/cache/huggingface/models/old_checkpoints'
        repo_id = 'wav2vec2-48KHz-audioset-natual-sounds-v1'
        local_dir = os.path.join(cache_dir, repo_id)    
        print(f"Loading model from local repo: {local_dir}")
        # Load the model
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(local_dir)
        self.model = AutoModel.from_pretrained(local_dir, use_safetensors=True)

        # model_identifier = "bilalhsp/wav2vec2-48KHz-audioset-natual-sounds-v1"
        # model_identifier = "bilalhsp/wav2vec2-audioset-natual-sounds-v53"
        # print(f"Loading weights form model identifier: \n {model_identifier}")

        # self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
        #     model_identifier, 
        #     cache_dir=cache_dir,
        #     force_download=True,
        #     )
        # self.model = AutoModel.from_pretrained(
        #     model_identifier,
        #     cache_dir=cache_dir,
        #     force_download=True,
        #     )
        self.device = device
        self.model = self.model.to(self.device)
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
        input_values = self.processor(input, sampling_rate=48000, return_tensors="pt", padding="longest").input_values  # Batch size 1
        self.model.eval()
        # with torch.no_grad():
        input_values = input_values.to(self.device)
        logits = self.model(input_values)

        # input = torch.tensor(aud, dtype=torch.float32)#, requires_grad=True)
        # input = input.unsqueeze(dim=0)
        # input.requires_grad=True
        # self.model.eval()
        # out = self.model(input)
        return logits
    
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
        logits = self.model(aud_tensor)
        return logits

    def transcribe(self, aud):
        """Transcribes speech audio."""
        logits = self.fwd_pass(aud)
        indexes = torch.argmax(logits, dim=-1)
        return self.processor.batch_decode(indexes)
        # self.processor.decode()

    def batch_predictions(self, audio_batch, label_normalizer):
        """Returns prediction for the batch of audio tensors."""
        predictions = []
        with torch.no_grad():
            self.model.eval()
            # audio, _, target_lens = batch
            for audio in audio_batch:
                audio = audio.to(self.device)
                indexes = torch.argmax(self.model(audio).logits, dim=-1)
                predictions.append(label_normalizer(self.processor.batch_decode(indexes)[0]))# 
        return predictions

class FeatureExtractorW2V2():
    def __init__(self, device):
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        self.device = device
        self.model = self.model.to(self.device)
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
        input_values = input_values.to(self.device)
        logits = self.model(input_values).logits

        # input = torch.tensor(aud, dtype=torch.float32)#, requires_grad=True)
        # input = input.unsqueeze(dim=0)
        # input.requires_grad=True
        # self.model.eval()
        # out = self.model(input)
        return logits
    
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

    def transcribe(self, aud):
        """Transcribes speech audio."""
        logits = self.fwd_pass(aud)
        indexes = torch.argmax(logits, dim=-1)
        return self.processor.batch_decode(indexes)
        # self.processor.decode()

    def batch_predictions(self, audio_batch, label_normalizer):
        """Returns prediction for the batch of audio tensors."""
        predictions = []
        with torch.no_grad():
            self.model.eval()
            # audio, _, target_lens = batch
            for audio in audio_batch:
                audio = audio.to(self.device)
                indexes = torch.argmax(self.model(audio).logits, dim=-1)
                predictions.append(label_normalizer(self.processor.batch_decode(indexes)[0]))# 
        return predictions

class FeatureExtractorW2V2Audioset():
    def __init__(self, device):
        self.processor = AutoProcessor.from_pretrained("ALM/wav2vec2-base-audioset")
        self.model = AutoModelForPreTraining.from_pretrained("ALM/wav2vec2-base-audioset")

        self.device = device
        self.model = self.model.to(self.device)
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
        input_values = input_values.to(self.device)
        out = self.model(input_values)

        # input = torch.tensor(aud, dtype=torch.float32)#, requires_grad=True)
        # input = input.unsqueeze(dim=0)
        # input.requires_grad=True
        # self.model.eval()
        # out = self.model(input)
        return out
    
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

    def transcribe(self, aud):
        """Transcribes speech audio."""
        logits = self.fwd_pass(aud)
        indexes = torch.argmax(logits, dim=-1)
        return self.processor.batch_decode(indexes)
        # self.processor.decode()

    def batch_predictions(self, audio_batch, label_normalizer):
        """Returns prediction for the batch of audio tensors."""
        predictions = []
        with torch.no_grad():
            self.model.eval()
            # audio, _, target_lens = batch
            for audio in audio_batch:
                audio = audio.to(self.device)
                indexes = torch.argmax(self.model(audio).logits, dim=-1)
                predictions.append(label_normalizer(self.processor.batch_decode(indexes)[0]))# 
        return predictions

class FeatureExtractorMERT():
    def __init__(self, device):
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-330M",trust_remote_code=True)
        # loading our model weights
        self.model = AutoModel.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)
        # loading the corresponding preprocessor config

        self.device = device
        self.model = self.model.to(self.device)
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
        input_values = self.processor(input, sampling_rate=24000, return_tensors="pt", padding="longest").input_values  # Batch size 1
        self.model.eval()
        # with torch.no_grad():
        input_values = input_values.to(self.device)
        out = self.model(input_values)

        return out
    

class FeatureExtractorCLAP():
    def __init__(self, device):
        self.processor = ClapProcessor.from_pretrained("laion/larger_clap_general")
        # loading our model weights
        self.model = ClapModel.from_pretrained("laion/larger_clap_general")
        # loading the corresponding preprocessor config

        self.device = device
        self.model = self.model.to(self.device)
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
        input = aud.astype(np.float32)
        self.model.eval()
        input_values = self.processor(audios=input, sampling_rate=48000, return_tensors="pt")
        input_values = input_values.to(self.device)
        out = self.model.get_audio_features(**input_values)

        return out
    

class FeatureExtractorS2T():
    def __init__(self, device):
        self.model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-large-librispeech-asr")
        self.processor = Speech2TextProcessor.from_pretrained("facebook/s2t-large-librispeech-asr")
        
        self.device = device
        self.model = self.model.to(self.device)


    def fwd_pass(self, aud):
        input_features = self.processor(aud,padding=True, sampling_rate=16000, return_tensors="pt").input_features
        # with torch.no_grad():
        self.model.eval()
        input_features = input_features.to(self.device)
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
    
    def batch_predictions(self, audio_batch, label_normalizer):
        """Returns prediction for the batch of audio tensors."""
        predictions = []
        with torch.no_grad():
            self.model.eval()
            # audio, _, target_lens = batch
            for audio in audio_batch:
                input_features = self.processor(audio.squeeze(), sampling_rate=16000, return_tensors="pt").input_features
                input_features = input_features.to(self.device)
                predicted_ids = self.model.generate(input_features)
                pred = label_normalizer(self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0])
                predictions.append(pred)
                
        return predictions

    
    
class FeatureExtractorWhisper():
    def __init__(self, saved_checkpoint, device):
        self.processor = AutoProcessor.from_pretrained(saved_checkpoint)
        self.model = WhisperForConditionalGeneration.from_pretrained(saved_checkpoint)
        self.device = device
        self.model = self.model.to(self.device)
        print(f"Loaded network from {saved_checkpoint}")	

    def fwd_pass(self, aud):
        input_features = self.processor(aud, sampling_rate=16000, return_tensors="pt").input_features
        # with torch.no_grad():
        self.model.eval()
        input_features = input_features.to(self.device)
        generated_ids = self.model.generate(inputs=input_features, max_new_tokens=400)
            # transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_ids
    
    def transcribe(self, audio):
        """Transcribes speech audio"""
        predicted_ids = self.fwd_pass(audio)
        return self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
    
    def batch_predictions(self, audio_batch, label_normalizer):
        """Returns prediction for the batch of audio tensors."""
        predictions = []
        with torch.no_grad():
            self.model.eval()
            # audio, _, target_lens = batch
            for audio in audio_batch:
                input_features = self.processor(audio.squeeze(), sampling_rate=16000, return_tensors="pt").input_features
                input_features = input_features.to(self.device)
                predicted_ids = self.model.generate(input_features)
                transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                predictions.append(self.processor.tokenizer._normalize(transcription))

        return predictions
    
class FeatureExtractorDeepSpeech2():
    def __init__(self, checkpoint, device):

        audio_config = SpectConfig()
        self.parser = data_loader.AudioParser(audio_config, normalize=True)
        self.model = DeepSpeech.load_from_checkpoint(checkpoint_path=checkpoint)
        self.device = device
        self.model = self.model.to(self.device)
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
        lengths = torch.tensor([spect.shape[-1]], dtype=torch.int64, device=self.device)
        spect = spect.to(self.device)
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
    
    def batch_predictions(self, audio_batch, label_normalizer):
        """Returns prediction for the batch of audio tensors."""
        predictions = []
        with torch.no_grad():
            self.model.eval()
            # audio, _, target_lens = batch
            for audio in audio_batch:
                spect = self.get_spectrogram(audio.squeeze())
                spect = spect.unsqueeze(dim=0).unsqueeze(dim=0)
                spect = spect.to(self.device)

                # length of the spect along time
                lengths = torch.tensor([spect.shape[-1]], dtype=torch.int64,
                    device=self.device)
                
                out = self.model(spect, lengths)

                output, output_sizes, *_ = out
                decoded_output, _ = self.model.evaluation_decoder.decode(output, output_sizes)
                predictions.append(label_normalizer(decoded_output[0][0]))

        return predictions
    
class FeatureExtractorW2V():
    def __init__(self, checkpoint):

        # cp_path = os.path.join(pretrained_dir, 'wav2vec', 'wav2vec_large.pt')
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

class FeatureExtractorS2V():
    def __init__(self, device):
        config_file = '/home/ahmedb/projects/Wav2Letter/hugging_face/config/spect2vec_config.json'

        with open(config_file, "r") as F:
            config_dict = json.load(F)

        # creating an instance of Wav2Vec2Config
        wav2vec_config = Wav2Vec2Config(**config_dict)
        model = Wav2Vec2ForPreTraining(wav2vec_config)
        self.num_freqs = 80
        conv_layer = torch.nn.Conv1d(
            in_channels=self.num_freqs, out_channels=512,
            kernel_size=3, stride=2, padding=1, bias=False
            )
        model.wav2vec2.feature_extractor.conv_layers[0].conv = conv_layer

        self.model = model

        self.device = device
        self.model = self.model.to(self.device)
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
        self.model.eval()
        spect = nl.features.auditory_spectrogram(aud, 48000, frame_len=10)
        spect = resample(spect, self.num_freqs, axis=1)
        input_values = torch.tensor(spect).transpose(1, 0).unsqueeze(0)

        input_values = input_values.to(self.device)
        # wav2vec2 forward pass customized...
        extract_features = self.model.wav2vec2.feature_extractor.conv_layers[0](input_values.float())
        extract_features = self.model.wav2vec2.feature_extractor.conv_layers[1](extract_features)
        mask_time_indices = None
        attention_mask = None
        output_hidden_states = None
        return_dict = None
        output_attentions = None

        extract_features = extract_features.transpose(1, 2)
        hidden_states, extract_features = self.model.wav2vec2.feature_projection(extract_features)

        hidden_states = self.model.wav2vec2._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )

        encoder_outputs = self.model.wav2vec2.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return encoder_outputs
