import os
import torch
import logging
import numpy as np
import torch.nn as nn
from scipy.signal import resample
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from transformers import AutoProcessor, WhisperForConditionalGeneration, AutoModelForPreTraining
from transformers import Speech2TextForConditionalGeneration, Speech2TextProcessor
from transformers import AutoModel, Wav2Vec2FeatureExtractor
from transformers import ClapModel, ClapProcessor

# import GPU specific packages...
from deepspeech_pytorch.model import DeepSpeech
import deepspeech_pytorch.loader.data_loader as data_loader
from deepspeech_pytorch.configs.train_config import SpectConfig
import importlib
from wav2letter.models import Wav2LetterRF, Wav2LetterSpect

# local imports
from auditory_cortex import utils
from .base_feature_extractor import BaseFeatureExtractor, register_feature_extractor
from auditory_cortex import results_dir, cache_dir

import logging
logger = logging.getLogger(__name__)

HF_CACHE_DIR = cache_dir / 'hf_cache'

@register_feature_extractor('wav2letter_modified')
class Wav2LetterModified(BaseFeatureExtractor):
    def __init__(self, shuffled=False):
        self.model_name = 'wav2letter_modified'
        config = utils.load_dnn_config(model_name=self.model_name)
        saved_checkpoint = config['saved_checkpoint']
        checkpoint = os.path.join(results_dir, 'pretrained_weights', self.model_name, saved_checkpoint)
        pretrained = config['pretrained']
        if pretrained:		
            model = Wav2LetterRF.load_from_checkpoint(checkpoint)
            logger.info(f"Loading from checkpoint: {checkpoint}")
        else:
            model = Wav2LetterRF()
            logger.info(f"Creating untrained network...!")
        super().__init__(model, config, shuffled=shuffled, sampling_rate=config['sampling_rate'])

    def fwd_pass(self, aud):
        """
        Forward passes audio input through the model and captures 
        the features in the 'self.features' dict.

        Args:
            aud (ndarray): single 'wav' input of shape (t,) 
        
        Returns:
            input (torch.Tensor): returns the torch Tensor of the input sent passed through the model.
        """
        if not isinstance(aud, torch.Tensor):
            aud = torch.tensor(aud, dtype=torch.float32, device=self.device)#, requires_grad=True)
            aud = aud.unsqueeze(dim=0)
        self.model.eval()
        with torch.no_grad():
            out = self.model(aud)
        return out
    
    def batch_predictions(self, audio_batch, label_normalizer):
        """Returns prediction for the batch of audio tensors."""
        # method not tested yet
        predictions = []
        with torch.no_grad():
            for audio in audio_batch:
                audio = audio.to(self.device)
                predictions.append(label_normalizer(self.model.decode(audio)[0]))# 
        return predictions
    
@register_feature_extractor('wav2vec2')
class Wav2Vec2(BaseFeatureExtractor):
    def __init__(self, shuffled=False):
        self.model_name = 'wav2vec2'

        config = utils.load_dnn_config(model_name=self.model_name)
        repo_name = config['repo_name']
        model = Wav2Vec2ForCTC.from_pretrained(repo_name, cache_dir=HF_CACHE_DIR)
        super().__init__(model, config, shuffled=shuffled, sampling_rate=config['sampling_rate'])

        self.processor = Wav2Vec2Processor.from_pretrained(repo_name, cache_dir=HF_CACHE_DIR)


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
        with torch.no_grad():
            input_values = input_values.to(self.device)
            logits = self.model(input_values).logits

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



@register_feature_extractor('speech2text')
class Speech2Text(BaseFeatureExtractor):
    def __init__(self, shuffled=False):
        self.model_name = 'speech2text'
        config = utils.load_dnn_config(model_name=self.model_name)
        repo_name = config['repo_name']
        model = Speech2TextForConditionalGeneration.from_pretrained(repo_name, cache_dir=HF_CACHE_DIR)
        super().__init__(model, config, shuffled=shuffled, sampling_rate=config['sampling_rate'])

        self.processor = Speech2TextProcessor.from_pretrained(repo_name, cache_dir=HF_CACHE_DIR)


    def fwd_pass(self, aud):
        input_features = self.processor(aud,padding=True, sampling_rate=16000, return_tensors="pt").input_features
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
    
    def process_input(self, aud):
        """Preprocesses the input audio."""
        aud = aud.squeeze()
        spect = self.processor(
            aud, padding=True, sampling_rate=16000, return_tensors="np"
            ).input_features[0]
        return spect
    
    
class FeatureExtractorWhisper(BaseFeatureExtractor):
    def __init__(self, model_name, shuffled=False):
        self.model_name = model_name
        config = utils.load_dnn_config(model_name=self.model_name)
        repo_name = config['repo_name']    
        model = WhisperForConditionalGeneration.from_pretrained(repo_name, cache_dir=HF_CACHE_DIR)

        super().__init__(model, config, shuffled=shuffled, sampling_rate=config['sampling_rate'])
        self.processor = AutoProcessor.from_pretrained(repo_name, cache_dir=HF_CACHE_DIR)
        
    def process_input(self, aud):
        """Preprocesses the input audio."""
        aud = aud.squeeze()
        spect = self.processor(
            aud, sampling_rate=16000, return_tensors="np", padding="longest"
            ).input_features[0]
        return spect.transpose(1, 0)

    def fwd_pass(self, aud):
        input_features = self.processor(aud, sampling_rate=16000, return_tensors="pt").input_features
        # with torch.no_grad():
        self.model.eval()
        input_features = input_features.to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(inputs=input_features, max_new_tokens=400)
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
            for audio in audio_batch:
                input_features = self.processor(audio.squeeze(), sampling_rate=16000, return_tensors="pt").input_features
                input_features = input_features.to(self.device)
                predicted_ids = self.model.generate(input_features)
                transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                predictions.append(self.processor.tokenizer._normalize(transcription))
        return predictions
    

@register_feature_extractor('whisper_tiny')
class WhisperTiny(FeatureExtractorWhisper):
    def __init__(self, shuffled=False):
        model_name = 'whisper_tiny'
        super().__init__(model_name, shuffled=shuffled)

@register_feature_extractor('whisper_base')
class WhisperBase(FeatureExtractorWhisper):
    def __init__(self, shuffled=False):
        model_name = 'whisper_base'
        super().__init__(model_name, shuffled=shuffled)

@register_feature_extractor('deepspeech2')
class DeepSpeech2(BaseFeatureExtractor):
    def __init__(self, shuffled=False):
        self.model_name = 'deepspeech2'
        config = utils.load_dnn_config(model_name=self.model_name)
        checkpoint = os.path.join(results_dir, 'pretrained_weights', self.model_name, config['saved_checkpoint'])
        model = DeepSpeech.load_from_checkpoint(checkpoint_path=checkpoint)

        super().__init__(model, config, shuffled=shuffled, sampling_rate=config['sampling_rate'])
        audio_config = SpectConfig()
        self.parser = data_loader.AudioParser(audio_config, normalize=True)

    
    def process_input(self, aud):
        """Preprocesses the input audio and returns the spectrogram.
        Spectrogram is expected to have features of shape (t, 80).
        
        Args:
            aud (ndarray): single 'wav' input of shape (t,)
        Returns:
            spect (ndarray): spectrogram of the input audio (t, 80)
        
        """
        aud = aud.squeeze()
        spect = self.get_spectrogram(aud).cpu().numpy().transpose(1, 0)
        return spect


    def get_spectrogram(self, aud):
        """Gives spectrogram of audio input."""
        if torch.is_tensor(aud):
            aud = aud.cpu().numpy()
        return self.parser.compute_spectrogram(aud)

    def fwd_pass(self, aud):

        # test if input is 1 dimensional (audio signal)
        if aud.ndim == 1:
            spect = self.get_spectrogram(aud)
        spect = spect.unsqueeze(dim=0).unsqueeze(dim=0)

        # length of the spect along time
        lengths = torch.tensor([spect.shape[-1]], dtype=torch.int64, device=self.device)
        spect = spect.to(self.device)
        self.model.eval()
        with torch.no_grad():
            out = self.model(spect, lengths)
        return out
    
    
    def batch_predictions(self, audio_batch, label_normalizer):
        """Returns prediction for the batch of audio tensors."""
        # method not tested yet
        predictions = []
        with torch.no_grad():
            self.model.eval()
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



@register_feature_extractor('w2v2_audioset')
class W2V2Audioset(BaseFeatureExtractor):
    def __init__(self, shuffled=False):
        self.model_name = 'w2v2_audioset'
        config = utils.load_dnn_config(model_name=self.model_name)
        repo_name = config['repo_name']
        model = Wav2Vec2ForCTC.from_pretrained(repo_name, cache_dir=HF_CACHE_DIR)
        super().__init__(model, config, shuffled=shuffled, sampling_rate=config['sampling_rate'])
        self.processor = AutoProcessor.from_pretrained(repo_name, cache_dir=HF_CACHE_DIR)

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
        with torch.no_grad():
            input_values = input_values.to(self.device)
            out = self.model(input_values)
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

    def batch_predictions(self, audio_batch, label_normalizer):
        """Returns prediction for the batch of audio tensors."""
        predictions = []
        with torch.no_grad():
            self.model.eval()
            
            for audio in audio_batch:
                audio = audio.to(self.device)
                indexes = torch.argmax(self.model(audio).logits, dim=-1)
                predictions.append(label_normalizer(self.processor.batch_decode(indexes)[0]))# 
        return predictions
    
###############        pretrained from Tuckute et al. 2023      ##################

class FeatureExtractorCoch(BaseFeatureExtractor):
    def __init__(self, model_name, shuffled=False):
        self.model_name = model_name        # cochresnet50
        config = utils.load_dnn_config(model_name=self.model_name)
        self.signal_length = config['signal_length']

        module_path = os.path.join(config['base_directory'], config['model'], 'build_network.py')
        build_network_spec = importlib.util.spec_from_file_location("build_network", module_path)
        build_network = importlib.util.module_from_spec(build_network_spec)
        build_network_spec.loader.exec_module(build_network)

        model, _, _ = build_network.main(return_metamer_layers=True)
        super().__init__(model, config, shuffled=shuffled, sampling_rate=config['sampling_rate'])

        num_layers = len(self.layer_ids)
    
        self.layer_rates = []
        for i in range(num_layers):
            self.layer_rates.append(self.config['layers'][i]['rate'])
        
        
    def fwd_pass(self, aud):
        """
        Forward passes audio input through the model and captures 
        the features in the 'self.features' dict.

        Args:
            aud (ndarray): single 'wav' input of shape (t,) 
        
        Returns:
            input (torch.Tensor): returns the torch Tensor of the input sent passed through the model.
        """
        aud_input = torch.tensor(aud, dtype=torch.float32, device=self.device)  # Convert to tensor
        self.model.eval()
        with torch.no_grad():
            out = self.model(aud_input)

        return out
    
    def extract_features_for_clip(self, audio, context_samples=0, retain_context=True):
        """
        Extracts features for a single audio clip. Audio clip must be
        sampled at 20kHz and smaller than 2s.
        For the first short clip, the context is retained, but for the later clips
        the context is not retained. 

        Args:
            aud (ndarray): single 'wav' input of shape (t,)
            context_samples (int): number of samples of context in the audio clip.
            retain_context (bool): whether to retain the context in the audio clip.

        Returns:
            features (dict): extracted features for all layers.
        """
        if audio.shape[0] > self.signal_length:
            raise ValueError(f"Audio is longer than signal length: {self.signal_length}")
            
        padding_length = self.signal_length - audio.shape[0]
        padded_audio = np.pad(audio, (0, padding_length), mode='constant')
        stim_features = self.get_features(padded_audio)
        for ii, (layer_name, feats) in enumerate(stim_features.items()):
            extra_samples_padded = int(padding_length*self.layer_rates[ii]/self.sampling_rate)
            if not retain_context:
                extra_samples_context = int(context_samples*self.layer_rates[ii]/self.sampling_rate)
            else:
                extra_samples_context = 0    
            # remove extra padded or context samples...
            num_samples = feats.shape[0]
            stim_features[layer_name] = feats[extra_samples_context:num_samples-extra_samples_padded]
        return stim_features
    

    def get_short_clips(self, audio, context_samples=0):
        """
        Splits the audio into short clips of length 2s with input context length.
        The audio is expected to be sampled at 20kHz.

        Args:
            audio (ndarray): single 'wav' input of shape (t,)
            context_samples (int): number of samples of context in the audio clip.

        Returns:
            list_clips (list): list of short audio clips each having length of 
                self.signal_length (including the context_length).
        """

        clip_length = self.signal_length - context_samples 
        list_clips = []

        num_samples = audio.shape[0]
        idx = 0
        while num_samples > clip_length:
            audio_clip = audio[idx*clip_length:(idx+1)*clip_length]
            if idx == 0:
                audio_context = np.zeros((context_samples,))
            else:
                audio_context = audio[idx*clip_length-context_samples:idx*clip_length]
            audio_clip = np.concatenate([audio_context, audio_clip])
            list_clips.append(audio_clip)
            idx += 1
            num_samples -= clip_length

        audio_clip = audio[idx*clip_length:]
        if idx == 0:
            audio_context = np.zeros((context_samples,))
        else:
            audio_context = audio[idx*clip_length-context_samples:idx*clip_length]
        audio_clip = np.concatenate([audio_context, audio_clip])
        list_clips.append(audio_clip)
        return list_clips

    
    def extract_features(self, stim_audios, sampling_rate, stim_durations=None, pad_time=None):
        """
        Returns raw features for all layers of the DNN..!

        Args:
            stim_audios (dict): dictionary of audio inputs for each sentence.
                {stim_id: audio}
            sampling_rate (int): sampling rate of the audio inputs.
            stim_durations (dict): dictionary of sentence durations.
                {stim_id: duration}
            pad_time (float): amount of padding time in seconds.

        Returns:
            dict of dict: read this as features[layer_id][stim_id]
        """
        features = {id:{} for id in self.layer_ids}
        for stim_id, audio in stim_audios.items():

            if sampling_rate != self.sampling_rate:
                n_samples = int(audio.size*self.sampling_rate/sampling_rate)
                audio = resample(audio, n_samples)
            
            if pad_time is not None:
                context_samples = int(pad_time*self.sampling_rate)
            else:
                context_samples = 0

            audio_clips = self.get_short_clips(audio, context_samples=context_samples)

            stim_features_list = []
            for ii, clip in enumerate(audio_clips):
                if ii == 0:
                    retain_context = True
                else:
                    retain_context = False
                    
                stim_features_list.append(
                    self.extract_features_for_clip(
                        clip, 
                        context_samples=context_samples, 
                        retain_context=retain_context)
                    )
            ### I need context for the first short clip, but for the later clips 
            ### I don't need it....

            for layer_id in self.layer_ids:
                layer_name = self.get_layer_name(layer_id)
                features[layer_id][stim_id] = np.concatenate([stim_feats[layer_name] for stim_feats in stim_features_list], axis=0)

            del stim_features_list
        return features

@register_feature_extractor('cochresnet50')
class CochResnet50(FeatureExtractorCoch):
    def __init__(self, shuffled=False):
        model_name = 'cochresnet50'
        super().__init__(model_name, shuffled=shuffled)

@register_feature_extractor('cochcnn9')
class CochCNN9(FeatureExtractorCoch):
    def __init__(self, shuffled=False):
        model_name = 'cochcnn9'
        super().__init__(model_name, shuffled)
        







class FeatureExtractorMERT(BaseFeatureExtractor):
    def __init__(self, model_name, shuffled=False):
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-330M",trust_remote_code=True)
        # loading our model weights
        model = AutoModel.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)
        # loading the corresponding preprocessor config
        super().__init__(model, config, shuffled=shuffled, sampling_rate=config['sampling_rate'])
        
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
    

class FeatureExtractorCLAP(BaseFeatureExtractor):
    def __init__(self, model_name, shuffled=False):
        self.processor = ClapProcessor.from_pretrained("laion/larger_clap_general")
        # loading our model weights
        model = ClapModel.from_pretrained("laion/larger_clap_general")
        # loading the corresponding preprocessor config
        super().__init__(model, config, shuffled=shuffled, sampling_rate=config['sampling_rate'])

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