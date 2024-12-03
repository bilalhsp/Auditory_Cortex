import os
import torch
import logging
import numpy as np
from auditory_cortex import results_dir, CACHE_DIR
from .base_feature_extractor import BaseFeatureExtractor
from wav2letter.models import Wav2LetterRF, Wav2LetterSpect
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from transformers import AutoProcessor, WhisperForConditionalGeneration, AutoModelForPreTraining
from transformers import Speech2TextForConditionalGeneration, Speech2TextProcessor
from transformers import AutoModel, Wav2Vec2FeatureExtractor
from transformers import ClapModel, ClapProcessor

# import GPU specific packages...
from auditory_cortex import hpc_cluster
if hpc_cluster:
	# import cupy as cp
	import fairseq
	from deepspeech_pytorch.model import DeepSpeech
	import deepspeech_pytorch.loader.data_loader as data_loader
	from deepspeech_pytorch.configs.train_config import SpectConfig

import logging
logger = logging.getLogger(__name__)

HF_CACHE_DIR = os.path.join(CACHE_DIR, 'hf_cache')

class FeatureExtractorW2L(BaseFeatureExtractor):
	def __init__(self, model_name, shuffled=False):

		self.model_name = model_name
		config = BaseFeatureExtractor.read_config_file(f"{self.model_name}_config.yml")
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
	
	# def fwd_pass_tensor(self, aud):
	# 	"""
	# 	Forward passes audio input through the model and captures 
	# 	the features in the 'self.features' dict.

	# 	Args:
	# 		aud (tensor): input tensor 'wav' input of shape (1, t) 
		
	# 	Returns:
	# 		input (torch.Tensor): returns the torch Tensor of the input sent passed through the model.
	# 	"""
	# 	with torch.no_grad():
	# 		out = self.model(aud)
	# 	return out
	

	def batch_predictions(self, audio_batch, label_normalizer):
		"""Returns prediction for the batch of audio tensors."""
		# method not tested yet
		predictions = []
		with torch.no_grad():
			# audio, _, target_lens = batch
			for audio in audio_batch:
				audio = audio.to(self.device)
				predictions.append(label_normalizer(self.model.decode(audio)[0]))# 
		return predictions


class FeatureExtractorDeepSpeech2(BaseFeatureExtractor):
	def __init__(self, model_name, shuffled=False):

		self.model_name = model_name
		config = BaseFeatureExtractor.read_config_file(f"{self.model_name}_config.yml")
		checkpoint = os.path.join(results_dir, 'pretrained_weights', model_name, config['saved_checkpoint'])
		model = DeepSpeech.load_from_checkpoint(checkpoint_path=checkpoint)
		super().__init__(model, config, shuffled=shuffled, sampling_rate=config['sampling_rate'])
		audio_config = SpectConfig()
		self.parser = data_loader.AudioParser(audio_config, normalize=True)
		# self.device = device
		# self.model = self.model.to(self.device)
		# self.processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
		# self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
	
	def get_spectrogram(self, aud):
		"""Gives spectrogram of audio input."""
		if torch.is_tensor(aud):
			aud = aud.cpu().numpy()
		return self.parser.compute_spectrogram(aud)

	def fwd_pass(self, aud):
		
		# spect = self.parser.compute_spectrogram(aud)
		# if not isinstance(aud, torch.Tensor): 

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
	
	# def fwd_pass_tensor(self, aud_spect):
	#     """
	#     Forward passes spectrogram of audio input through the model and captures 
	#     the features in the 'self.features' dict.

	#     Args:
	#         aud_spect (tensor): spectrogram of input tensor, shape (1, t, 80) 
		
	#     Returns:
	#         output (torch.Tensor): returns the torch Tensor of the input sent passed through the model.
	#     """
	#     aud_spect = aud_spect.unsqueeze(dim=0)
	#     lengths = torch.tensor([aud_spect.shape[-1]], dtype=torch.int64)
	#     self.model.eval()
	#     out = self.model(aud_spect, lengths)
	#     return out
	
	def batch_predictions(self, audio_batch, label_normalizer):
		"""Returns prediction for the batch of audio tensors."""
		# method not tested yet
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
	
class FeatureExtractorS2T(BaseFeatureExtractor):
	def __init__(self, model_name, shuffled=False):
		self.model_name = model_name

		config = BaseFeatureExtractor.read_config_file(f"{self.model_name}_config.yml")
		repo_name = config['repo_name']
		model = Speech2TextForConditionalGeneration.from_pretrained(repo_name, cache_dir=HF_CACHE_DIR)
		super().__init__(model, config, shuffled=shuffled, sampling_rate=config['sampling_rate'])

		self.processor = Speech2TextProcessor.from_pretrained(repo_name, cache_dir=HF_CACHE_DIR)


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
	
class FeatureExtractorW2V2(BaseFeatureExtractor):
	def __init__(self, model_name, shuffled=False):
		self.model_name = model_name

		config = BaseFeatureExtractor.read_config_file(f"{self.model_name}_config.yml")
		repo_name = config['repo_name']
		model = Wav2Vec2ForCTC.from_pretrained(repo_name, cache_dir=HF_CACHE_DIR)
		super().__init__(model, config, shuffled=shuffled, sampling_rate=config['sampling_rate'])

		self.processor = Wav2Vec2Processor.from_pretrained(repo_name, cache_dir=HF_CACHE_DIR)
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
		with torch.no_grad():
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
	
class FeatureExtractorWhisper(BaseFeatureExtractor):
	def __init__(self, model_name, shuffled=False):
		self.model_name = model_name
		config = BaseFeatureExtractor.read_config_file(f"{self.model_name}_config.yml")
		repo_name = config['repo_name']    
		model = WhisperForConditionalGeneration.from_pretrained(repo_name, cache_dir=HF_CACHE_DIR)
		super().__init__(model, config, shuffled=shuffled, sampling_rate=config['sampling_rate'])
		self.processor = AutoProcessor.from_pretrained(repo_name, cache_dir=HF_CACHE_DIR)

	def fwd_pass(self, aud):
		input_features = self.processor(aud, sampling_rate=16000, return_tensors="pt").input_features
		# with torch.no_grad():
		self.model.eval()
		input_features = input_features.to(self.device)
		with torch.no_grad():
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


class FeatureExtractorW2V2Audioset(BaseFeatureExtractor):
	def __init__(self, model_name, shuffled=False):
		self.processor = AutoProcessor.from_pretrained("ALM/wav2vec2-base-audioset")
		model = AutoModelForPreTraining.from_pretrained("ALM/wav2vec2-base-audioset")

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