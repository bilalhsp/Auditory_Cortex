"""
pretrained_models.py

This module contains classes for using pretrained speech-recognition
models with the methods needed for 'DNNFeatureExtractor'.

Author: Bilal Ahmed
Date: 07-05-2024
Version: 1.0
License: MIT
Dependencies: None

Purpose
-------
This module is designed to integrate pretrained speech-recognition
models into the study of computational models of auditory cortex.


Change Log
----------
- 07-05-2024: Initial version created by Bilal Ahmed.
"""




class FeatureExtractorW2L(BaseFeatureExtractor):
	def __init__(self):

		self.model_name = 'wav2letter_modified'
		config = BaseFeatureExtractor.read_config_file(f"{self.model_name}_config.yml")
		saved_checkpoint = config['saved_checkpoint']
		pretrained = config['pretrained']
		checkpoint = os.path.join(results_dir, 'pretrained_weights', self.model_name, saved_checkpoint)
		if pretrained:		
			model = Wav2LetterRF.load_from_checkpoint(checkpoint)
			logger.info(f"Loading from checkpoint: {checkpoint}")
		else:
			model = Wav2LetterRF()
			logger.info(f"Creating untrained network...!")
		super().__init__(model, config)
		

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


















# class CLAP(BasePreTrained):
# 	def __init__(self) -> None:
# 		super(CLAP, self).__init__('CLAP')
# 		self.processor = ClapProcessor.from_pretrained("laion/larger_clap_general")
# 		self.model = ClapModel.from_pretrained("laion/larger_clap_general")
		

# 	@property
# 	def model(self):
# 		"""getter for 'model' property"""
# 		return self._model

# 	@model.setter
# 	def model(self, model):
# 		"""setter for 'model' property"""
# 		self._model = model


# # read yaml config file
# 		config_file = os.path.join(
# 			aux_dir, f"{self.model_name}_config.yml")
# 		with open(config_file, 'r') as f:
# 			self.config = yaml.load(f, yaml.FullLoader)