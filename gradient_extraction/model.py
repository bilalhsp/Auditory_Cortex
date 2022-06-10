from transformers import Speech2TextForConditionalGeneration, Speech2TextProcessor
from auditory_cortex.feature_extractors import Feature_Extractor_S2T

class TransformerModel:	
    def __init__(self):
        self.layers = ["model.encoder.conv.conv_layers.0","model.encoder.conv.conv_layers.1","model.encoder.layers.0.fc2",
                        "model.encoder.layers.1.fc2","model.encoder.layers.2.fc2","model.encoder.layers.3.fc2",
						"model.encoder.layers.4.fc2","model.encoder.layers.5.fc2","model.encoder.layers.6.fc2",
						"model.encoder.layers.7.fc2","model.encoder.layers.8.fc2","model.encoder.layers.9.fc2"]
        self.model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-librispeech-asr")
        self.processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")