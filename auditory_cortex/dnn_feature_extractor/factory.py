from .dnn_feature_extractors import FeatureExtractorW2L, FeatureExtractorDeepSpeech2
from .dnn_feature_extractors import FeatureExtractorS2T, FeatureExtractorW2V2, FeatureExtractorWhisper


# mapping strings to classes
FEATURE_EXTRACTORS = {
    'wav2letter_modified': FeatureExtractorW2L,
    'deepspeech2': FeatureExtractorDeepSpeech2,
    'speech2text': FeatureExtractorS2T,
    'wav2vec2': FeatureExtractorW2V2,
    'whisper_tiny': FeatureExtractorWhisper,
    'whisper_base': FeatureExtractorWhisper,
}

def create_feature_extractor(model_name, shuffled=False):
    """
    Factory method to create a feature extractor object.

    Args:
        model_name (str): name of the model to be used.
        shuffled: (bool): If True, get features from untrained model.

    Returns:
        object: returns the object of the feature extractor class.
    """
    return FEATURE_EXTRACTORS[model_name](model_name, shuffled=shuffled)