from ..dnn_feature_extractors import FeatureExtractorW2L, FeatureExtractorDeepSpeech2
from ..dnn_feature_extractors import FeatureExtractorS2T, FeatureExtractorWhisper
from ..dnn_feature_extractors import FeatureExtractorW2V2, FeatureExtractorW2V2Audioset
from ..dnn_feature_extractors import FeatureExtractorCoch
from auditory_cortex import DNN_MODELS

# mapping strings to classes
DNN_MODELS_MAP = {
    DNN_MODELS[0]: FeatureExtractorDeepSpeech2,
    DNN_MODELS[1]: FeatureExtractorS2T,
    DNN_MODELS[2]: FeatureExtractorW2L,
    DNN_MODELS[3]: FeatureExtractorWhisper,
    DNN_MODELS[4]: FeatureExtractorWhisper,
    DNN_MODELS[5]: FeatureExtractorW2V2,
    DNN_MODELS[6]: FeatureExtractorW2V2Audioset,
    DNN_MODELS[7]: FeatureExtractorCoch,
    DNN_MODELS[8]: FeatureExtractorCoch,
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
    return DNN_MODELS_MAP[model_name](model_name, shuffled=shuffled)