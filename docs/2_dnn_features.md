## 🤖 DNN Features

This guide provides information on the DNN feature extractors currently supported in this repository and instructions for adding new extractors.

For every pretrained deep neural network that we want to analyze, we need to define a feature extractor object that extracts and saves hidden layer activations for the stimulus set. To standardize these feature extractors, a base class is defined with functionality based on forward hooks to extract DNN features. It defines a common interface and utilities to extract and manage representations in a modular and extensible way. Here is how the directory is organized


```
dnn_feature_extractor/
├── base_feature_extractor.py  ← Defines `BaseFeatureExtractor` class (interface for all extractors)
├── factory.py                 ← Maps model names to their corresponding extractor classes
```

### 🛠️ Adding a New Feature Extractor

To add support for a new pretrained model (DNN), follow these steps:

1. **Create a Subclass of `BaseFeatureExtractor`**  
   Define a new class (in a separate file or within the same module) that inherits from `BaseFeatureExtractor`.  
   Implement method for doing a forward pass through the DNN.

2. **Update the Configuration**  
   Add the new model's name to the `dnn_models` field in the configuration file.

3. **Register in the Factory**  
   In `dnn_feature_extractor/factory.py`, map the model name string to your new feature extractor class for dynamic instantiation.

### 🧪 Example: Creating a DNN Feature Extractor

```python
from auditory_cortex.feature_extractors import create_feature_extractor

model_name = 'wav2vec2'
shuffled = False
extractor = create_feature_extractor(model_name, shuffled)
```


### ✅ Supported DNNs

#### 🔹 deepspeech2
#### 🔹 speech2text
#### 🔹 wav2letter_modified
#### 🔹 whisper_tiny
#### 🔹 whisper_base
#### 🔹 wav2vec2
