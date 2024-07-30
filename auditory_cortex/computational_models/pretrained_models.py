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
import os
import yaml
from abc import ABC, abstractmethod, abstractproperty
import torch
from transformers import ClapModel, ClapProcessor

from auditory_cortex import aux_dir



class BasePreTrained(ABC):
    def __init__(self, model_name) -> None:
        super().__init__()
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        

        self.layer_names = []
        self.layer_ids = []
        self.layer_types = []
        self.receptive_fields = []

        self.read_config()


    @abstractproperty
    def model(self):
        pass


    def read_config(self):
        """
        Reads the model-specific configuration file and
        and details of the layers to be analysed.
        """
        # read yaml config file
        config_file = os.path.join(
            aux_dir, f"{self.model_name}_config.yml")
        with open(config_file, 'r') as f:
            self.config = yaml.load(f, yaml.FullLoader)
        self.num_layers = len(self.config['layers'])
        
        for i in range(self.num_layers):
            self.layer_names.append(self.config['layers'][i]['layer_name'])
            self.layer_ids.append(self.config['layers'][i]['layer_id'])
            self.layer_types.append(self.config['layers'][i]['layer_type'])
            self.receptive_fields.append(self.config['layers'][i]['RF'])

    def get_layer_names(self):
        return self.layer_names
    
    def get_layer_ids(self):
        return self.layer_ids
    
    def get_model_layers(self):
        """Retrieves the layer objects for all layer names
        in the config.
        """
        return {
            layer_name: dict([*self.model.named_modules()])[layer_name]
            for layer_name in self.layer_names
            }


class CLAP(BasePreTrained):
    def __init__(self) -> None:
        super(CLAP, self).__init__('CLAP')
        self.processor = ClapProcessor.from_pretrained("laion/larger_clap_general")
        self.model = ClapModel.from_pretrained("laion/larger_clap_general")
        

    @property
    def model(self):
        """getter for 'model' property"""
        return self._model

    @model.setter
    def model(self, model):
        """setter for 'model' property"""
        self._model = model
