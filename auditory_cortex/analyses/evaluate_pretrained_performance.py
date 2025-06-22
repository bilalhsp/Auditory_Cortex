import os
import torch
import numpy as np

import csv
import jiwer
import soundfile
import scipy
# from pydub import AudioSegment
import pandas as pd
import torchaudio

from auditory_cortex.deprecated.dataloader import DataLoader

data_dir = '/scratch/gilbreth/ahmedb/data/'

# voxpopuli: /scratch/gilbreth/ahmedb/data/voxpopuli

class LibriSpeechDataset():
    """Dataset for Librispeech data"""
    def __init__(self, manifest, labels_normalizer=None):
        self.manifest = pd.read_csv(manifest)
        self.labels_normalizer = labels_normalizer
    
    def __len__(self):
        return len(self.manifest)
        
    def __getitem__(self, idx):
        audio_path = self.manifest.iloc[idx]['audio']
        trans = self.manifest.iloc[idx]['trans']
        audio, fs = soundfile.read(audio_path, always_2d=True)
        audio = audio.squeeze()
        # In order to match dimensions/dtype of audio required by librosa processing...!
        # return audio.transpose(), fs, trans
        # In order to match dimensions/dtype of audio returned by pytorch loader...!
        return torch.tensor(np.expand_dims(audio, axis=0), dtype=torch.float32), fs, trans
    
    def collate_fn(self, data):
        """Defines how individual data items should be combined,
        simply returns them as a list"""    
        waves = []
        labels = []
        target_lens = []
        for (wav, fs, trans, *_) in data:
            waves.append(wav)
            # waves.append(wav.squeeze())
            if self.labels_normalizer is not None:
                trans = self.labels_normalizer(trans)
            else:
                trans = trans.lower()
            labels.append(trans) #.lower()
            target_len = np.ceil(50*wav.shape[-1]/fs) - 1
            target_lens.append(target_len)
        # padded_waves = torch.nn.utils.rnn.pad_sequence(sequences=waves, batch_first=True)
        # return padded_waves, labels, target_lens
        return waves, labels, target_lens



def get_LibriSpeech_dataloader(
        test_data:str=None, batch_size=32, labels_normalizer=None
    ):
    """
    Returns dataloader for Librispeech test data
    
    Args:
        test_data: str = choices = ['test_clean', 'test_other']
        batch_size: int = Default = 32
    """
    if test_data is None:
        test_data = 'test_other'
    assert test_data in ['test_clean', 'test_other'], "test data must be either 'test_clean' or 'test_other'"
    data_dir = '/scratch/gilbreth/ahmedb/data/LibriSpeech'
    test_dataset_file = f'{test_data}_manifest.csv'
    file_path = os.path.join(data_dir, test_dataset_file)
    dataset = LibriSpeechDataset(file_path, labels_normalizer)

    test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
        # num_workers=self.num_workers, 
        shuffle=False, collate_fn=dataset.collate_fn, 
        #  pin_memory=True, persistent_workers=self.persistant_workers  
        )

    return test_dataloader

################################################
########        TEDLIUM          ###############

class TEDLIUM3():
    """Dataset for TEDLIUM3 data"""
    def __init__(self, labels_normalizer=None):
        data_dir = '/scratch/gilbreth/ahmedb/data/'
        dataset_path = os.path.join(data_dir, "TEDLIUM3")

        self.dataset =  torchaudio.datasets.TEDLIUM(
            root = dataset_path,
            release = 'release3',
            subset = 'test',
            download=False
        )
        self.labels_normalizer = labels_normalizer
    
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def collate_fn(self, data):
        """Defines how individual data items should be combined,
        simply returns them as a list"""    
        waves = []
        labels = []
        target_lens = []
        for (wav, fs, trans, *_) in data:
            waves.append(wav)
            # waves.append(wav.squeeze())
            if self.labels_normalizer is not None:
                trans = self.labels_normalizer(trans)
            else:
                trans = trans.lower()
            labels.append(trans) #.lower()
            target_len = np.ceil(50*wav.shape[-1]/fs) - 1
            target_lens.append(target_len)
        # padded_waves = torch.nn.utils.rnn.pad_sequence(sequences=waves, batch_first=True)
        # return padded_waves, labels, target_lens
        return waves, labels, target_lens

def TEDLIUM_remove_non_speech_segments(labels, predictions):
        """TEDLIUM dataset has background sounds and other non-speech segments,
        labels corresponding to these label say 'ignore_time_segment_in_scoring',
        this functions looks for these labels and pops the corresponding entries
        from both list of labels and predictions.
        """
        # checking for string 'ignore time segment in scoring'...
        for i, ref in enumerate(labels):
            if ('ignore' in ref) and ('time' in ref) and ('segment' in ref) and ('scoring' in ref):
                labels.pop(i)
                predictions.pop(i)
        return labels, predictions


def get_TEDLIUM_dataloader(
        batch_size=32, labels_normalizer=None
    ):
    """
    Returns dataloader for Librispeech test data
    
    Args:
        test_data: str = choices = ['test_clean', 'test_other']
        batch_size: int = Default = 32
    """

    dataset = TEDLIUM3(labels_normalizer)

    test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
        # num_workers=self.num_workers, 
        shuffle=False, collate_fn=dataset.collate_fn, 
        #  pin_memory=True, persistent_workers=self.persistant_workers  
        )

    return test_dataloader


################################################
########        VoxPopuli.en          ###############s

class VoxPopuliDataset():
    """VoxPopuli (en)"""
    def __init__(self, labels_normalizer=None):
        # data_dir = '/scratch/gilbreth/ahmedb/data/'
        voxpopuli_dir = os.path.join(data_dir, 'voxpopuli/transcribed_data/en')
        filename = 'asr_test.tsv'
        manifest_file = os.path.join(voxpopuli_dir, filename)
        audio_paths = []
        audio_trans = []
            
        with open(manifest_file, 'r') as file:
            reader  = csv.reader(file, delimiter='\t')
            for i, row in enumerate(reader):
                if i>0:
                    if row[2] != '':
                        id = row[0]
                        year = id[:4]
                        audio_filepath = os.path.join(
                            voxpopuli_dir, year, f'{id}.ogg'
                        )
                        audio_paths.append(audio_filepath)
                        audio_trans.append(row[2])
        data = np.array([audio_paths, audio_trans], dtype=str).transpose()
        self.manifest = pd.DataFrame(
                columns=['audio', 'trans'],
                data = data
            )
        self.labels_normalizer = labels_normalizer
        self.fs = 16000
    
    def __len__(self):
        return len(self.manifest)
        
    def __getitem__(self, idx):
        audio_path = self.manifest.iloc[idx]['audio']
        trans = self.manifest.iloc[idx]['trans']

        audio, fs = soundfile.read(audio_path, always_2d=True)
        audio = audio.squeeze()
        # In order to match dimensions/dtype of audio required by librosa processing...!
        # return audio.transpose(), fs, trans
        # In order to match dimensions/dtype of audio returned by pytorch loader...!
        return torch.tensor(np.expand_dims(audio, axis=0), dtype=torch.float32), self.fs, trans
    
    def collate_fn(self, data):
        """Defines how individual data items should be combined,
        simply returns them as a list"""    
        waves = []
        labels = []
        target_lens = []
        for (wav, fs, trans, *_) in data:
            waves.append(wav)
            # waves.append(wav.squeeze())
            if self.labels_normalizer is not None:
                trans = self.labels_normalizer(trans)
            else:
                trans = trans.lower()
            labels.append(trans) #.lower()
            target_len = np.ceil(50*wav.shape[-1]/fs) - 1
            target_lens.append(target_len)
        # padded_waves = torch.nn.utils.rnn.pad_sequence(sequences=waves, batch_first=True)
        # return padded_waves, labels, target_lens
        return waves, labels, target_lens



def get_VoxPopuli_dataloader(
        batch_size=32, labels_normalizer=None
    ):
    """
    Returns dataloader for Common Voice 5.1
    
    Args:
        batch_size: int = Default = 32
    """

    dataset = VoxPopuliDataset(labels_normalizer)

    test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
        # num_workers=self.num_workers, 
        shuffle=False, collate_fn=dataset.collate_fn, 
        #  pin_memory=True, persistent_workers=self.persistant_workers  
        )

    return test_dataloader







################################################
########        Common voice 5.1          ###############s

class CommonVoiceDataset():
    """Common Voice 5.1 (en)"""
    def __init__(self, labels_normalizer=None):
        # data_dir = '/scratch/gilbreth/ahmedb/data/'
        common_voice_dir = os.path.join(data_dir, 'common_voice/data/cv-corpus-5.1-2020-06-22/en/')
        filename = 'test.tsv'
        manifest_file = os.path.join(common_voice_dir, filename)
        audio_paths = []
        audio_trans = []
            
        with open(manifest_file, 'r') as file:
            reader  = csv.reader(file, delimiter='\t')
            for i, row in enumerate(reader):
                if i>0:
                    if row[2] != '':
                        audio_filepath = row[1]
                        audio_paths.append(os.path.join(common_voice_dir, 'clips', audio_filepath))
                        audio_trans.append(row[2])
        data = np.array([audio_paths, audio_trans], dtype=str).transpose()
        self.manifest = pd.DataFrame(
                columns=['audio', 'trans'],
                data = data
            )
        self.labels_normalizer = labels_normalizer
        self.fs = 16000
    
    def __len__(self):
        return len(self.manifest)
        
    def __getitem__(self, idx):
        audio_path = self.manifest.iloc[idx]['audio']
        trans = self.manifest.iloc[idx]['trans']

        audio, fs = soundfile.read(audio_path, always_2d=True)
        duration_in_seconds = audio.size/fs
        num = int(duration_in_seconds*self.fs)
        audio = scipy.signal.resample(audio, num)
        audio = audio.squeeze()
        # In order to match dimensions/dtype of audio required by librosa processing...!
        # return audio.transpose(), fs, trans
        # In order to match dimensions/dtype of audio returned by pytorch loader...!
        return torch.tensor(np.expand_dims(audio, axis=0), dtype=torch.float32), self.fs, trans
    
    def collate_fn(self, data):
        """Defines how individual data items should be combined,
        simply returns them as a list"""    
        waves = []
        labels = []
        target_lens = []
        for (wav, fs, trans, *_) in data:
            waves.append(wav)
            # waves.append(wav.squeeze())
            if self.labels_normalizer is not None:
                trans = self.labels_normalizer(trans)
            else:
                trans = trans.lower()
            labels.append(trans) #.lower()
            target_len = np.ceil(50*wav.shape[-1]/fs) - 1
            target_lens.append(target_len)
        # padded_waves = torch.nn.utils.rnn.pad_sequence(sequences=waves, batch_first=True)
        # return padded_waves, labels, target_lens
        return waves, labels, target_lens



def get_CommonVoice_dataloader(
        batch_size=32, labels_normalizer=None
    ):
    """
    Returns dataloader for Common Voice 5.1
    
    Args:
        batch_size: int = Default = 32
    """

    dataset = CommonVoiceDataset(labels_normalizer)

    test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
        # num_workers=self.num_workers, 
        shuffle=False, collate_fn=dataset.collate_fn, 
        #  pin_memory=True, persistent_workers=self.persistant_workers  
        )

    return test_dataloader



################################################
########        WER          ###############

def compute_WER(
        model_name, benchmark=None, batch_size=32
    ):
    """
    Args:
        model_name: str= computational model supported by auditory_cortex.model_names 
        benchmark: str = specifies dataset to be teseted on, MUST include dataset name 
            and any sub-category in the dataset e.g. Librispeech-clean
        
    """

    if benchmark is None:
        benchmark = 'librispeech-test-clean'
    benchmark = benchmark.lower()
    auditory_dataloader = DataLoader()
    # using the same labels normalizer for all networks..
    # labels_normalizer = auditory_dataloader.get_DNN_obj('whisper_tiny').get_labels_normalizer()

    dnn_obj = auditory_dataloader.get_DNN_obj(model_name)
    labels_normalizer = dnn_obj.get_labels_normalizer() # all DNN's have access to normalizer form whisper.
        
    TED_talks = False
    if 'librispeech' in benchmark:
        if 'clean' in benchmark:
            test_data = 'test_clean'
        elif 'other' in benchmark:
            test_data = 'test_other'

        test_dataloader = get_LibriSpeech_dataloader(
            test_data, batch_size=batch_size, labels_normalizer=labels_normalizer
            )
    
    elif 'tedlium' in benchmark:
        TED_talks = True
        test_dataloader = get_TEDLIUM_dataloader(
            batch_size=batch_size, labels_normalizer=labels_normalizer
            )
    elif 'common-voice' in benchmark:
        test_dataloader = get_CommonVoice_dataloader(
            batch_size=batch_size, labels_normalizer=labels_normalizer
        )
    elif 'voxpopuli' in benchmark:
        test_dataloader = get_VoxPopuli_dataloader(
            batch_size=batch_size, labels_normalizer=labels_normalizer
        )
    else:
        raise NameError(f"Upsupported dataset or invalid name!")

    print(f"Computing WER for '{model_name}' now...")
    cumulative_WER = []
    for audio, labels, target_lens in test_dataloader:
        predictions = dnn_obj.batch_predictions(audio)
        if TED_talks:
            labels, predictions = TEDLIUM_remove_non_speech_segments(labels, predictions)

        out = jiwer.process_words(
            labels,
            predictions
        )
        cumulative_WER.append(out.wer)

    # avg_wer in percentages..
    avg_wer = 100*sum(cumulative_WER)/len(cumulative_WER)
    return avg_wer