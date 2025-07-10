# Auditory_cortex


## 📂 Repository Structure

This section outlines the main directories and files. See the corresponding documentation for details.

```
auditory_cortex/
├── auditory_cortex/             ← Core functionality code
│   ├── config/                  ← Configuration files
│   ├── dnn_feature_extractor/   ← DNN feature extraction
│   ├── neural_data/             ← Neural data handling
│   ├── plotters/                ← Plotting utilities
│   ├── analyses/                ← Analysis utilities
│   ├── io_utils/                ← To save/load/cache results
│   ├── dataloader.py            ← Data loading
│   ├── encoding.py              ← Encoding models
│   └── utils.py                 ← Helper functions
├── docs/                        ← Documentation
├── scripts/                     ← Experiment scripts
├── notebooks/                   ← Analysis notebooks
├── Gilbreth_job_submission/     ← Cluster job scripts
└── README.md                    ← Project overview
```



## 📚 Documentation

- 🧑‍🔬 [User Guide](docs/guide_users.md) — How to run scripts and use the models
- 👨‍💻 [Developer Guide](docs/guide_developers.md) — Contributing, structure, dev notes
- 🎧 [Dataset Guide](docs/guide_datasets.md) — Audio/neural dataset formats and loading
- 🏋️ [Training Guide](docs/guide_training.md) — Training new models or fine-tuning
- 📈 [Evaluation Guide](docs/guide_evaluation.md) — Metrics, evaluation scripts, visualization



## 🛠️ Installation
The repository uses following github repos:
```
https://github.com/bilalhsp/Wav2Letter
https://github.com/SeanNaren/deepspeech.pytorch.git
https://github.com/mcdermottLab/pycochleagram.git

```
Make sure the following packages are installed, they are used for deepspeech repo but are not installed during its installation.
python packges
- omegaconf

[Repo Link](https://github.com/bilalhsp/Auditory_Cortex)

```git clone https://github.com/bilalhsp/Auditory_Cortex.git```

go to the directory Auditory_Cortex

```pip install -e .```





## **IF memory error**
go to the subject directory:
```rm -rf .*```


Modelling the auditory cortex using task optimized Deep Learning models 

```Dataset.py``` module defines the ```Neural_Data``` class that provides routines for loading the neural data. 

Example code to use "Dataset" class:

```
from auditory_cortex.Dataset import Neural_Data
data = dataset.Neural_Data(path, session)
```

#"path_of_working_directory" is the address of the folder containing channel files and json file.

### Functions:

```data.retrieve_spike_times(sent = sentence_code)```

Returns times of spikes, relative to stimulus onset or absolute time

provide 'sentence_code' or 'trial_number' as argument

Returns a dictionary, with channel # as the keys to the spike_times of specific channel.

```data.retrieve_spikes_count(sent = sentence_code, win=bin_size)```

Returns number of spikes in every 'win' miliseconds duration following the stimulus onset time.

provide 'sentence_code' or 'trial_number' as argument

provide 'bin_size' as the desired size of bin in miliseconds

Returns a dictionary, with channel # as the keys to the spikes_count of specific channel.

#### Installing the virtual environment on Gilbreth
In order to install the virtual environment on Gilbreth..
- Load latest cuda and anaconda module to be able to use conda command on Gilbreth
    module load cuda/12.1.1     
    module load anaconda/2020.11-py38
- Creating virtual environemt as a 'module' for ease of loading on Gilbreth. -p specifies location where packages are installed, -m specifies location where module file is created.
    conda-env-mod create -p /depot/jgmakin/data/conda_env/cortex_project -m /depot/jgmakin/data/conda_env/etc/modules -j 
to remove it:
    conda-env-mod delete -p /depot/jgmakin/data/conda_env/cortex_project -m /depot/jgmakin/data/conda_env/etc/modules -j

- Loading the newly created environment as a module.
    module purge
    module load anaconda/2020.11-py38
    module use /depot/jgmakin/data/conda_env/etc/modules
    module load conda-env/cortex_project-py3.8.5
- Install rest of the packages.
    conda install numpy scipy matplotlib pandas jiwer cupy pytorch=2.0.1 torchaudio=2.0.2 tensorflow=2.13 tensorflow-probability=0.21 -c conda-forge -c pytorch
    
    pip install tensorflow==2.13 tensorflow-probability==0.21
    
    
    module loaded for cudatoolkit=11.8 cudnn=8.6



#### pretrained 'Wav2Letter':
Link of the website that provides pretrained 'Wav2letter' is: https://github.com/flashlight/wav2letter/tree/wav2letter-lua?tab=readme-ov-file#pre-trained-models

The checkpoint 'wget https://s3.amazonaws.com/wav2letter/models/librispeech-glu-highdropout.bin' is downloaded to: /scratch/gilbreth/ahmedb/wav2letter/pretrained

