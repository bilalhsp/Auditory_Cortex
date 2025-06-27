## 🧠 Neural Data

This guide provides information on the neural datasets currently supported in this repository and instructions for adding new datasets.

The `neural_data/` directory provides core functionality for integrating neural datasets into the repository. It defines a common interface and utilities to load, preprocess, and access neural data in a consistent manner.

```
neural_data/
├── base_dataset.py      ← Defines `BaseDataset` class (interface for all datasets)
├── factory.py           ← Maps dataset names to corresponding dataset classes
```

### 🛠️ Adding a New Neural Dataset

To add support for a new dataset, follow these steps:

1. **Create a Subclass of `BaseDataset`**  
   Define a new class in a separate file (or within an existing module) that inherits from `BaseDataset`.  
   This class should handle dataset-specific file access, parsing, and return data in the expected format.

2. **Update the Configuration**  
   Add the new dataset's name to the `dataset_name` field in your `config.yml`.

3. **Register in the Factory**  
   In `neural_data/factory.py`, map the dataset name string to your new dataset class so that it can be instantiated dynamically.

### 🧪 Example: Creating a Neural Dataset

```python
from auditory_cortex.neural_data import create_neural_dataset

dataset_name = 'ucsf'
session = 200206
ucsf_data = create_neural_dataset(dataset_name, session)
```

---

### ✅ Supported Datasets

#### 🔹 UCSF

Malone Lab neural recordings from squirrel monkeys, recorded from primary and non-primary auditory cortex.  
Approximately 40 recording sessions are available. Each session is identified by the recording date (e.g., `200206` corresponds to February 6, 2020).

**Required Files:**

- Neural recordings (one folder per recording session)
- Metadata files:
  - `MonkVocs_15Blocks.wav`
  - `SqMoPhys_MVOCStimcodes.mat`
  - `out_sentence_details_timit_all_loudness.mat`

These files and directories should be placed inside the `neural_data_dir` specified in the config file (`config.yml`).

**Directory Structure:**
```
neural_data_dir/ucsf/
├── SqMoPhys_MVOCStimcodes.mat
├── MonkVocs_15Blocks.wav
├── out_sentence_details_timit_all_loudness.mat
├── 200205/
├── 200206/
...
```

#### 🔹 UCDAVIS

Neural data recorded at the Recanzone Lab from macaque monkeys. This dataset is currently in the early stages, and more sessions are being collected.

- Neural recordings
    + a separate .mat file for every session
    + all files placed in ./Data directory
- Metadata files:
    + **recanzone_timit_details.mat**: details about TIMIT stimuli presentations
    + **MSL.mat**: details about mVocs stimuli presentations
- Stimuli (audio files):
    + **NIMH_Mvoc_WFM**: directory containing files for all mVocs stimuli
    + **TIMIT_48000**: directory containing files for all TIMIT stimuli 

**Directory Structure:**
```
neural_data_dir/ucdavis/
├── MSL.mat
├── recanzone_timit_details.mat
├── Data/
│   └── ...                     # recordings for all sessions
├── NIMH_Mvoc_WFM/
│   ├── 9-Cue-41-Grunt.wfm
│   └── ...                     # mVocs audio files
├── TIMIT_48000/
│   ├── 127-fsdc0_si2234.wfm
│   └── ...                     # TIMIT audio files
```

Ensure these are placed under the `neural_data_dir/ucdavis/` directory as defined in the config file.
