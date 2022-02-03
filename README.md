# Auditory_cortex


[Repo Link](https://github.com/bilalhsp/Auditory_Cortex)

```git clone https://github.com/bilalhsp/Auditory_Cortex.git```

go to the directory Auditory_Cortex

```pip install -e .```

```
git add .
git commit -m "message"
git push origin main

username:
password:
```
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
