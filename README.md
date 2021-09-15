# Auditory_cortex
Modelling the auditory cortex using task optimized Deep Learning models 

"Dataset.py" module defines the "Dataset" class that provides routines for loading the neural data. 
Use "Neural_data_files.json" to specify which channel files (_*_MUspk.mat)are to be loaded. You may need to edit this file to add all the channel filenames.
Example code to use "Dataset" class:

import Auditory_Cortex.Dataset as dataset
data = dataset.Neural_Data("path_of_working_directory") 
#"path_of_working_directory" is the address of the folder containing channel files and json file.

#Functions:
data.retrieve_spike_times(sent = sentence_code) 
#Returns times of spikes, relative to stimulus onset or absolute time
#provide 'sentence_code' or 'trial_number' as argument
#Returns a dictionary, with channel # as the keys to the spike_times of specific channel.

data.retrieve_spikes_count(sent = sentence_code, win=bin_size) 
#Returns number of spikes in every 'win' miliseconds duration following the stimulus onset time.
#provide 'sentence_code' or 'trial_number' as argument
#provide 'bin_size' as the desired size of bin in miliseconds.
#Returns a dictionary, with channel # as the keys to the spikes_count of specific channel.
