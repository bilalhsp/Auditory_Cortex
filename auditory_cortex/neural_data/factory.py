from .ucdavis_data.ucdavis_dataset import UCDavisDataset
from .ucsf_data.ucsf_dataset import UCSFDataset
from auditory_cortex import NEURAL_DATASETS


# mapping strings to classes
NEURAL_DATASET_MAP = {
	NEURAL_DATASETS[0]: UCSFDataset,
	NEURAL_DATASETS[1]: UCDavisDataset,
}

def create_neural_dataset(dataset_name, *args, **kwargs):
	"""Create a neural dataset object.

	Args:
		dataset_name (str): name of the dataset.
		*args: Positional arguments to pass to the dataset class constructor.
        **kwargs: Keyword arguments to pass to the dataset class constructor.
	"""
	return NEURAL_DATASET_MAP[dataset_name](*args, **kwargs)
