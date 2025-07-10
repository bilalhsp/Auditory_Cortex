from ..ucdavis_data.ucdavis_dataset import UCDavisDataset
from ..ucsf_data.ucsf_dataset import UCSFDataset
from ..ucdavis_data.ucdavis_metadata import UCDavisMetaData
from ..ucsf_data.ucsf_metadata import UCSFMetaData


from auditory_cortex import NEURAL_DATASETS

def list_neural_datasets():
    """Returns the list of available neural datasets."""
    return NEURAL_DATASETS


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


METADATA_MAP = {
	NEURAL_DATASETS[0]: UCSFMetaData,
	NEURAL_DATASETS[1]: UCDavisMetaData,
}

def create_neural_metadata(dataset_name, *args, **kwargs):
	"""Create a neural metadata object.

	Args:
		dataset_name (str): name of the dataset.
		*args: Positional arguments to pass to the dataset class constructor.
        **kwargs: Keyword arguments to pass to the dataset class constructor.
	"""
	return METADATA_MAP[dataset_name](*args, **kwargs)
