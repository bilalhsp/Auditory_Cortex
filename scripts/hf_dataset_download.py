import os
import time
import argparse
from datasets import load_dataset
# from auditory_cortex import cache_dir


def is_not_speech_or_music(sample):
	"""Returns true if 'speech' is not one of the labels."""
	for label in sample['human_labels']:
		if 'speech' in label.lower() or 'music' in label.lower():
			return False
	return True 

import pickle
def read_from_disk(filepath):
	if os.path.exists(filepath):
		with open(filepath, 'rb') as F:
			return pickle.load(F)
	else:
		return None


def write_to_disk(list_indices, i, filepath):
	existing_dict = read_from_disk(filepath)

	if existing_dict is None:
		existing_dict = {}
	existing_dict[i] = list_indices
	with open(filepath, 'wb') as F:
		pickle.dump(existing_dict, F)
	print(f"Dict updated to {filepath}")
      
def save_corrupted_indices(dataset, index_i, corrupt_indices_path):
    # index_i = 1
    step = 100000
    start = int(step*index_i)
    endd = int(step*(index_i+1))
    total_indices = len(dataset)
    train_dataset = dataset
    corrupt_file_indices = []
    if start < total_indices:
        if endd > total_indices:
            endd = total_indices

        for i in range(start,endd):
            try:
                sample = train_dataset[i]
            except:
                corrupt_file_indices.append(i)

        write_to_disk(corrupt_file_indices, index_i, filepath=corrupt_indices_path)


def get_corrupt_indices_list():
	# save datasets to scratch..
	dataset_name = 'audioset_natual_sounds'
	hf_cache_dir = os.path.join('/scratch/gilbreth/ahmedb/cache', 'huggingface')
	hf_datasets_cache = os.path.join(hf_cache_dir, 'datasets')

	list_indices = []
	for ind in range(18):
		corrupt_indices_path = os.path.join(hf_datasets_cache, f"corrupt_indices_{ind}.pkl")
		corr_indices_dict = read_from_disk(corrupt_indices_path)
		for k, v in corr_indices_dict.items():
			list_indices.extend(v)
	return list_indices



# ------------------  get parser ----------------------#
def get_parser():
    parser = argparse.ArgumentParser(
        description='This is to download Audioset dataset '+
            ', filter and preprocess the dataset.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-tr','--training', dest='training', action='store_true', default=False,
        help="Specify if training data to be downloaded."
    )
    parser.add_argument(
        '-ts','--test', dest='test', action='store_true', default=False,
        help="Specify if validation data to be downloaded."
    )
    parser.add_argument(
        '-i','--index_i', dest='index_i', type=int, default=0,
        choices=range(18),
        help="Specify if validation data to be downloaded."
    )

    return parser


# def clean_dataset(dataset_input):
     

# from datasets import Dataset

def remove_corrupted_entries(dataset):
    """
    Removes corrupted entries from the dataset based on the given indices.

    Parameters:
    - corrupted_indices (list): A list of indices of the corrupted entries.

    Returns:
    - Dataset: A cleaned dataset with corrupted entries removed.
    """
    print(f"Cleaning dataset...")
    # corrupted_indices = []
    # for i in range(len(dataset)):
    #     try:
    #         sample = dataset[i]
    #     except:
    #         corrupted_indices.append(i)
    all_indices = list(range(len(dataset)))
    corrupted_indices = get_corrupt_indices_list()
    # Filter out the corrupted indices
    valid_indices = [idx for idx in all_indices if idx not in corrupted_indices]
    # Select the valid indices to create the cleaned dataset
    cleaned_dataset = dataset.select(valid_indices)
    print(f"Original dataset size: {len(dataset)}")
    print(f"Cleaned dataset size: {len(cleaned_dataset)}")
    return cleaned_dataset
     
     
def download_and_preprocess_audioset(args):
    
    cache_dir = '/scratch/gilbreth/ahmedb/cache'

    num_tasks = 8

    cache_dir = os.path.join(cache_dir, 'huggingface', 'datasets')
    dataset = load_dataset(
        "agkphysics/AudioSet",
        'unbalanced',
        trust_remote_code=True,
        cache_dir=cache_dir,
        num_proc=num_tasks,
        )



    # save datasets to scratch..
    dataset_name = 'audioset_natual_sounds'
    hf_cache_dir = os.path.join('/scratch/gilbreth/ahmedb/cache', 'huggingface')
    hf_datasets_cache = os.path.join(hf_cache_dir, 'datasets')

    # corrupt_indices_path = os.path.join(hf_datasets_cache, f"corrupt_indices_{args.index_i}.pkl")
    # save_corrupted_indices(dataset['train'], args.index_i, corrupt_indices_path)


    if args.training:
        # filter out speech and music
        train_dataset = dataset['train']
        train_dataset = remove_corrupted_entries(train_dataset)

        print(f"Filtering out speech and music...")
        train_dataset = train_dataset.filter(
                is_not_speech_or_music,
                # num_proc=num_tasks,
                )
        print(f"done, writing to disk...!")
        train_dataset_path = os.path.join(hf_datasets_cache, f"{dataset_name}_train")
        train_dataset.save_to_disk(
                train_dataset_path,
                num_proc=num_tasks,
                )

    if args.test:
        test_dataset = dataset['test']
        test_dataset = remove_corrupted_entries(test_dataset)
        print(f"Filtering out speech and music...")
        test_dataset = test_dataset.filter(is_not_speech_or_music)
        test_dataset_path = os.path.join(hf_datasets_cache, f"{dataset_name}_test")
        test_dataset.save_to_disk(test_dataset_path)


# ------------------  main function ----------------------#

if __name__ == '__main__':

    start_time = time.time()
    parser = get_parser()
    args = parser.parse_args()

    # display the arguments passed
    for arg in vars(args):
        print(f"{arg:15} : {getattr(args, arg)}")

    download_and_preprocess_audioset(args)
    elapsed_time = time.time() - start_time
    print(f"It took {elapsed_time/60:.1f} min. to run.")