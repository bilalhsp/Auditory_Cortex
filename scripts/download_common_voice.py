import os
import shutil
import torchaudio

data_dir = '/scratch/gilbreth/ahmedb/data/'
dataset_path = os.path.join(data_dir, "common_voice")

if os.path.exists(dataset_path):
    shutil.rmtree(dataset_path)
# create parent folder for the datadirectory.
os.makedirs(dataset_path)
print("Successfully created an empty parent directory.")


dataset =  torchaudio.datasets.COMMONVOICE(
    root = dataset_path,
    tsv = 'test.tsv'
)