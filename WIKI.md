### Neural Data
sent ID's with repeats:  [12,13,32,43,56,163,212,218,287,308]


### Useful Gilbreth commands:
- Creating conda env with jupyter kernel at user specified location...
    conda-env-mod create -p /depot/jgmakin/data/conda_env/w2l_cortex --jupyter
- Using the conda-env
     or module load anaconda/2020.11-py38
    module load learning/conda-2020.11-py38-gpu
    module load cuda/11.0.3
    module load ml-toolkit-gpu/pytorch/1.7.1

    module load use.own
    module load conda-env/w2l_cortex-py3.8.5


### Export tensorboard logs to web...
    tensorboard dev upload --logdir=/scratch/gilbreth/ahmedb/wav2letter/modified_w2l/logs/lightning_logs/


### creating a new environment:
conda-env-mod create -n huggingface --jupyter

#### load the newly created environment
module load anaconda/2020.11-py38
module load use.own
module load conda-env/huggingface-py3.8.5

#### installing required packages
- Install pytorch alingwith suitable 
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


numpy scipy matplotlib pandas sentencepiece transformers[Audio] datasets
soundfile librosa