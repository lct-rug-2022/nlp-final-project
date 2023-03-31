## Training

### Colab training 

Just copy specific script folder content to `/content/`: 
```shell
content/
  - params.json
  - train.py
```

Export `NEPTUNE_PROJECT` and `NEPTUNE_API_TOKEN`, login into `HF Hub` via `huggingface_login`


### Habrok Run

#### Habrok setup (once)

ALL INSIDE REPO 

1. Create `.env` file with `NEPTUNE_PROJECT` and `NEPTUNE_API_TOKEN`
2. Load modules same as in `jobscript` (for baseline it is `single_task.sh`)
   ```shell
   module purge
   module load CUDA/11.7.0
   module load cuDNN/8.4.1.50-CUDA-11.7.0
   module load NCCL/2.12.12-GCCcore-11.3.0-CUDA-11.7.0
   module load Python/3.10.4-GCCcore-11.3.0
   module load GCC/11.3.0
   ```
3. Create new venv and make `.venv` alias in the folder
    ```shell
    python3 -m venv /scratch/$USER/.envs/nlp-final-project
    ln -d -s /scratch/$USER/.envs/nlp-final-project .venv
    ```
4. Install libs 
    ```shell
    source .venv/bin/activate
    pip install -U -r requirements.txt
    ```
5. Make `results` folder at `/scratch` to log there (more space)
    ```shell
    mkdir -p /scratch/$USER/nlp-final-project/results
    ln -d -s /scratch/$USER/nlp-final-project/results results
6. Make `models` folder at `/scratch` to log there (more space)
    ```shell
    mkdir -p /scratch/$USER/nlp-final-project/models
    ln -d -s /scratch/$USER/nlp-final-project/models models
    ```
   
#### Each session setup 

Ether run sbatch jobs only or setup modules (see above) and activate venv. 
   
#### Habrok run

0. Copy files to the server, e.g:
   ```shell
   scp -r ./ habrok:~/nlp-final-project
   ```
   Or setup PyCharm deployment (settings -> deployment -> add new -> SFTP)
1. Run job script 
    ```shell
    sbatch scripts/classification/train.sh --base-model=roberta-base [other options]
    ```
2. Monitor status with 
    ```shell
    squeue | grep $USER
    squeue | grep gpu
    squeue -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %.18R %p" | grep gpu
    ```


sbatch scripts/generation/train.sh --base-model=t5-small
sbatch scripts/generation/train.sh --base-model=google/flan-t5-small
sbatch scripts/generation/train.sh --base-model=google/t5-v1_1-small
sbatch scripts/generation/train.sh --base-model=google/byt5-small
