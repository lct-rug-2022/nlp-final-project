## Training

### Colab training 

Just copy specific script folder content to `/content/`: 
```shell
content/
  - params.json
  - train.py
```

Export `NEPTUNE_PROJECT` and `NEPTUNE_API_TOKEN`, login into `HF Hub` via `huggingface_login`


### Peregrine Run

#### peregrine setup (once)

ALL INSIDE REPO 

1. Create `.env` file with `NEPTUNE_PROJECT` and `NEPTUNE_API_TOKEN`
2. Load modules same as in `jobscript` (for baseline it is `single_task.sh`)
3. Create new venv 
    ```shell
    python3 -m venv /data/$USER/.envs/nlp-final-project
    ln -d -s /data/$USER/.envs/nlp-final-project venv
    ```
4. Install libs 
    ```shell
    source /data/$USER/.envs/nlp-final-project/bin/activate
    pip install pygit2 --prefer-binary
    pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
    pip install -U -r requirements.txt
    ```
5. Make `results` folder at `/data` to log there (more space)
    ```shell
    mkdir /data/$USER/nlp-final-project
    mkdir /data/$USER/nlp-final-project/results
    ln -d -s /data/$USER/nlp-final-project/results results
6. Make `models` folder at `/data` to log there (more space)
    ```shell
    mkdir /data/$USER/nlp-final-project/models
    ln -d -s /data/$USER/nlp-final-project/models models
    ln -d -s /scratch/$USER/nlp-final-project/results results
    ```
7. Make alias `.venv` folder at `/data/$USER/.envs/nlp-final-project` 
    ```shell
    ln -d -s /data/$USER/.envs/nlp-final-project .venv
    ln -d -s /data/$USER/.envs/nlp-final-project .venv
    ```
   
#### each session setup 

Run peregrine setup script (with activate venv, load modules and read `.env` file)
`bash training/peregrine-setup.sh`
   
#### peregrine run

1. Run job script 
    ```shell
    sbatch scripts/classification/train.sh --base-model=roberta-base [other options]
    ```
2. Monitor status with 
    ```shell
    squeue | grep $USER
    squeue | grep gpu
    ```


sbatch scripts/generation/train.sh --base-model=t5-small
sbatch scripts/generation/train.sh --base-model=google/flan-t5-small
sbatch scripts/generation/train.sh --base-model=google/t5-v1_1-small
sbatch scripts/generation/train.sh --base-model=google/byt5-small
