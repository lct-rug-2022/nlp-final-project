#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --partition=gpushort
#SBATCH --gpus-per-node=a100.20gb:1
#SBATCH --job-name=esnli
#SBATCH --mem=20G


module purge
module load CUDA/11.7.0
module load cuDNN/8.4.1.50-CUDA-11.7.0
module load NCCL/2.12.12-GCCcore-11.3.0-CUDA-11.7.0
module load Python/3.10.4-GCCcore-11.3.0
module load GCC/11.3.0


source .venv/bin/activate


export TOKENIZERS_PARALLELISM=false


# nli -> explanation_use_prompt_label
echo "+++++++++++++++++ nli -> explanation_use_prompt_label +++++++++++++++++"
echo "-------------- nli:"
python scripts/pipeline/predict_classification.py k4black/roberta-large-e-snli-classification-nli-base --classification-type=nli
echo "-------------- explanation_use_prompt_label:"
python scripts/pipeline/predict_generation.py k4black/google-flan-t5-small-e-snli-generation-explanation_use_prompt_label-selected-b64 --generation-type=explanation_use_prompt_label --use-label-from=k4black-roberta-large-e-snli-classification-nli-base.csv

# explanation_only -> nli_use_explanation
echo "+++++++++++++++++ explanation_only -> nli_use_explanation +++++++++++++++++"
echo "-------------- explanation_only:"
python scripts/pipeline/predict_generation.py k4black/t5-small-e-snli-generation-explanation_only-selected-b64 --generation-type=explanation_only
echo "-------------- nli_use_explanation:"
python scripts/pipeline/predict_classification.py k4black/roberta-base-e-snli-classification-nli_explanation-base --classification-type=nli_explanation --use-explanation-from=k4black-t5-small-e-snli-generation-explanation_only-selected-b64.csv


deactivate
