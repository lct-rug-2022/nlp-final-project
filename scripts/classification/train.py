import json
import random
from pathlib import Path

import pandas as pd
import typer
from datasets import load_dataset, ClassLabel
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, EarlyStoppingCallback
import evaluate
import numpy as np
from torchinfo import summary
import nltk
from transformers.integrations import NeptuneCallback
import neptune.new as neptune


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


ROOT_FOLDER = Path(__file__).parent if Path(__file__).parent.name == 'content' else Path(__file__).parent.parent.parent  # detect colab run
with open(Path(__file__).parent / 'params.json') as f:
    EDOS_EVAL_PARAMS = json.load(f)

IS_CUDA_AVAILABLE = torch.cuda.is_available()
IS_BF16_AVAILABLE = IS_CUDA_AVAILABLE and torch.cuda.is_bf16_supported()
print('IS_CUDA_AVAILABLE', IS_CUDA_AVAILABLE)
print('IS_BF16_AVAILABLE', IS_BF16_AVAILABLE)


nltk.download("punkt", quiet=True)


app = typer.Typer(add_completion=False)


def _load_dataset(tokenizer, classification_type='nli'):
    """
    :param tokenizer:
    :param classification_type: one of 'nli', 'nli_explanation', TODO
    :return:
    """
    cl = ClassLabel(names=['entailment', 'neutral', 'contradiction'])
    label2id, id2label = {n: i for i, n in enumerate(cl.names)}, {i: n for i, n in enumerate(cl.names)}

    dataset = load_dataset('esnli')
    dataset = dataset.cast_column('label', cl)

    def tokenize_function(examples):
        if classification_type == 'nli':
            examples = tokenizer(examples['premise'], examples['hypothesis'], truncation=True, padding='do_not_pad')
        elif classification_type == 'nli_explanation':
            raise NotImplementedError()
        else:
            raise RuntimeError(f'Unknown classification_type="{classification_type}"')
        return examples

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    return tokenized_dataset, label2id, id2label


def _get_metrics_function(tokenizer):
    # metrics = evaluate.combine([
    #     evaluate.load('f1', average='macro'),
    #     evaluate.load('accuracy'),
    # ])
    metric_f1 = evaluate.load('f1')
    metric_acc = evaluate.load('accuracy')

    def _compute_metrics(eval_preds):
        preds, labels = eval_preds
        predictions = np.argmax(preds, axis=-1)
        return {
            **metric_f1.compute(predictions=predictions, references=labels, average='macro'),
            **metric_acc.compute(predictions=predictions, references=labels),
        }

    return _compute_metrics


def _get_trainer_args(params, hub_model_name, output_dir, push_to_hub=False, model_support_fp16=True, resume_from_checkpoint=False):
    return TrainingArguments(
        output_dir=output_dir,
        report_to='none',

        learning_rate=params['learning_rate'],
        lr_scheduler_type='linear',
        weight_decay=params.get('weight_decay', 0.01),
        optim=params.get('optim', 'adamw_torch'),

        auto_find_batch_size=True,  # divide by 2 in case of OOM
        per_device_train_batch_size=params['batch_size'],
        per_device_eval_batch_size=params['batch_size'],
        num_train_epochs=params['max_epochs'],
        warmup_ratio=params.get('warmup_ratio', 0.05),

        no_cuda=not IS_CUDA_AVAILABLE,
        fp16=IS_CUDA_AVAILABLE and model_support_fp16,  # always use fp16 on gpu, if not a special model
        fp16_full_eval=IS_CUDA_AVAILABLE,
        bf16=IS_BF16_AVAILABLE,

        logging_strategy='steps',
        logging_steps=params['eval_steps'],
        evaluation_strategy='steps',
        eval_steps=params['eval_steps'],
        save_strategy='steps',
        save_steps=params['eval_steps'],

        metric_for_best_model='f1',
        greater_is_better=True,
        load_best_model_at_end=True,
        save_total_limit=1,

        torch_compile=False,  # not working as Tesla T4 for now

        hub_model_id=hub_model_name,
        resume_from_checkpoint=resume_from_checkpoint,
        push_to_hub=push_to_hub,
        hub_strategy='checkpoint',
    )


@app.command()
def main(
        base_model: str = typer.Option('roberta-base', help='Pretrained model to finetune: HUB or Path'),
        config_name: str = typer.Option('default', help='Config name to use: see params.json'),
        classification_type: str = typer.Option('nli', help='What we classify'),
        resume_training_id: str = typer.Option(None, help='Neptune tag to resume training from or None'),
        postfix: str = typer.Option('', help='Model name postfix'),
        push_to_hub: bool = typer.Option(False, help='Push model to HuggingFace Hub'),
        save_model: bool = typer.Option(False, help='Save model locally'),
        results_folder: Path = typer.Option(ROOT_FOLDER / 'results', dir_okay=True, writable=True, help='Folder to save results'),
        save_folder: Path = typer.Option(ROOT_FOLDER / 'models', dir_okay=True, writable=True, help='Folder to save trained model'),
):
    clear_base_model = base_model.replace('/', '-')
    model_name_to_save = f'{clear_base_model}-e-snli-classification-{classification_type}-{config_name}'
    if postfix:
        model_name_to_save += f'{model_name_to_save}-{postfix}'
    output_dir = str(results_folder / model_name_to_save)
    model_save_folder = save_folder / model_name_to_save
    hub_model_name = f'k4black/{model_name_to_save}'

    # load config
    params = EDOS_EVAL_PARAMS[config_name.split('-')[0]]  # read base config
    params.update(EDOS_EVAL_PARAMS[config_name])  # update with specific config
    model_support_fp16 = True
    print('model_support_fp16', model_support_fp16)

    print('\n', '-' * 32, 'Loading...', '-' * 32, '\n')

    # create neptune run
    neptune_run = neptune.init_run(with_id=resume_training_id, tags=[f'task:{classification_type}', f'model:{base_model}', f'conf:{config_name}'])
    neptune_callback = NeptuneCallback(run=neptune_run)
    neptune_object_id = neptune_run['sys/id'].fetch()
    print('neptune_object_id', neptune_object_id)

    # load pretrained tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding='longest',
    )

    # load data
    tokenized_dataset, label2id, id2label = _load_dataset(tokenizer, classification_type=classification_type)

    # load new pretrained model
    config = AutoConfig.from_pretrained(base_model, label2id=label2id, id2label=id2label)
    model = AutoModelForSequenceClassification.from_pretrained(base_model, config=config)
    summary(model)

    # load metrics
    compute_metrics = _get_metrics_function(tokenizer)

    # create trainer
    training_args = _get_trainer_args(
        params, hub_model_name, output_dir,
        push_to_hub=push_to_hub, model_support_fp16=model_support_fp16, resume_from_checkpoint=resume_training_id is not None
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=params.get('early_stopping_patience', 5)),
            neptune_callback,
        ],
    )

    print('\n', '-' * 32, 'Training...', '-' * 32, '\n')

    # train itself
    trainer.train(resume_from_checkpoint=resume_training_id is not None)

    # save model
    if save_model:
        if model_save_folder:
            model_save_folder.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(model_save_folder))

    print('\n', '-' * 32, 'End', '-' * 32, '\n')

    test_prediction = trainer.predict(tokenized_dataset['test'])

    neptune_callback.run['finetuning/parameters'] = {
        'base_model': base_model,
        'config_name': config_name,
        'classification_type': classification_type,
    }
    neptune_callback.run['finetuning/final_metrics'] = dict(test_prediction.metrics)


if __name__ == '__main__':
    app()
