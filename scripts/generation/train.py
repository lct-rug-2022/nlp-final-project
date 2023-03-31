import json
import random
from pathlib import Path

import typer
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq, EarlyStoppingCallback
import evaluate
import numpy as np
from torchinfo import summary
import nltk
from transformers.integrations import NeptuneCallback
import neptune.new as neptune
from sklearn.metrics import f1_score


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


ROOT_FOLDER = Path(__file__).parent if Path(__file__).parent.name == 'content' else Path(__file__).parent.parent.parent  # detect colab run
with open(Path(__file__).parent / 'params.json') as f:
    EDOS_EVAL_PARAMS = json.load(f)


# prefer bf16, https://www.reddit.com/r/MachineLearning/comments/vndtn8/d_mixed_precision_training_difference_between/
IS_CUDA_AVAILABLE = torch.cuda.is_available()
IS_BF16_AVAILABLE = IS_CUDA_AVAILABLE and torch.cuda.is_bf16_supported()
IS_FP16_AVAILABLE = IS_CUDA_AVAILABLE and (not IS_BF16_AVAILABLE)
print('IS_CUDA_AVAILABLE', IS_CUDA_AVAILABLE)
print('IS_FP16_AVAILABLE', IS_FP16_AVAILABLE)
print('IS_BF16_AVAILABLE', IS_BF16_AVAILABLE)


nltk.download("punkt", quiet=True)


app = typer.Typer(add_completion=False)


def _load_dataset(tokenizer, generation_type='explanation_only', max_length=512):
    """generation_type: one of 'explanation_only', 'explanation_use_label', 'explanation_use_prompt_label', 'explanation_use_flan_prompt_label', 'label_and_explanation'"""
    dataset = load_dataset('esnli')
    cl = dataset['train'].features['label']
    dataset = dataset.rename_column('label', 'raw_label')

    def _join_with_sep(list_a, list_b):
        # TODO: add new special token instead of using tokenizer.unk_token
        # special_token = tokenizer.additional_special_tokens[0]
        special_token = ':'
        return [f'{h}{special_token} {l}' for h, l in zip(list_a, list_b)]

    def _form_target(list_a, list_b, list_c, list_d, include_label: bool, sep_token = ';', special_token = ':'):
        """Constructs the target sequence for the model.
        In case of training use first explanation, for validation use all explanations connacted with sep.
        """
        if include_label:
            if len(max(list_c)) == 0:  # training
                return [f'{l}{special_token} {e1}' for l, e1 in zip(list_a, list_b)]
            else:
                return [f'{l}{special_token} {e1} {sep_token}{l}{special_token} {e2} {sep_token}{l}{special_token} {e3}'
                        for l, e1, e2, e3 in zip(list_a, list_b, list_c, list_d)]
        else:
            if len(max(list_c)) == 0:  # training
                return list_b
            else:
                return [f'{e1} {sep_token} {e2} {sep_token} {e3}' for e1, e2, e3 in zip(list_b, list_c, list_d)]

    def tokenize_function(examples):
        _text_target = _form_target(cl.int2str(examples['raw_label']), examples['explanation_1'], examples['explanation_2'], examples['explanation_3'], include_label=generation_type=='label_and_explanation')

        if generation_type == 'explanation_only':
            # using "[premise] SEP [hypothesis]" generate "[explanation]"
            examples = tokenizer(examples['premise'], examples['hypothesis'], text_target=_text_target, truncation=True, padding='do_not_pad', max_length=max_length)
        elif generation_type == 'explanation_use_label':
            # using "[premise] SEP [hypothesis] SEP [label]" generate "[explanation]"
            examples = tokenizer(examples['premise'], _join_with_sep(examples['hypothesis'], cl.int2str(examples['raw_label'])), text_target=_text_target, truncation=True, padding='do_not_pad', max_length=max_length)
        elif generation_type == 'explanation_use_prompt_label':
            # using "[premise] SEP [hypothesis] SEP It was [label]" generate "[explanation]"
            examples = tokenizer(examples['premise'], _join_with_sep(examples['hypothesis'], [f'It was {i}.' for i in cl.int2str(examples['raw_label'])]), text_target=_text_target, truncation=True, padding='do_not_pad', max_length=max_length)
        elif generation_type == 'explanation_use_flan_prompt_label':
            # using "[premise] SEP [hypothesis] SEP It is [label]. Give an explanation why." generate "[explanation]"
            examples = tokenizer(examples['premise'], _join_with_sep(examples['hypothesis'], [f'It is {i}. Give an explanation why.' for i in cl.int2str(examples['raw_label'])]), text_target=_text_target, truncation=True, padding='do_not_pad', max_length=max_length)
        elif generation_type == 'label_and_explanation':
            # using "[premise] SEP [hypothesis]" generate "[label] SEP [explanation]"
            examples = tokenizer(examples['premise'], examples['hypothesis'], text_target=_text_target, truncation=True, padding='do_not_pad', max_length=max_length)
        else:
            raise RuntimeError(f'Unknown generation_type="{generation_type}"')
        return examples

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    return tokenized_dataset


def _get_metrics_function(tokenizer, generation_type='explanation_only'):
    """generation_type: one of 'explanation_only', 'explanation_use_label', 'explanation_use_prompt_label', 'explanation_use_flan_prompt_label', 'label_and_explanation'"""
    # metrics = evaluate.combine([
    #     evaluate.load('bertscore', lang='en'),
    #     evaluate.load('rouge', use_stemmer=False),
    #     evaluate.load('bleu'),
    # ])
    metric_f1 = evaluate.load('f1')

    metric_bs = evaluate.load('bertscore')
    metric_rouge = evaluate.load('rouge')
    metric_bleu = evaluate.load('bleu')

    def _compute_metrics(eval_preds, sep_token = ';'):
        preds, labels = eval_preds

        # decode preds and labels
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        if generation_type == 'label_and_explanation':
            # split first token [label] from the other tokens [explanation]
            # TODO: add new special token instead@see tokenize_function)
            # special_token = tokenizer.additional_special_tokens[0]
            special_token = ':'
            class_prods, decoded_preds = zip(*[i.split(special_token, 1) if special_token in i else ('', special_token) for i in decoded_preds])
            class_labels = [i.split(special_token, 1)[0].strip() for i in decoded_labels]
            classification_metrics = {
                'f1': f1_score(class_labels, class_prods, average='macro'),
            }
            # classification_metrics = metric_f1.compute(predictions=class_prods, references=class_labels, awerage='macro')
        else:
            classification_metrics = {'f1': None}

        # rougeLSum expects newline after each sentence
        decoded_preds = ['\n'.join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = [['\n'.join(nltk.sent_tokenize(label.strip())) for label in item_labels.split(sep_token)] for item_labels in decoded_labels]

        result = {
            **classification_metrics,
            'bertscore_f1': np.mean(metric_bs.compute(predictions=decoded_preds, references=decoded_labels, lang='en', use_fast_tokenizer=True)['f1']),
            # 'bleurt': np.mean(metric_bleurt.compute(predictions=decoded_preds, references=decoded_labels, checkpoint='bleurt-tiny-128')['scores']),
            **metric_rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=False),
            **metric_bleu.compute(predictions=decoded_preds, references=decoded_labels),
        }
        return {
            k: result[k]
            for k in ['f1', 'bertscore_f1', 'rouge1', 'rouge2', 'rougeL', 'rougeLsum', 'bleu']
        }

    return _compute_metrics


def _get_trainer_args(params, hub_model_name, output_dir, push_to_hub=False, model_support_fp16=True, resume_from_checkpoint=False):
    return Seq2SeqTrainingArguments(
        output_dir=output_dir,
        report_to='none',

        learning_rate=params['learning_rate'],
        lr_scheduler_type='linear',
        weight_decay=params.get('weight_decay', 0.01),
        optim=params.get('optim', 'adafactor'),

        auto_find_batch_size=False,  # divide by 2 in case of OOM
        per_device_train_batch_size=params['batch_size'],
        per_device_eval_batch_size=params['batch_size'],
        num_train_epochs=params['max_epochs'],
        warmup_ratio=params.get('warmup_ratio', 0.05),

        no_cuda=not IS_CUDA_AVAILABLE,
        fp16=IS_FP16_AVAILABLE and model_support_fp16,  # always use fp16 on gpu, if not a special model
        fp16_full_eval=IS_FP16_AVAILABLE,
        bf16=IS_BF16_AVAILABLE,
        bf16_full_eval=IS_BF16_AVAILABLE,

        logging_strategy='steps',
        logging_steps=params['eval_steps'],
        evaluation_strategy='steps',
        eval_steps=params['eval_steps'],
        save_strategy='steps',
        save_steps=params['eval_steps'],

        metric_for_best_model='eval_loss',
        greater_is_better=False,
        load_best_model_at_end=True,
        save_total_limit=3,

        predict_with_generate=True,
        generation_max_length=256,
        generation_num_beams=1,  # faster evaluation, but test score will be better
        torch_compile=False,  # not working as Tesla T4 for now

        hub_model_id=hub_model_name,
        resume_from_checkpoint=resume_from_checkpoint,
        push_to_hub=push_to_hub,
        hub_strategy='checkpoint',
    )


@app.command()
def main(
        base_model: str = typer.Option('t5-small', help='Pretrained model to finetune: HUB or Path'),
        config_name: str = typer.Option('default', help='Config name to use: see params.json'),
        generation_type: str = typer.Option('explanation_only', help='What we generate'),
        resume_training_id: str = typer.Option(None, help='Neptune tag to resume training from or None'),
        postfix: str = typer.Option('', help='Model name postfix'),
        push_to_hub: bool = typer.Option(False, help='Push model to HuggingFace Hub'),
        save_model: bool = typer.Option(False, help='Save model locally'),
        results_folder: Path = typer.Option(ROOT_FOLDER / 'results', dir_okay=True, writable=True, help='Folder to save results'),
        save_folder: Path = typer.Option(ROOT_FOLDER / 'models', dir_okay=True, writable=True, help='Folder to save trained model'),
):
    clear_base_model = base_model.replace('/', '-')
    model_name_to_save = f'-{clear_base_model}-e-snli-generation-{generation_type}-{config_name}'
    if postfix:
        model_name_to_save += f'{model_name_to_save}-{postfix}'
    output_dir = str(results_folder / model_name_to_save)
    model_save_folder = save_folder / model_name_to_save
    hub_model_name = f'k4black/{model_name_to_save}'

    # load config
    params = EDOS_EVAL_PARAMS[config_name.split('-')[0]]  # read base config
    params.update(EDOS_EVAL_PARAMS[config_name])  # update with specific config
    model_support_fp16 = True and 'flan-t5' not in base_model and 'byt5' not in base_model  # flan-t5 and byt5 do not support fp16
    model_max_length = 512 if 'byt5' not in base_model else 1024
    print('model_support_fp16', model_support_fp16)
    print('model_max_length', model_max_length)

    print('\n', '-' * 32, 'Loading...', '-' * 32, '\n')

    # create neptune run
    neptune_run = neptune.init_run(with_id=resume_training_id, tags=[f'task:{generation_type}', f'model:{base_model}', f'conf:{config_name}'])
    neptune_callback = NeptuneCallback(run=neptune_run)
    neptune_object_id = neptune_run['sys/id'].fetch()
    print('neptune_object_id', neptune_object_id)
    neptune_run['finetuning/parameters'] = {
        'base_model': base_model,
        'config_name': config_name,
        'generation_type': generation_type,
    }

    # load pretrained tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding='longest',
    )

    # load data
    tokenized_dataset = _load_dataset(tokenizer, generation_type=generation_type, max_length=model_max_length)

    # load new pretrained model
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
    summary(model)

    # load metrics
    compute_metrics = _get_metrics_function(tokenizer, generation_type=generation_type)

    # create trainer
    training_args = _get_trainer_args(
        params, hub_model_name, output_dir,
        push_to_hub=push_to_hub, model_support_fp16=model_support_fp16, resume_from_checkpoint=resume_training_id is not None
    )
    trainer = Seq2SeqTrainer(
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

    test_prediction = trainer.predict(tokenized_dataset['test'], num_beams=1)
    print('metrics (n_bins=1)', dict(test_prediction.metrics))
    test_prediction = trainer.predict(tokenized_dataset['test'], num_beams=2)
    print('metrics (n_bins=2)', dict(test_prediction.metrics))
    test_prediction = trainer.predict(tokenized_dataset['test'], num_beams=4)
    print('metrics (n_bins=4)', dict(test_prediction.metrics))

    neptune_callback.run['finetuning/final_metrics'] = dict(test_prediction.metrics)


if __name__ == '__main__':
    app()
