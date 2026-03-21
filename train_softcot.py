import argparse
from tqdm import tqdm

import torch
import pandas as pd

from datasets import Dataset
from transformers import AutoTokenizer
from transformers import TrainingArguments, Trainer
from fastNLP import logger


from data_loader import GSM8KLoader, StrategyQALoader, AugASDivLoader, AQuALoader
from llm_model import EfficientSoftCoTFromSmallModel
from utils import pre_process_strategy_qa, pre_process_gsm8k, pre_process_aqua, CustomDataCollator


args = argparse.ArgumentParser()
args.add_argument('--large_model_id', type=str, default='meta-llama/Llama-3.1-8B-Instruct')
args.add_argument('--small_model_id', type=str, default='meta-llama/Llama-3.2-1B-Instruct')
args.add_argument('--output_name', type=str, required=True)
args.add_argument('--batch_size', type=int, default=1)
args.add_argument('--task_name', type=str, choices=[
    'gsm8k', 'strategyqa', 'asdiv-aug', 'aqua',
])
args.add_argument('--num_thought_tokens', type=int, default=2)
args.add_argument('--n_epochs', type=float, default=3.0)
args.add_argument('--k_shot', type=int, default=0)
args.add_argument('--tune_base_model', action='store_true', default=False)
args.add_argument('--tune_assistant_model', action='store_true', default=False)
arg = args.parse_args()

logger.info(f'args: {arg.__dict__}')

large_model_id = arg.large_model_id
small_model_id = arg.small_model_id
output_name = arg.output_name
batch_size = arg.batch_size
task_name = arg.task_name
n_epochs = arg.n_epochs
num_thought_tokens = arg.num_thought_tokens
k_shot = arg.k_shot
tune_base_model = arg.tune_base_model
tune_assistant_model = arg.tune_assistant_model


large_model_name = large_model_id.split('/')[-1]
small_model_name = small_model_id.split('/')[-1]
post_fix = f'{task_name}-{n_epochs}-{num_thought_tokens}-{large_model_name}-{small_model_name}'
output_dir = f'./results/{output_name}-{post_fix}'
log_dir = f'./logs/{output_name}-{post_fix}'
save_model_dir = f'./ckpt/{output_name}-{post_fix}'

logger.info(f'Output Dir: {output_dir}')
logger.info(f'Log Dir: {log_dir}')
logger.info(f'Save Model Dir: {save_model_dir}')

model_dtype = torch.bfloat16
param_dtype = str(model_dtype)

base_tokenizer = AutoTokenizer.from_pretrained(large_model_id, token='your-huggingface-token')
assistant_tokenizer = AutoTokenizer.from_pretrained(small_model_id, token='your-huggingface-token')

if 'Llama' in large_model_id:
    base_special_token = ['<|end_of_text|>', '<|reserved_special_token_0|>', '<|reserved_special_token_1|>']
    base_backbone = 'llama'
elif 'Qwen' in large_model_id:
    base_special_token = ['<|endoftext|>', '<|box_start|>', '<|box_end|>']
    base_backbone = 'qwen'
else:
    raise NotImplementedError
if 'Llama' in small_model_id:
    assistant_special_token = ['<|end_of_text|>', '<|reserved_special_token_0|>', '<|reserved_special_token_1|>']
    assistant_backbone = 'llama'
elif 'Qwen' in small_model_id:
    assistant_special_token = ['<|endoftext|>', '<|box_start|>', '<|box_end|>']
    assistant_backbone = 'qwen'
else:
    raise NotImplementedError

model = EfficientSoftCoTFromSmallModel(
    small_model_id,
    large_model_id,
    num_thought_tokens,
    tune_base_model=tune_base_model,
    tune_assistant_model=tune_assistant_model,
)

logger.info(f'Successfully Init Model `{model.__class__.__name__}`')

trainable_param = 0
total_param = 0
for n, p in model.named_parameters():
    if p.requires_grad:
        trainable_param += p.view(-1).size(0)
    total_param += p.view(-1).size(0)
logger.info(f'Trainable Parameters: {trainable_param}; Total Parameters: {total_param}')

if task_name in ['gsm8k']:
    db = GSM8KLoader().load()
    preprocess_method = pre_process_gsm8k
elif task_name in ['strategyqa']:
    db = StrategyQALoader().load()
    preprocess_method = pre_process_strategy_qa
elif task_name in ['asdiv-aug']:
    db = AugASDivLoader().load()
    preprocess_method = pre_process_gsm8k
elif task_name in ['aqua']:
    db = AQuALoader().load()
    preprocess_method = pre_process_aqua
else:
    raise NotImplementedError

train_dataset = db.get_dataset('train')
eval_dataset = db.get_dataset('dev')

if k_shot > 0:
    train_dataset = train_dataset[: k_shot]

train_rows = []
for ins in tqdm(train_dataset, desc='Preprocess Training Set'):
    train_rows.append(
        preprocess_method(
            ins, base_tokenizer, assistant_tokenizer, num_thought_tokens,
            add_bot_eot=True, split='train',
            base_special_token=base_special_token,
            assistant_special_token=assistant_special_token,
            base_backbone=base_backbone,
            assistant_backbone=assistant_backbone,
        )
    )

eval_rows = []
for ins in tqdm(eval_dataset, desc='Preprocess Testing Set'):
    eval_rows.append(
        preprocess_method(
            ins, base_tokenizer, assistant_tokenizer, num_thought_tokens,
            add_bot_eot=True, split='dev',
            base_special_token=base_special_token,
            assistant_special_token=assistant_special_token,
            base_backbone=base_backbone,
            assistant_backbone=assistant_backbone,
        )
    )

train_data = Dataset.from_pandas(pd.DataFrame(train_rows))
eval_data = Dataset.from_pandas(pd.DataFrame(eval_rows))

training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    eval_strategy='epoch',
    save_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=n_epochs,
    save_total_limit=10 if task_name in ['gsm8k', 'aqua'] else 2,
    bf16=True,
    logging_dir=log_dir,
    logging_steps=500,
    remove_unused_columns=True,
    save_safetensors=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=eval_data,
    data_collator=CustomDataCollator(),
)
trainer.train()

model.save_pretrained(save_model_dir)
logger.info(f'Finish training, save model to dir `{save_model_dir}`')


