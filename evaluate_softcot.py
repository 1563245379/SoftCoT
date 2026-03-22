
import os
import re
import argparse

from tqdm import tqdm
import torch
import torch.nn.functional as F


from transformers import AutoTokenizer, GenerationConfig
from fastNLP import logger

from llm_model import EfficientSoftCoTFromSmallModel
from data_loader import GSM8KLoader, StrategyQALoader, AugASDivLoader, AQuALoader, DULoader
from utils import pre_process_gsm8k, pre_process_strategy_qa, pre_process_aqua, pre_process_du


def downsample_attention_matrix(attention_matrix, resolution):
    return F.interpolate(
        attention_matrix.unsqueeze(0).unsqueeze(0).to(dtype=torch.float32),
        size=(resolution, resolution),
        mode='bilinear',
        align_corners=False,
    ).squeeze(0).squeeze(0)


def downsample_position_mask(position_mask, resolution):
    return F.interpolate(
        position_mask.view(1, 1, -1).to(dtype=torch.float32),
        size=resolution,
        mode='linear',
        align_corners=False,
    ).view(-1)


def extract_last_layer_attention(model, inputs_embeds, attention_mask):
    with torch.no_grad():
        outputs = model.base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_attentions=True,
            return_dict=True,
            use_cache=False,
        )
    if outputs.attentions is None or outputs.attentions[-1] is None:
        raise RuntimeError('Base model does not return attentions for visualization.')

    last_layer_attention = outputs.attentions[-1].mean(dim=1)[0]
    valid_seq_len = int(attention_mask[0].sum().item())
    return last_layer_attention[:valid_seq_len, :valid_seq_len].detach().float().cpu()


def locate_thought_region(average_thought_mask):
    if average_thought_mask is None:
        return None

    max_value = average_thought_mask.max().item()
    if max_value <= 0:
        return None

    threshold = max(0.05, max_value * 0.5)
    thought_bins = torch.nonzero(average_thought_mask >= threshold, as_tuple=False).view(-1)
    if thought_bins.numel() == 0:
        peak_idx = int(torch.argmax(average_thought_mask).item())
        return peak_idx, peak_idx + 1

    return int(thought_bins[0].item()), int(thought_bins[-1].item()) + 1


def resolve_attention_output_paths(output_path, task_name, base_model_name, assistant_model_name, seed, resolution):
    file_stem = (
        f'{task_name}-seed_{seed}-res_{resolution}-{base_model_name}-{assistant_model_name}-avg-last-layer-attention'
    )
    if output_path not in [None, '', 'None']:
        if output_path.endswith('.png'):
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            heatmap_path = output_path
        else:
            os.makedirs(output_path, exist_ok=True)
            heatmap_path = os.path.join(output_path, f'{file_stem}.png')
    else:
        output_dir = './attention_visualizations'
        os.makedirs(output_dir, exist_ok=True)
        heatmap_path = os.path.join(output_dir, f'{file_stem}.png')

    stats_path = os.path.splitext(heatmap_path)[0] + '.txt'
    return heatmap_path, stats_path


def save_attention_heatmap(
    average_attention_matrix,
    thought_region,
    heatmap_path,
    title,
):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))
    heatmap = ax.imshow(average_attention_matrix.tolist(), cmap='magma', origin='lower', aspect='equal')
    colorbar = fig.colorbar(heatmap, ax=ax)
    colorbar.set_label('Average attention score')

    if thought_region is not None:
        thought_start, thought_end = thought_region
        ax.axvspan(thought_start - 0.5, thought_end - 0.5, color='cyan', alpha=0.12)
        ax.axhspan(thought_start - 0.5, thought_end - 0.5, color='cyan', alpha=0.12)
        ax.axvline(thought_start - 0.5, color='cyan', linestyle='--', linewidth=1.2)
        ax.axvline(thought_end - 0.5, color='cyan', linestyle='--', linewidth=1.2)
        ax.axhline(thought_start - 0.5, color='cyan', linestyle='--', linewidth=1.2)
        ax.axhline(thought_end - 0.5, color='cyan', linestyle='--', linewidth=1.2)
        ax.text(
            thought_start,
            average_attention_matrix.size(0) - 1,
            'thought tokens',
            color='cyan',
            fontsize=10,
            ha='left',
            va='top',
        )

    ax.set_xlabel('Key positions (downsampled)')
    ax.set_ylabel('Query positions (downsampled)')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(heatmap_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def save_attention_stats(
    stats_path,
    heatmap_path,
    attention_sample_count,
    attention_vis_resolution,
    thought_region,
    average_thought_attention_received,
    average_thought_attention_emitted,
    average_thought_self_attention,
):
    thought_region_text = 'None' if thought_region is None else f'[{thought_region[0]}, {thought_region[1]})'
    with open(stats_path, 'w', encoding='utf-8') as stats_file:
        stats_file.write(f'heatmap_path: {heatmap_path}\n')
        stats_file.write(f'attention_sample_count: {attention_sample_count}\n')
        stats_file.write(f'attention_resolution: {attention_vis_resolution}\n')
        stats_file.write(f'thought_region_bins: {thought_region_text}\n')
        stats_file.write(f'average_thought_attention_received: {average_thought_attention_received:.6f}\n')
        stats_file.write(f'average_thought_attention_emitted: {average_thought_attention_emitted:.6f}\n')
        stats_file.write(f'average_thought_self_attention: {average_thought_self_attention:.6f}\n')


args = argparse.ArgumentParser()
args.add_argument('--base_model_id', type=str, default='meta-llama/Llama-3.1-8B-Instruct')
args.add_argument('--assistant_model_id', type=str, default='meta-llama/Llama-3.2-1B-Instruct')
args.add_argument('--params_file_name', type=str, default=None)
args.add_argument('--base_model_ckpt', type=str, default=None)
args.add_argument('--assistant_model_ckpt', type=str, default=None)
args.add_argument('--num_thought_tokens', type=int, default=2)
args.add_argument('--num_return_sequences', type=int, default=1)
args.add_argument('--task_name', type=str, choices=[
    'gsm8k', 'strategyqa', 'asdiv-aug', 'aqua', 'du'
])
args.add_argument('--print_input', action='store_true', default=False)
args.add_argument('--print_response', action='store_true', default=False)
args.add_argument('--print_thought_token_topk', action='store_true', default=False)
args.add_argument('--visualize_attention_matrix', action='store_true', default=False)
args.add_argument('--attention_vis_resolution', type=int, default=64)
args.add_argument('--attention_vis_output', type=str, default=None)
args.add_argument('--test_k', type=int, default=0)
args.add_argument('--seed', type=int, default=42)
args.add_argument('--tune_base_model', action='store_true', default=False)
args.add_argument('--tune_assistant_model', action='store_true', default=False)
arg = args.parse_args()
logger.info(f'Args: {arg.__dict__}')

base_model_id = arg.base_model_id
assistant_model_id = arg.assistant_model_id
params_file_name = arg.params_file_name
base_model_ckpt = arg.base_model_ckpt
assistant_model_ckpt = arg.assistant_model_ckpt
num_thought_tokens = arg.num_thought_tokens
num_return_sequences = arg.num_return_sequences
task_name = arg.task_name
print_input = arg.print_input
print_response = arg.print_response
print_thought_token_topk = arg.print_thought_token_topk
visualize_attention_matrix = arg.visualize_attention_matrix
attention_vis_resolution = arg.attention_vis_resolution
attention_vis_output = arg.attention_vis_output
test_k = arg.test_k
seed = arg.seed
tune_base_model = arg.tune_base_model
tune_assistant_model = arg.tune_assistant_model

if attention_vis_resolution <= 0:
    raise ValueError('`attention_vis_resolution` must be a positive integer.')

large_model_name = base_model_id.split('/')[-1]
small_model_name = assistant_model_id.split('/')[-1]

if base_model_ckpt in ['None']:
    base_model_ckpt = None
if assistant_model_ckpt in ['None']:
    assistant_model_ckpt = None

model_dtype = torch.bfloat16
param_dtype = str(model_dtype)

base_tokenizer = AutoTokenizer.from_pretrained(base_model_id, token='your-huggingface-token')
assistant_tokenizer = AutoTokenizer.from_pretrained(assistant_model_id, token='your-huggingface-token')

if 'Llama' in base_model_id:
    base_special_token = ['<|end_of_text|>', '<|reserved_special_token_0|>', '<|reserved_special_token_1|>']
    base_backbone = 'llama'
elif 'Qwen' in base_model_id:
    base_special_token = ['<|endoftext|>', '<|box_start|>', '<|box_end|>']
    # generation_config.pad_token_id = 151643
    base_backbone = 'qwen'
else:
    raise NotImplementedError
if 'Llama' in assistant_model_id:
    assistant_special_token = ['<|end_of_text|>', '<|reserved_special_token_0|>', '<|reserved_special_token_1|>']
    assistant_backbone = 'llama'
elif 'Qwen' in assistant_model_id:
    assistant_special_token = ['<|endoftext|>', '<|box_start|>', '<|box_end|>']
    assistant_backbone = 'qwen'
else:
    raise NotImplementedError

model = EfficientSoftCoTFromSmallModel(
    assistant_model_id,
    base_model_id,
    num_thought_tokens,
    tune_base_model=tune_base_model,
    tune_assistant_model=tune_assistant_model,
    path_to_projection_module=params_file_name,
    path_to_small_language_model=assistant_model_ckpt,
)
logger.info(f'Successfully Init Model `{model.__class__.__name__}`')
model.eval()
model.assistant_model.eval()
model.base_model.eval()

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
elif task_name in ['du']:
    db = DULoader().load()
    preprocess_method = pre_process_du
else:
    raise NotImplementedError

ds = db.get_dataset('test')

if test_k > 0:
    ds = ds[: test_k]

generation_config = GenerationConfig.from_pretrained(base_model_id)
if base_backbone in ['llama']:
    generation_config.pad_token_id = 128009
elif base_backbone in ['qwen']:
    generation_config.pad_token_id = 151643
else:
    raise NotImplementedError
generation_config.top_p = 1.0
generation_config.temperature = 1.0

correct_count = 0
average_attention_sum = None
average_thought_mask_sum = None
attention_sample_count = 0
thought_attention_received_sum = 0.0
thought_attention_emitted_sum = 0.0
thought_self_attention_sum = 0.0
for idx, ins in enumerate(tqdm(ds)):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if task_name in ['gsm8k', 'asdiv-aug', 'aqua']:
        answer = ins['answer'].split('\n')[-1]
        assert answer.startswith('####')
        answer = answer.replace(',', '')
        if task_name in ['gsm8k', 'asdiv-aug']:
            if '.' in answer:
                answer = float(answer[4:])
            else:
                answer = int(answer[4:])
        else:
            answer = answer[4:].strip()
    elif task_name in ['strategyqa', 'du']:
        answer = ins['answer']
    else:
        raise NotImplementedError

    logger.info(f'Ground Truth Answer: {answer}')

    inputs = preprocess_method(
        ins, base_tokenizer, assistant_tokenizer, num_thought_tokens,
        add_bot_eot=(num_thought_tokens > 0), split='test',
        base_special_token=base_special_token,
        assistant_special_token=assistant_special_token,
        base_backbone=base_backbone,
        assistant_backbone=assistant_backbone,
        device=model.device,
    )
    if print_input:
        logger.info(f'Raw Inputs for Base Model: {base_tokenizer.decode(inputs["input_ids"][0])}')
        # logger.info(f'Raw Inputs for Assistant Model: {assistant_tokenizer.decode(inputs["assistant_input_ids"][0])}')

    terminators = [
        base_tokenizer.eos_token_id,
    ]
    if base_backbone in ['llama']:
        terminators.append(base_tokenizer.convert_tokens_to_ids("<|eot_id|>"))

    model_answer_list = []
    model_answer_count = {}

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    inputs_embeds = model.base_model.get_input_embeddings()(inputs['input_ids'])

    torch.manual_seed(41)
    torch.cuda.manual_seed(41)

    inputs_embeds = model.get_inputs_embeds_for_base_model(
        inputs['assistant_input_ids'],
        inputs['assistant_attention_mask'],
        inputs['input_ids'],
        inputs_embeds,
        inputs['thought_index'],
        print_input,
        print_thought_token_topk,
    )

    if visualize_attention_matrix:
        last_layer_attention = extract_last_layer_attention(
            model=model,
            inputs_embeds=inputs_embeds,
            attention_mask=inputs['attention_mask'],
        )
        downsampled_attention = downsample_attention_matrix(last_layer_attention, attention_vis_resolution)
        if average_attention_sum is None:
            average_attention_sum = torch.zeros_like(downsampled_attention)
            average_thought_mask_sum = torch.zeros(attention_vis_resolution, dtype=torch.float32)
        average_attention_sum += downsampled_attention

        thought_start_idx = inputs['thought_index'][0, 0].item()
        thought_end_idx = inputs['thought_index'][0, 1].item()
        if thought_end_idx > thought_start_idx:
            thought_position_mask = torch.zeros(last_layer_attention.size(0), dtype=torch.float32)
            thought_position_mask[thought_start_idx: thought_end_idx] = 1.0
            average_thought_mask_sum += downsample_position_mask(thought_position_mask, attention_vis_resolution)
            thought_attention_received_sum += last_layer_attention[:, thought_start_idx: thought_end_idx].mean().item()
            thought_attention_emitted_sum += last_layer_attention[thought_start_idx: thought_end_idx, :].mean().item()
            thought_self_attention_sum += last_layer_attention[
                thought_start_idx: thought_end_idx, thought_start_idx: thought_end_idx
            ].mean().item()
        attention_sample_count += 1

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    outputs = model.base_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=inputs['attention_mask'],
        max_new_tokens=1024,
        eos_token_id=terminators,
        do_sample=True,
        generation_config=generation_config,
        num_return_sequences=num_return_sequences,
    )

    for i in range(outputs.shape[0]):
        # response = outputs[i][inputs['input_ids'].shape[-1]:]
        response = outputs[i]
        raw_model_answer = base_tokenizer.decode(response, skip_special_tokens=True)

        if print_response:
            logger.info(f'Answer ({idx + 1}-{i + 1}/{len(ds)}): {base_tokenizer.decode(response)}<|end-of-response|>')

        if task_name in ['gsm8k', 'asdiv-aug']:
            cleaned_model_answer = raw_model_answer.replace(',', '')
            cleaned_model_answer = cleaned_model_answer.replace('%', '')
            cleaned_model_answer = cleaned_model_answer.replace('$', '')
        else:
            cleaned_model_answer = raw_model_answer

        match = re.findall(r'\s*([\d,]+(?:\.\d+)?)\s*', cleaned_model_answer)

        if task_name in ['gsm8k', 'asdiv-aug']:
            try:
                if match:
                    last_match = match[-1]
                    cleaned_match = last_match.replace(',', '')
                    cleaned_match = cleaned_match.replace('%', '')
                    cleaned_match = cleaned_match.replace('$', '')
                    if '.' in cleaned_match:
                        model_answer = round(float(cleaned_match), 2)
                    else:
                        model_answer = int(cleaned_match)
                else:
                    model_answer = None
                if model_answer is None and not print_response:
                    logger.info(f'None Model Answer ({idx + 1}-{i + 1}/{len(ds)}): {base_tokenizer.decode(response)}')
            except Exception as e:
                model_answer = None
                logger.error(f'Error: {e}')
        elif task_name in ['strategyqa']:
            last_yes = re.search(r'\bsey\b', raw_model_answer.lower()[::-1])
            if last_yes is not None:
                last_yes = last_yes.start()
            else:
                last_yes = len(raw_model_answer)
            last_no = re.search(r'\bon\b', raw_model_answer.lower()[::-1])
            if last_no is not None:
                last_no = last_no.start()
            else:
                last_no = len(raw_model_answer)
            if last_yes == last_no == len(raw_model_answer):
                model_answer = None
            else:
                model_answer = last_yes < last_no
        elif task_name in ['aqua', 'du']:
            m_answer = re.search(r'\b[a-f]\b', raw_model_answer.lower()[::-1])
            if m_answer is not None:
                model_answer = m_answer.group(0).upper()
            else:
                model_answer = None
        else:
            raise NotImplementedError

        model_answer_list.append(model_answer)
        if model_answer in model_answer_count and model_answer is not None:
            model_answer_count[model_answer] += 1
        else:
            model_answer_count[model_answer] = 1

    max_model_count = 0
    final_model_answer = None

    for k, v in model_answer_count.items():
        if v > max_model_count:
            final_model_answer = k
            max_model_count = v

    logger.info(f'Ground Truth Answer: {answer}')
    logger.info(f'Model Answer: {final_model_answer}')
    is_correct = (final_model_answer == answer)
    logger.info(f'Is Correct: {is_correct}')
    if is_correct:
        correct_count += 1
    logger.info(f'Correct Count: {correct_count}/{idx + 1}')
    logger.info(f'{"-" * 20}')

if visualize_attention_matrix:
    if attention_sample_count == 0:
        logger.warning('Attention visualization is enabled, but no samples were processed.')
    else:
        average_attention_matrix = average_attention_sum / attention_sample_count
        average_thought_mask = average_thought_mask_sum / attention_sample_count
        thought_region = locate_thought_region(average_thought_mask)
        heatmap_path, stats_path = resolve_attention_output_paths(
            attention_vis_output,
            task_name,
            large_model_name,
            small_model_name,
            seed,
            attention_vis_resolution,
        )

        average_thought_attention_received = thought_attention_received_sum / attention_sample_count
        average_thought_attention_emitted = thought_attention_emitted_sum / attention_sample_count
        average_thought_self_attention = thought_self_attention_sum / attention_sample_count

        save_attention_heatmap(
            average_attention_matrix=average_attention_matrix,
            thought_region=thought_region,
            heatmap_path=heatmap_path,
            title=(
                f'Average Last-Layer Attention Heatmap\n'
                f'{task_name} | seed={seed} | res={attention_vis_resolution}'
            ),
        )
        save_attention_stats(
            stats_path=stats_path,
            heatmap_path=heatmap_path,
            attention_sample_count=attention_sample_count,
            attention_vis_resolution=attention_vis_resolution,
            thought_region=thought_region,
            average_thought_attention_received=average_thought_attention_received,
            average_thought_attention_emitted=average_thought_attention_emitted,
            average_thought_self_attention=average_thought_self_attention,
        )

        logger.info(f'Saved average attention heatmap to `{heatmap_path}`')
        logger.info(f'Saved attention summary to `{stats_path}`')
        logger.info(f'Average thought attention received: {average_thought_attention_received:.6f}')
        logger.info(f'Average thought attention emitted: {average_thought_attention_emitted:.6f}')
        logger.info(f'Average thought self attention: {average_thought_self_attention:.6f}')
