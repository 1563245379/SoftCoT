#!/usr/bin/env bash

# Default variables
base_model_id="meta-llama/Llama-3.1-8B-Instruct"
assistant_model_id="meta-llama/Llama-3.2-1B-Instruct"
params_file_name="None"
base_model_ckpt="None"
assistant_model_ckpt="None"
num_thought_tokens=0
num_return_sequences=1
task_name="gsm8k"
seed_from=41
seed_to=45
test_k=0
tune_assistant_model=false
print_input=false
print_response=false
print_thought_token_topk=false
visualize_attention_matrix=false
attention_vis_resolution=64
attention_vis_output=""
log_dir="inference_logs"
run_name=""

# Argument parsing
while [[ $# -gt 0 ]]; do
    case ${1} in
        --base_model_id) base_model_id="${2}"; shift 2 ;;
        --assistant_model_id) assistant_model_id="${2}"; shift 2 ;;
        --params_file_name) params_file_name="${2}"; shift 2 ;;
        --assistant_model_ckpt) assistant_model_ckpt="${2}"; shift 2 ;;
        --base_model_ckpt) base_model_ckpt="${2}"; shift 2 ;;
        --num_thought_tokens) num_thought_tokens="${2}"; shift 2 ;;
        --num_return_sequences) num_return_sequences="${2}"; shift 2 ;;
        --task_name) task_name="${2}"; shift 2 ;;
        --seed_from) seed_from="${2}"; shift 2 ;;
        --seed_to) seed_to="${2}"; shift 2 ;;
        --test_k) test_k="${2}"; shift 2 ;;
        --tune_assistant_model) tune_assistant_model=true; shift ;;
        --print_input) print_input=true; shift ;;
        --print_response) print_response=true; shift ;;
        --print_thought_token_topk) print_thought_token_topk=true; shift ;;
        --visualize_attention_matrix) visualize_attention_matrix=true; shift ;;
        --attention_vis_resolution) attention_vis_resolution="${2}"; shift 2 ;;
        --attention_vis_output) attention_vis_output="${2}"; shift 2 ;;
        --log_dir) log_dir="${2}"; shift 2 ;;
        --run_name) run_name="${2}"; shift 2 ;;
        *) echo "Unknown argument: ${1}"; shift ;;
    esac
done

# Display configuration
echo "Script started at: $(date)"
echo "Base Model ID: ${base_model_id}"
echo "Assistant Model ID: ${assistant_model_id}"
echo "Task Name: ${task_name}"
echo "Num Thought Tokens: ${num_thought_tokens}"
echo "Num Return Sequences: ${num_return_sequences}"
echo "Seed Range: ${seed_from} to ${seed_to}"
echo "Test K: ${test_k}"
echo "Tune Assistant Model: ${tune_assistant_model}"
echo "Print Input: ${print_input}"
echo "Print Response: ${print_response}"
echo "Print Thought Token TopK: ${print_thought_token_topk}"
echo "Visualize Attention Matrix: ${visualize_attention_matrix}"
echo "Attention Visualization Resolution: ${attention_vis_resolution}"
echo "Attention Visualization Output: ${attention_vis_output}"
echo "Logs will be saved in: ${log_dir}"
echo "Run Name is : <|start|>${run_name}<|end|>"
echo "----------------------------------------"

base_model_name="${base_model_id#*/}"
assistant_model_name="${assistant_model_id#*/}"

# Ensure log directory exists
mkdir -p "${log_dir}"

# Check if the Python script exists
if [[ ! -f "evaluate_softcot.py" ]]; then
    echo "Error: evaluate_softcot.py not found!"
    exit 1
fi

# Loop over seed range
for ((seed=seed_from; seed<=seed_to; seed++)); do
    start_time=$(date +%s)
    echo "Running evaluation for seed: ${seed} at $(date)"

    log_file_name="${log_dir}/${run_name}${task_name}-${model_name}-seed_${seed}-ntt_${num_thought_tokens}-nrs_${num_return_sequences}-${base_model_name}-${assistant_model_name}.log"

    cmd="python evaluate_softcot.py \
--base_model_id \"${base_model_id}\" \
--assistant_model_id \"${assistant_model_id}\" \
--params_file_name \"${params_file_name}\" \
--assistant_model_ckpt \"${assistant_model_ckpt}\" \
--base_model_ckpt \"${base_model_ckpt}\" \
--num_thought_tokens ${num_thought_tokens} \
--num_return_sequences ${num_return_sequences} \
--task_name \"${task_name}\" \
--seed ${seed} \
--test_k ${test_k} \
--attention_vis_resolution ${attention_vis_resolution}"

    # Add optional flags
    ${tune_assistant_model} && cmd+=" --tune_assistant_model"
    ${print_input} && cmd+=" --print_input"
    ${print_response} && cmd+=" --print_response"
    ${print_thought_token_topk} && cmd+=" --print_thought_token_topk"
    ${visualize_attention_matrix} && cmd+=" --visualize_attention_matrix"
    [[ -n "${attention_vis_output}" ]] && cmd+=" --attention_vis_output \"${attention_vis_output}\""

    # Run the command and redirect output
    echo "${cmd} > \"${log_file_name}\""
    eval "${cmd} > \"${log_file_name}\""

    end_time=$(date +%s)
    elapsed_time=$((end_time - start_time))

    echo "Evaluation for seed ${seed} finished at: $(date)"
    echo "Elapsed time: ${elapsed_time} seconds"
    echo "========================================"
done

# Display the script end time
echo "Script finished at: $(date)"
