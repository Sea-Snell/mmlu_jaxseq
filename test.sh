#!/bin/bash

# 4/7/2023

# python evaluate_koala_jaxseq.py --name koala_distill_13B_answer_in_gpt_prompt --host http://127.0.0.1:8004/ --k-shot 5 --data-dir /home/csnell/mmlu_easylm/data/ --save-dir /home/csnell/mmlu_easylm/outputs/ --prompt-prefix "BEGINNING OF CONVERSATION: USER: " --prompt-suffix " GPT: Answer:"

# python evaluate_llama_jaxseq.py --name llama_13B_original_prompt --host http://127.0.0.1:8005/ --k-shot 5 --data-dir /home/csnell/mmlu_easylm/data/ --save-dir /home/csnell/mmlu_easylm/outputs/

# test logprob timing
# python evaluate_koala_jaxseq.py --name koala_distill_13B_answer_in_gpt_prompt --host http://127.0.0.1:8006/ --k-shot 5 --data-dir /home/csnell/mmlu_easylm/data/ --save-dir /home/csnell/mmlu_easylm/outputs/ --prompt-prefix "BEGINNING OF CONVERSATION: USER: " --prompt-suffix " GPT: Answer:"

# echo "koala"
# python evaluate_eval_harness_jaxseq.py \
#     --host http://127.0.0.1:8004/ \
#     --k-shot 5 \
#     --input-process koala \
#     --bsize 8
# echo "koala"

# echo "llama"
# python evaluate_eval_harness_jaxseq.py \
#     --host http://127.0.0.1:8005/ \
#     --k-shot 5 \
#     --input-process identity \
#     --bsize 8
# echo "llama"

# old

# python evaluate_koala_jaxseq.py --name koala_distill_13B --host http://127.0.0.1:8004/ --k-shot 5 --data-dir /home/csnell/mmlu_easylm/data/ --save-dir /home/csnell/mmlu_easylm/outputs/ --prompt-prefix "BEGINNING OF CONVERSATION: USER: " --prompt-suffix "Answer: GPT:"

# python evaluate_koala_jaxseq.py --name llama_13B --host http://127.0.0.1:8005/ --k-shot 5 --data-dir /home/csnell/mmlu_easylm/data/ --save-dir /home/csnell/mmlu_easylm/outputs/ --prompt-suffix "Answer:"
