#!/bin/bash

# 5/8/2023

# nlp4 pane 6 bottom – koala-13B koala step_1250 – served on charlie-pod2
# python evaluate_eval_harness_jaxseq_multiple_host.py \
#     --host http://34.148.106.55:8000/ http://34.74.15.70:8000/ http://35.237.51.173:8000/ http://34.73.22.103:8000/ \
#     --k-shot 5 \
#     --input-process koala \
#     --bsize 32

# nlp4 pane 6 top – gpt2-xl koala last – served on charlie-pod
# python evaluate_eval_harness_jaxseq_multiple_host.py \
#     --host http://35.185.31.188:8000/ http://34.74.95.238:8000/ http://34.73.119.72:8000/ http://104.196.32.17:8000/ \
#     --k-shot 5 \
#     --input-process koala \
#     --bsize 128 \
#     --max-input-length 512 \
#     --max-output-length 512


# rail-a100 – 13B koala last – served on rail-a100
# python evaluate_eval_harness_jaxseq.py \
#     --host http://127.0.0.1:8000/ \
#     --k-shot 5 \
#     --input-process koala \
#     --bsize 512

# nlp4 pane 6 – 7B koala last – served on charlie-pod2
# python evaluate_eval_harness_jaxseq_multiple_host.py \
#     --host http://34.148.106.55:8000/ http://34.74.15.70:8000/ http://35.237.51.173:8000/ http://34.73.22.103:8000/ \
#     --k-shot 5 \
#     --input-process koala \
#     --bsize 8

# 5/6/2023

# nlp1 top-left pane – gpt2-xl
# python evaluate_eval_harness_jaxseq.py \
#     --host http://127.0.0.1:8079/ \
#     --k-shot 5 \
#     --input-process identity \
#     --bsize 4 \
#     --max-input-length 512 \
#     --max-output-length 512 \

# 5/5/2023

# nlp1 top-left pane – gpt2-xl
# python evaluate_llama_jaxseq.py \
#     --name gpt2_xl_original_prompt_5_shot \
#     --host http://127.0.0.1:8079/ \
#     --k-shot 5 \
#     --data-dir /home/csnell/mmlu_jaxseq/data/ \
#     --save-dir /home/csnell/mmlu_jaxseq/outputs/ \
#     --max-input-length 512 \
#     --max-output-length 512 \

# 5/4/2023

# nlp1 top-left – koala gpt2-xl
# python evaluate_eval_harness_jaxseq.py \
#     --host http://127.0.0.1:8079/ \
#     --k-shot 5 \
#     --input-process koala \
#     --bsize 4 \
#     --max-input-length 512 \
#     --max-output-length 512 \

# nlp1 top-right pane – llama 7B
# python evaluate_llama_jaxseq.py \
#     --name llama_7B_original_prompt_5_shot \
#     --host http://127.0.0.1:8089/ \
#     --k-shot 5 \
#     --data-dir /home/csnell/mmlu_jaxseq/data/ \
#     --save-dir /home/csnell/mmlu_jaxseq/outputs/

# nlp1 bottom-left pane – koala 7B 1 epoch last
# python evaluate_eval_harness_jaxseq.py \
#     --host http://127.0.0.1:8099/ \
#     --k-shot 5 \
#     --input-process koala \
#     --bsize 4



# 4/23/2023

# nlp4 window 15 bottom pane
# python evaluate_llama_jaxseq.py \
#     --name flan_llama_7B_80k_original_prompt \
#     --host http://127.0.0.1:8079/ \
#     --k-shot 5 \
#     --data-dir /home/csnell/mmlu_easylm/data/ \
#     --save-dir /home/csnell/mmlu_easylm/outputs/

# 4/22/2023

# nlp4 window 15 bottom pane
# python evaluate_llama_jaxseq.py \
#     --name flan_llama_7B_original_prompt \
#     --host http://127.0.0.1:8079/ \
#     --k-shot 5 \
#     --data-dir /home/csnell/mmlu_easylm/data/ \
#     --save-dir /home/csnell/mmlu_easylm/outputs/


# nlp4 window 15 bottom pane
# python evaluate_eval_harness_jaxseq.py \
#     --host http://127.0.0.1:8079/ \
#     --k-shot 5 \
#     --input-process identity \
#     --bsize 4

# 4/16/2023

# nlp3 top left pane
# echo "koala 1250"
# python evaluate_eval_harness_jaxseq.py \
#     --host http://127.0.0.1:8006/ \
#     --k-shot 5 \
#     --input-process koala \
#     --bsize 4 
# echo "koala 1250"

# nlp3 bottom left pane
# echo "koala 5000"
# python evaluate_eval_harness_jaxseq.py \
#     --host http://127.0.0.1:8007/ \
#     --k-shot 5 \
#     --input-process koala \
#     --bsize 4 
# echo "koala 5000"

# nlp3 top right pane
# echo "koala last"
# python evaluate_eval_harness_jaxseq.py \
#     --host http://127.0.0.1:8008/ \
#     --k-shot 5 \
#     --input-process koala \
#     --bsize 4 
# echo "koala last"

# 4/13/2023

# chatgpt 0 shot

# nlp4 right pane
# python chatgpt_eval.py --name chatgpt_eval_zero_shot --k-shot 0 --data-dir /home/csnell/mmlu_easylm/data/ --save-dir /home/csnell/mmlu_easylm/outputs/


# 4/12/2023

# nlp4 left pane
# echo "koala"
# python evaluate_eval_harness_jaxseq.py \
#     --host http://127.0.0.1:8005/ \
#     --k-shot 0 \
#     --input-process koala \
#     --bsize 4 
# echo "koala"

# nlp4 right pane
# echo "llama"
# python evaluate_llama_jaxseq.py \
#     --name llama_13B_original_prompt_zero_shot \
#     --host http://127.0.0.1:8004/ \
#     --k-shot 0 \
#     --data-dir /home/csnell/mmlu_easylm/data/ \
#     --save-dir /home/csnell/mmlu_easylm/outputs/
# echo "llama"

# nlp3 top left pane
# echo "koala 10k"
# python evaluate_eval_harness_jaxseq.py \
#     --host http://127.0.0.1:8006/ \
#     --k-shot 5 \
#     --input-process koala \
#     --bsize 4 
# echo "koala 10k"

# nlp3 bottom left pane
# echo "koala 50k"
# python evaluate_eval_harness_jaxseq.py \
#     --host http://127.0.0.1:8007/ \
#     --k-shot 5 \
#     --input-process koala \
#     --bsize 4 
# echo "koala 50k"

# nlp3 top right pane
# echo "koala 100k"
# python evaluate_eval_harness_jaxseq.py \
#     --host http://127.0.0.1:8008/ \
#     --k-shot 5 \
#     --input-process koala \
#     --bsize 4 
# echo "koala 100k"


# 4/11/2023

# left pane
# python evaluate_koala_jaxseq2.py --name koala_distill_13B_matches_default_answer_in_prompt --host http://127.0.0.1:8004/ --k-shot 5 --data-dir /home/csnell/mmlu_easylm/data/ --save-dir /home/csnell/mmlu_easylm/outputs/ --prompt-prefix "BEGINNING OF CONVERSATION: USER: " --prompt-suffix " GPT: Answer:"

# right pane
# python evaluate_koala_jaxseq.py --name koala_distill_13B_answer_in_gpt_prompt2 --host http://127.0.0.1:8005/ --k-shot 5 --data-dir /home/csnell/mmlu_easylm/data/ --save-dir /home/csnell/mmlu_easylm/outputs/ --prompt-prefix "BEGINNING OF CONVERSATION: USER: " --prompt-suffix " GPT: Answer:"

# 4/10/2023

# top left pane
# python evaluate_koala_jaxseq2.py --name koala_distill_13B_matches_default_answer_in_prompt --host http://127.0.0.1:8004/ --k-shot 5 --data-dir /home/csnell/mmlu_easylm/data/ --save-dir /home/csnell/mmlu_easylm/outputs/ --prompt-prefix "BEGINNING OF CONVERSATION: USER: " --prompt-suffix " GPT: Answer:"

# bottom left pane
# python evaluate_koala_jaxseq.py --name koala_distill_13B_answer_in_gpt_prompt2 --host http://127.0.0.1:8005/ --k-shot 5 --data-dir /home/csnell/mmlu_easylm/data/ --save-dir /home/csnell/mmlu_easylm/outputs/ --prompt-prefix "BEGINNING OF CONVERSATION: USER: " --prompt-suffix " GPT: Answer:"

# top right pane
# python chatgpt_eval.py --name chatgpt_eval --k-shot 5 --data-dir /home/csnell/mmlu_easylm/data/ --save-dir /home/csnell/mmlu_easylm/outputs/

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
#     --bsize 4
# echo "koala"

# echo "llama"
# python evaluate_eval_harness_jaxseq.py \
#     --host http://127.0.0.1:8005/ \
#     --k-shot 5 \
#     --input-process identity \
#     --bsize 4
# echo "llama"

# old

# python evaluate_koala_jaxseq.py --name koala_distill_13B --host http://127.0.0.1:8004/ --k-shot 5 --data-dir /home/csnell/mmlu_easylm/data/ --save-dir /home/csnell/mmlu_easylm/outputs/ --prompt-prefix "BEGINNING OF CONVERSATION: USER: " --prompt-suffix "Answer: GPT:"

# python evaluate_koala_jaxseq.py --name llama_13B --host http://127.0.0.1:8005/ --k-shot 5 --data-dir /home/csnell/mmlu_easylm/data/ --save-dir /home/csnell/mmlu_easylm/outputs/ --prompt-suffix "Answer:"
