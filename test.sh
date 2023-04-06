#!/bin/bash

# python evaluate_koala_jaxseq.py --name koala_distill_13B --host http://127.0.0.1:8004/ --k-shot 5 --data-dir /home/csnell/mmlu_easylm/data/ --save-dir /home/csnell/mmlu_easylm/outputs/ --prompt-prefix "BEGINNING OF CONVERSATION: USER: " --prompt-suffix " GPT:"

python evaluate_koala_jaxseq.py --name llama_13B --host http://127.0.0.1:8005/ --k-shot 5 --data-dir /home/csnell/mmlu_easylm/data/ --save-dir /home/csnell/mmlu_easylm/outputs/
