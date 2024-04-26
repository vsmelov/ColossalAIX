#!/usr/bin/env bash

torchrun --standalone --nproc_per_node 8 inference_tp.py --pretrained grok-1 \
    --max_new_tokens 1000 \
    --text "I will write you very detailed explanations of what is Ethereum and how it works. Ethereum is"
