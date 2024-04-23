#!/usr/bin/env bash

torchrun --standalone --nproc_per_node 8 inference_tp.py --pretrained grok-1 \
    --max_new_tokens 100 \
    --text "All books have the same weight, 10 books weigh 5kg, what is the weight of 2 books?"
