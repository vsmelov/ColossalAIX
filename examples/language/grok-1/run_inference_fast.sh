#!/usr/bin/env bash

torchrun --standalone --nproc_per_node 8 inference_tp.py --pretrained grok-1 \
    --max_new_tokens 300 \
    --text "I will write you very detailed explanations of what is Ethereum and how it works. Ethereum is"


# original
#torchrun --standalone --nproc_per_node 8 inference_tp.py --pretrained grok-1 \
#    --max_new_tokens 100 \
#    --text "The company's annual conference, featuring keynote speakers and exclusive product launches, will be held at the Los Angeles Convention Center from October 20th to October 23rd, 2021. Extract the date mentioned in the above sentence."
