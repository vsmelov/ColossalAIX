#!/usr/bin/env bash

torchrun --standalone --nproc_per_node 8 inference_tp.py --pretrained grok-1
