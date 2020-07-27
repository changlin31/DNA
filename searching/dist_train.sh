#!/bin/bash
python -m torch.distributed.launch --nproc_per_node=8 train.py