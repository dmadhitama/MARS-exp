#!/bin/bash

python run_multipurpose.py \
    --output_direct results/model_mri_kan-RGB-e200/ \
    --batch_size 1 \
    --input_type video \
    --cache_dir /home/ubuntu/MARS-RGB-Radar-cache-npy-bak \
    --kan
