#!/bin/bash

# Example script for running the retrieval pipeline

# Set paths (modify these for your setup)
DATASET_NAME="custom_dataset"
ANNS_FILE="/path/to/your/annotations.pt"
CAPTIONS_FILE="/path/to/your/captions.pt"  # Optional: for mask-based extraction
WEIGHTS_FILE="/path/to/your/weights.ckpt"  # Optional: for fine-tuned models

# Run index creation and search
python create_index.py \
    tag=example_run \
    dataset=$DATASET_NAME \
    extractor=dinov2_mi_features \
    B_model=true \
    vec_dim=768 \
    mi_alpha=0.03 \
    lora_adapt=true \
    lora_rank=256 \
    single_from_sc=true \
    mi_sum=true \
    global_features=true \
    smart_crop=true \
    anns_file=$ANNS_FILE \
    captions_file=$CAPTIONS_FILE \
    weights=$WEIGHTS_FILE

# The search will be automatically triggered after index creation
# To run search separately:
# python search_index.py \
#     dataset=$DATASET_NAME \
#     experiment=custom_dataset_dinov2_mi_features_global_smart_crop_B_model_mi_sum_example_run \
#     k_search=100 \
#     features_dir=outputs/${DATASET_NAME}_retrieval/
