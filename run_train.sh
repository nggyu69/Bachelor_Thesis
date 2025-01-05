#!/bin/bash

# Define the list of models
models=(
    "control" 
    "canny" 
    "active_canny" 
    # "HED/1" 
    # "HED/2" 
    "anime_style" 
    "opensketch_style" 
    "contour_style"
    )

# Path to the dataset directory
dataset_path="/data/reddy/Bachelor_Thesis/datasets/8object"

# Loop through each model and run the Python training script
for model in "${models[@]}"; do
    echo "Starting training for model: $model"

    # Run the Python script with the current model as an argument
    python train.py --model "$model" --dataset_path "$dataset_path"

    if [ $? -ne 0 ]; then
        echo "Training failed for model: $model"
        exit 1
    fi

    echo "Training completed for model: $model"
done

echo "All models trained successfully."
