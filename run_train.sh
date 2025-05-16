#!/bin/bash

# Define the list of models
models=(
    "active_canny"
    "anime_style" 
    "adaptive_threshold"
    "canny"
    "control" 
    "contour_style"
    "HED/1" 
    "HED/2" 
    "opensketch_style"
    )

# Path to the dataset directory
dataset_path="/data/reddy/Bachelor_Thesis/datasets/publish_dataset"

#loop through all sizes
sizes=("s")
for size in "${sizes[@]}"; do
    # Loop through each model and run the Python training script
    for model in "${models[@]}"; do
        echo "Starting training for model: $model"

        # Run the Python script with the current model as an argument
        python3 train.py --model "$model" --dataset_path "$dataset_path" --model_size "$size"

        if [ $? -ne 0 ]; then
            echo "Training failed for model: $model"
            exit 1
        fi

        echo "Training completed for model: $model"
    done
done

echo "All models trained successfully."
