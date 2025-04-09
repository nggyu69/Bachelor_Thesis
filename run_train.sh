#!/bin/bash

# Define the list of models
models=(
    # "control" 
    # "canny" 
    # "active_canny" 
    # "HED/1" 
    # "HED/2" 
    # "anime_style" 
    # "opensketch_style" 
    # "contour_style"
    "adaptive_threshold"
    )

# Path to the dataset directory
dataset_path="/data/reddy/Bachelor_Thesis/datasets/8object_dataset"

#loop through all sizes
sizes=("x")
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
