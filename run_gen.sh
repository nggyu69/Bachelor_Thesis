#!/bin/bash

# Path to your Python script
PYTHON_SCRIPT="Generate.py"

# Directory containing STL files
STL_DIR="$1"
# Base directory to store generated files
OUTPUT_DIR="gen_data"
# Log file
LOG_FILE="gen_log.txt"
# Number of files to generate
TARGET_FILES=1000
#start time in python datetime format
START_TIME=$(date +%s)
# Ensure the STL directory is provided and exists
if [ -z "$STL_DIR" ] || [ ! -d "$STL_DIR" ]; then
    echo "Usage: $0 <path_to_stl_directory>"
    exit 1
fi

# Trap to handle interruptions
trap 'echo "Script interrupted at $(date)" >> "$LOG_FILE"; exit 1' SIGINT SIGTERM

SECONDS=0

# Function to count files in the output directory for a specific model
count_files() {
    find "$1" -maxdepth 1 -type f | wc -l
}

{
    echo "Starting script at $(date)"
    echo "STL directory: $STL_DIR"
    echo "Output base directory: $OUTPUT_DIR"
    echo "Target files: $TARGET_FILES"
    echo "Python script: $PYTHON_SCRIPT"
    echo "Log file: $LOG_FILE"
    echo "PID: $$"

    # Run for each STL file in the directory
    for model_path in "$STL_DIR"*.stl; do
        # Extract model name without extension
        model_name=$(basename "$model_path" .stl)
        # Create output directory for this model
        model_output_dir="$OUTPUT_DIR/TrainData_$model_name/images"
        mkdir -p "$model_output_dir"
        
        INITIAL_IMAGE_COUNT=$(count_files "$model_output_dir")

        echo "Processing model: $model_path"
        echo "Output directory: $model_output_dir"

        # Continue running until the target number of files is reached
        while (( $(count_files "$model_output_dir") < TARGET_FILES )); do
            for i in $(seq 1 2); do
                if (( i % 2 == 0 )); then
                    # Generate a black image
                    blenderproc run "$PYTHON_SCRIPT" --model_path "$model_path" --color "#0f0f13" --start_time "$START_TIME" --initial_count "$INITIAL_IMAGE_COUNT" 
                else
                    # Generate a white image
                    blenderproc run "$PYTHON_SCRIPT" --model_path "$model_path" --color "#FFFFFF" --start_time "$START_TIME" --initial_count "$INITIAL_IMAGE_COUNT" 
                fi

                # Break the loop if the target file count is reached
                if (( $(count_files "$model_output_dir") >= TARGET_FILES )); then
                    echo "Target file count reached for $model_name. Moving to next model."
                    break 2
                else
                    echo "Generated $(count_files "$model_output_dir") files for $model_name. Continuing..."
                fi
            done
        done
    done

    duration=$SECONDS
    echo "All instances completed at $(date) in $((duration / 60)) minutes and $((duration % 60)) seconds."
} >> "$LOG_FILE" 2>&1
