# #!/bin/bash

# # Path to your Python script
# PYTHON_SCRIPT="Generate.py"

# # Function to run the Python script
# run_python_script() {
#     blenderproc run "$PYTHON_SCRIPT"
# }

# # Infinite loop to ensure the script runs until successful exit
# while true; do
#     run_python_script
#     exit_code=$?
    
#     # Check if the exit code is 0 (successful execution)
#     if [ $exit_code -eq 69 ]; then
#         echo "Python script executed successfully."
#         break
#     else
#         echo "Python script failed with exit code $exit_code. Restarting..."
#     fi
# done

#!/bin/bash

# Path to your Python script
PYTHON_SCRIPT="Generate.py"

# Number of instances to run sequentially
NUM_INSTANCES=1000
SECONDS=0

# Run each instance sequentially
for i in $(seq 1 $NUM_INSTANCES); do
    if (( i % 2 == 0 )); then
        # Generate a black image
        blenderproc run "$PYTHON_SCRIPT" --num_images "1" --color "#0f0f13"
    else
        # Generate a white image
        blenderproc run "$PYTHON_SCRIPT" --num_images "1" --color "#FFFFFF"
    fi
done

duration=$SECONDS
echo "All instances completed in $((duration / 60)) minutes and $((duration % 60)) seconds."
