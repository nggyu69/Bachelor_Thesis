#!/bin/bash

# Path to your Python script
PYTHON_SCRIPT="Annotate.py"

python3 $PYTHON_SCRIPT

duration=$SECONDS
echo "All instances completed in $((duration / 60)) minutes and $((duration % 60)) seconds."
