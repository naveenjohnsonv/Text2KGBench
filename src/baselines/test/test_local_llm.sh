#!/bin/bash
if [ $# -eq 0 ]; then
    echo "Usage: ./run_model.sh 'your prompt here'"
    exit 1
fi

python3 init_model.py && python3 get_response.py "$1"
