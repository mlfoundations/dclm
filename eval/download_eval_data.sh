#!/bin/bash

# Clone the repository
git clone https://github.com/mosaicml/llm-foundry.git

# Check if the clone was successful
if [ $? -eq 0 ]; then
    echo "Repository cloned successfully."
else
    echo "Failed to clone the repository."
    exit 1
fi

# Copy the directory to the current directory
cp -r llm-foundry/scripts/eval/local_data .

# Check if the copy was successful
if [ $? -eq 0 ]; then
    echo "Directory copied successfully."
else
    echo "Failed to copy the directory."
    exit 1
fi

# Remove the cloned repository
rm -rf llm-foundry

# Check if the delete was successful
if [ $? -eq 0 ]; then
    echo "Repository deleted successfully."
else
    echo "Failed to delete the repository."
    exit 1
fi