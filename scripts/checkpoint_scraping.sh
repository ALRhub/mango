#!/bin/bash

# Usage:
# ./copy_checkpoints.sh /path/to/input_folder /path/to/target_folder

set -e  # exit immediately on error
set -u  # treat unset variables as errors

INPUT_DIR="$1"
TARGET_DIR="$2"

# Create target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Find all directories named "checkpoints" under the input directory
find "$INPUT_DIR" -type d -name "checkpoints" | while read -r CHECKPOINT_DIR; do
    # Compute the relative path from INPUT_DIR to this checkpoints directory
    REL_PATH="${CHECKPOINT_DIR#$INPUT_DIR/}"

    # Get the parent path of this checkpoints directory
    PARENT_PATH="$(dirname "$REL_PATH")"

    # Create the corresponding directory structure in TARGET_DIR
    mkdir -p "$TARGET_DIR/$PARENT_PATH/checkpoints"

    # Copy the content of the checkpoints directory
    cp -r "$CHECKPOINT_DIR/"* "$TARGET_DIR/$PARENT_PATH/checkpoints/"

    echo "Copied: $CHECKPOINT_DIR -> $TARGET_DIR/$PARENT_PATH/checkpoints"
done

echo "Done copying all checkpoints folders!"
