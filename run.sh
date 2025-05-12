#!/bin/bash

target=$1
if [ -z "$target" ]; then
    echo "Usage: $0 <target>"
    exit 1

# target does not exist
elif [ ! -f "$target" ]; then
    echo "target not found: $target"
    exit 1
fi

# cd to the directory of the target file
folder=$(dirname "$target")
cd "$folder" || exit 1
file=$(basename "$target")

echo "Running $file in $folder"
# for collection in indoor outdoor combined; do
for collection in indoor; do
    sbatch ../../task.job $file --category $collection
done