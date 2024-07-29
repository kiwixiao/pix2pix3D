#!/bin/bash

# Check if a directory is provided as an argument
if [ $# -eq 0 ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

# Get the directory from the first argument
directory="$1"

# Check if the directory exists
if [ ! -d "$directory" ]; then
    echo "Error: Directory '$directory' does not exist."
    exit 1
fi

echo "Changing to directory: $directory"
# Change to the specified directory
cd "$directory" || exit

echo "Current directory: $(pwd)"
echo "Files in directory:"
ls -la

# Loop through all files in the specified directory
for file in *; do
    echo "Processing file: $file"
    # Check if the filename starts and ends with a single quote
    if [[ $file == \'*\' ]]; then
        # Remove the single quotes
        newname=$(echo "$file" | sed "s/^'//;s/'$//")
        
        # Rename the file
        mv "$file" "$newname"
        echo "Renamed: $file -> $newname"
    else
        echo "File does not need renaming: $file"
    fi
done

echo "Script completed."