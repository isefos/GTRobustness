#!/bin/bash

# Base directory where the files are located
BASE_DIR="results_t"

# Destination directory
DEST_DIR="copy_plots"

# Create destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Find and process files
find "$BASE_DIR" -type f -path "*/plots/agg_ab_bt/*.pdf" | while read -r FILE; do
    # Extract components from the file path
    FILENAME=$(basename "$FILE" .pdf)
    COLLECTION=$(echo "$FILE" | awk -F'/' '{print $(NF-4)}')
    ID=$(echo "$FILE" | awk -F'/' '{print $(NF-3)}')

    # Check if the filename contains the word "Accuracy"
    if [[ "$FILENAME" == *Accuracy* ]] && [[ "$FILENAME" != *UPFD_gos* ]] && [[ "$COLLECTION" != *adv* ]] && [[ "$COLLECTION" != *gat* ]] && [[ "$COLLECTION" != *ga2* ]] && [[ "$COLLECTION" != *gpsgcn* ]]; then
        # Construct new filename
        NEW_FILENAME="${FILENAME}.pdf"

        # Copy file to destination with new name
        cp "$FILE" "$DEST_DIR/$NEW_FILENAME"
    fi
done

echo "All files containing 'Accuracy' have been copied and renamed into $DEST_DIR"
