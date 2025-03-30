#!/bin/bash

# Create artifacts directory if it doesn't exist
mkdir -p artifacts

# Copy artifacts from the project root
cp ../artifacts/*.joblib artifacts/

# # Copy src code
# mkdir -p src
# cp -r ../src src/

echo "Artifacts copied successfully" 