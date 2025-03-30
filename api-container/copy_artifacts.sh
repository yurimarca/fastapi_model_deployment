#!/bin/bash

# Create artifacts directory if it doesn't exist
mkdir -p artifacts

# Copy artifacts from the project root
cp ../artifacts/*.joblib artifacts/

echo "Artifacts copied successfully" 