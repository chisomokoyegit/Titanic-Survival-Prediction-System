#!/bin/bash
# build.sh - Script to run before deployment

echo "Setting up model..."

# Check if model needs to be built
if [ ! -f "model/titanic_survival_model.pkl" ] || [ ! -s "model/titanic_survival_model.pkl" ]; then
    echo "Model not found or empty. Please ensure titanic.csv is in the root directory and run:"
    echo "  python model/model_building.py"
    echo ""
    echo "For now, creating a placeholder..."
    mkdir -p model
    touch model/titanic_survival_model.pkl
else
    echo "Model found: $(ls -lh model/titanic_survival_model.pkl)"
fi

echo "Build setup complete!"
