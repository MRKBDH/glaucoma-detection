#!/bin/bash

echo "=== Reorganizing Repository Structure ==="

# Create directories
mkdir -p models utils training evaluation figures data

# Move model files
echo "Moving model files..."
[ -f lstm_model.py ] && mv lstm_model.py models/
[ -f gru_model.py ] && mv gru_model.py models/
[ -f cnn_model.py ] && mv cnn_model.py models/

# Move utility files
echo "Moving utility files..."
[ -f data_utils.py ] && mv data_utils.py utils/
[ -f text_preprocessing.py ] && mv text_preprocessing.py utils/
[ -f dataset.py ] && mv dataset.py utils/

# Move training scripts
echo "Moving training scripts..."
[ -f train_lstm.py ] && mv train_lstm.py training/
[ -f train_gru.py ] && mv train_gru.py training/
[ -f train_cnn.py ] && mv train_cnn.py training/

# Move evaluation scripts
echo "Moving evaluation scripts..."
[ -f evaluate_*.py ] && mv evaluate_*.py evaluation/
[ -f visualize_*.py ] && mv visualize_*.py evaluation/
[ -f compare_*.py ] && mv compare_*.py evaluation/

# Move figures
echo "Moving figures..."
[ -f *.png ] && mv *.png figures/ 2>/dev/null
[ -f figures_*/*.png ] && mv figures_*/*.png figures/ 2>/dev/null

# Move data (but don't commit large files)
echo "Moving data files..."
[ -f *.csv ] && mv *.csv data/ 2>/dev/null

# Clean up
echo "Cleaning up..."
rm -rf figures_lstm figures_gru figures_cnn figures_comparison 2>/dev/null
rm -rf __pycache__ */__pycache__ 2>/dev/null

echo "✓ Repository reorganized!"
echo ""
echo "New structure:"
tree -L 2 -I '__pycache__|*.pyc|venv' || ls -R
