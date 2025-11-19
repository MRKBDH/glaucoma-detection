# 🏥 Glaucoma Detection from Clinical Notes using Deep Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Deep learning models for detecting glaucoma from clinical notes using LSTM, GRU, and 1D CNN architectures.

## 📋 Table of Contents
- [Overview](#overview)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Citation](#citation)

## 🎯 Overview

**Course:** CSCE 566 - Deep Learning  
**Institution:** University of Louisiana at Lafayette  
**Dataset:** FairCLIP (10,000 clinical notes with demographic labels)  
**Task:** Binary classification for glaucoma detection

### Key Features
✅ Three deep learning architectures (LSTM, GRU, CNN)  
✅ Fairness evaluation across demographics  
✅ High performance (87.58% AUC with CNN)  
✅ Production-ready code with proper structure  

## 📊 Results

### Overall Performance

| Model | AUC    | Sensitivity | Specificity | Parameters |
|-------|--------|-------------|-------------|------------|
| LSTM  | 82.21% | 75.46%      | 74.21%      | 5.62M      |
| GRU   | 85.19% | 79.18%      | 73.90%      | 4.94M      |
| **CNN** | **87.58%** | **89.15%** | 66.84%   | **3.35M** |

### Performance by Demographics (AUC)

| Model | White  | Black  | Asian  |
|-------|--------|--------|--------|
| LSTM  | 81.36% | 84.45% | 87.13% |
| GRU   | 83.91% | 88.55% | 91.84% |
| **CNN** | **86.17%** | **90.63%** | **93.67%** |

**Key Finding:** All models achieve >81% AUC across all demographic groups, demonstrating fairness.

## 📁 Repository Structure
```
glaucoma-detection/
├── models/                      # Neural network architectures
│   ├── __init__.py
│   ├── lstm_model.py           # LSTM implementation
│   ├── gru_model.py            # GRU implementation
│   └── cnn_model.py            # 1D CNN implementation
│
├── utils/                       # Utility functions
│   ├── __init__.py
│   ├── data_utils.py           # Data loading and splitting
│   ├── text_preprocessing.py   # Text preprocessing
│   └── dataset.py              # PyTorch dataset class
│
├── training/                    # Training scripts
│   ├── __init__.py
│   ├── train_lstm.py
│   ├── train_gru.py
│   └── train_cnn.py
│
├── evaluation/                  # Evaluation and visualization
│   ├── __init__.py
│   ├── evaluate_all_models.py
│   ├── visualize_results.py
│   └── compare_models.py
│
├── figures/                     # Generated visualizations
│   ├── architecture_comparison.png
│   ├── roc_curves.png
│   ├── performance_by_race.png
│   └── ...
│
├── data/                        # Data directory (not committed)
│   └── clinical_notes.csv      # FairCLIP dataset
│
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .gitignore                  # Git ignore rules
└── REFLECTION.txt              # Project reflection
```

## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA (optional, for GPU support)

### Setup
```bash
# Clone repository
git clone https://github.com/MRKBDH/glaucoma-detection.git
cd glaucoma-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Download Dataset
Download the FairCLIP dataset and place `clinical_notes.csv` in the `data/` directory.

## 💻 Usage

### Train Models
```bash
# Train LSTM
python training/train_lstm.py

# Train GRU  
python training/train_gru.py

# Train CNN (best performance)
python training/train_cnn.py
```

### Evaluate Models
```bash
# Evaluate all models
python evaluation/evaluate_all_models.py

# Generate visualizations
python evaluation/visualize_results.py

# Compare models
python evaluation/compare_models.py
```

### Example: Quick Test
```python
import torch
from models.cnn_model import CNN1DClassifier
from utils.text_preprocessing import TextPreprocessor

# Load model
model = CNN1DClassifier(vocab_size=10000, embedding_dim=300)
model.load_state_dict(torch.load('best_cnn.pth'))

# Preprocess text
preprocessor = TextPreprocessor()
text = "Patient presents with elevated intraocular pressure..."
processed = preprocessor.preprocess(text)

# Predict
prediction = model(processed)
print(f"Glaucoma probability: {prediction.item():.2%}")
```

## 🧠 Models

### LSTM (Long Short-Term Memory)
- **Architecture:** 2-layer Bidirectional LSTM
- **Hidden Size:** 256 (512 with bidirectional)
- **Parameters:** 5.62M
- **Best For:** Capturing long-term dependencies

### GRU (Gated Recurrent Unit)
- **Architecture:** 2-layer Bidirectional GRU
- **Hidden Size:** 256 (512 with bidirectional)
- **Parameters:** 4.94M
- **Best For:** Balance of performance and efficiency

### 1D CNN (Convolutional Neural Network)
- **Architecture:** Parallel filters (kernel sizes: 3, 4, 5)
- **Filters:** 128 per kernel size
- **Parameters:** 3.35M
- **Best For:** Fast inference, high accuracy

## 📈 Hyperparameters
```python
EMBEDDING_DIM = 300
HIDDEN_SIZE = 256        # LSTM/GRU
NUM_FILTERS = 128        # CNN
KERNEL_SIZES = [3, 4, 5] # CNN
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
DROPOUT = 0.3
MAX_SEQ_LENGTH = 200
```

## 📚 Citation
```bibtex
@misc{glaucoma_detection_2025,
  title={Glaucoma Detection from Clinical Notes using Deep Learning},
  author={Mithun Ranjan Kar},
  year={2025},
  institution={University of Louisiana at Lafayette},
  url={https://github.com/MRKBDH/glaucoma-detection}
}
```

## 📝 References

1. **FairCLIP:** Luo et al., "FairCLIP: Harnessing Fairness in Vision-Language Learning", CVPR 2024
2. **LSTM:** Hochreiter & Schmidhuber, "Long Short-Term Memory", Neural Computation 1997
3. **GRU:** Cho et al., "Learning Phrase Representations using RNN Encoder-Decoder", EMNLP 2014

## 👤 Author

**Student:** Mithun Ranjan Kar  
**Email:** mithun-ranjan.kar1@louisiana.edu  
**Course:** CSCE 566 - Deep Learning, Fall 2025  
**Instructor:** Dr. Min Shi

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- Dr. Min Shi for project guidance
- FairCLIP dataset creators
- University of Louisiana at Lafayette

---

**⭐ If you find this project useful, please consider starring the repository!**
