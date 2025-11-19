# Glaucoma Detection from Clinical Notes using Deep Learning

Deep learning models (LSTM, GRU, 1D CNN) for detecting glaucoma from clinical notes using the FairCLIP dataset.

## 🎯 Project Overview

- **Course:** CSCE 566 - Deep Learning
- **Institution:** University of Louisiana at Lafayette
- **Task:** Binary classification (Glaucoma detection from clinical text)
- **Dataset:** FairCLIP - 10,000 clinical notes with fairness labels

## 📊 Results Summary

| Model | AUC    | Sensitivity | Specificity | Parameters |
|-------|--------|-------------|-------------|------------|
| LSTM  | 82.21% | 75.46%      | 74.21%      | 5.62M      |
| GRU   | 85.19% | 79.18%      | 73.90%      | 4.94M      |
| **CNN** | **87.58%** | **89.15%** | 66.84%   | **3.35M** |

## 🏆 Key Achievements

✅ CNN achieves best overall AUC (87.58%)  
✅ Fairness across demographics (>86% AUC for all groups)  
✅ High sensitivity (89.15%) - critical for medical screening  
✅ Most efficient model (3.35M parameters)  

## 📁 Repository Structure
```
glaucoma_detection/
├── models/              # Model architectures
├── utils/               # Data processing utilities
├── training/            # Training scripts
├── evaluation/          # Evaluation and visualization
├── figures/             # Results visualizations
└── README.md           # This file
```

## 🚀 Quick Start
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/glaucoma-detection.git
cd glaucoma-detection

# Install dependencies
pip install torch numpy pandas scikit-learn matplotlib seaborn tqdm

# Run training (example)
python training/train_cnn.py
```

## 📈 Performance by Demographics

| Model | White  | Black  | Asian  |
|-------|--------|--------|--------|
| LSTM  | 81.36% | 84.45% | 87.13% |
| GRU   | 83.91% | 88.55% | 91.84% |
| CNN   | 86.17% | 90.63% | 93.67% |

## �� References

1. FairCLIP Dataset: Luo et al., CVPR 2024
2. LSTM: Hochreiter & Schmidhuber, Neural Computation 1997
3. GRU: Cho et al., EMNLP 2014

## 👤 Author

**Student:** [Your Name]  
**Email:** [Your Email]  
**Course:** CSCE 566, Fall 2024  
**Instructor:** Dr. Min Shi

## 📄 License

MIT License
