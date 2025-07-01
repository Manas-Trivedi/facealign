# Robust Face Recognition and Gender Classification under Adverse Visual Conditions

## COMSYS Hackathon-5, 2025 Submission

### Team Information
- **Author**: Manas Trivedi
- **Theme**: Robust Face Recognition and Gender Classification under Adverse Visual Conditions
- **Organization**: COMSYS Educational Trust, Kolkata

## Project Overview

This project implements a comprehensive solution for two challenging computer vision tasks under adverse visual conditions:

1. **Task A**: Gender Classification (Binary Classification) - Male/Female prediction
2. **Task B**: Face Recognition (Multi-class Classification) - Person identity matching using embeddings

The solution is designed to handle real-world degradations including motion blur, overexposed scenes, foggy conditions, rainy weather simulation, low light visibility, and uneven lighting.

## Dataset: FACECOM

**FACECOM (Face Attributes in Challenging Environments)** is a purpose-built dataset with over 5,000 face images captured or synthesized under challenging visual conditions.

### Dataset Structure
```
data/facecom/
├── Task_A/                 # Gender Classification
│   ├── train/
│   │   ├── male/          # Male face images
│   │   └── female/        # Female face images
│   └── val/
│       ├── male/
│       └── female/
└── Task_B/                 # Face Recognition
    ├── train/              # Identity folders (001_frontal, 002_frontal, etc.)
    └── val/                # Validation identity folders
```

## Architecture

### Task A: Gender Classification Model
- **Backbone**: ResNet34 (pretrained on ImageNet)
- **Architecture**: Feature extraction + Custom classifier with dropout
- **Output**: Binary classification (Male/Female)
- **Loss Function**: Weighted Cross-Entropy (to handle class imbalance)

### Task B: Face Recognition Model
- **Backbone**: ResNet18/34/50 (configurable, pretrained)
- **Architecture**: Siamese Network with embedding learning
- **Embedding Dimension**: 256D normalized embeddings
- **Loss Functions**:
  - Triplet Loss with margin
  - Alignment Loss (pulls positive pairs together)
  - Uniformity Loss (spreads embeddings uniformly on hypersphere)
- **Matching Strategy**: Cosine similarity with threshold-based decision

## Key Features

### Advanced Loss Functions
- **Alignment-Uniformity Loss**: Combines triplet loss with alignment and uniformity objectives
- **Decorrelated Batch Normalization**: Prevents dimensional collapse in embeddings
- **Numerical Stability**: Robust implementations for MPS/GPU compatibility

### Data Handling
- **Class Balancing**: Weighted sampling for gender classification
- **Triplet Mining**: Hard negative mining for face recognition
- **Data Augmentation**: Comprehensive transformations for robustness

### Evaluation Metrics
- **Task A**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Task B**: Top-1 Accuracy, Macro-averaged F1-Score, Threshold Analysis

## Installation and Setup

### Requirements
```bash
pip install -r requirements.txt
```

### Dependencies
- torch
- torchvision
- pillow
- matplotlib
- scikit-learn
- tqdm

### Hardware Support
- CUDA GPU (recommended)
- Apple Silicon MPS
- CPU fallback

## Usage

### Training

#### Task A - Gender Classification
```bash
python train_a.py
```

#### Task B - Face Recognition
```bash
python train_b.py
```

### Testing and Evaluation

#### Official Competition Evaluation (For Judges)
```bash
# Quick official evaluation (recommended for judges)
python run_evaluation.py

# Alternative: Use unified test runner
python run_all_tests.py --official     # Official evaluation only
python run_all_tests.py --both         # Both comprehensive and official
```

**Evaluation Metrics:**
- **Task A**: Accuracy | Precision | Recall | F1-Score
- **Task B**: Top-1 Accuracy | Macro-averaged F1-Score

#### Comprehensive Testing and Analysis
```bash
# Comprehensive analysis (default)
python run_all_tests.py

# Individual comprehensive tests
python test_gender_model.py          # Task A comprehensive testing
python test_model.py                 # Task B comprehensive testing
```

#### Demo Predictions
```bash
# Gender classification demo
python demo_gender.py

# Face recognition demo
python show_predictions.py
```

## Model Performance

### Task A: Gender Classification
- **Overall Accuracy**: 99.79%
- **Overall Precision**: 99.24%
- **Overall Recall**: 99.75%
- **Overall F1-Score**: 99.49%
- **ROC AUC**: 100.00%

#### Per-Class Performance
- **Male**: Precision: 99.93%, Recall: 99.80%, F1: 99.87%
- **Female**: Precision: 99.24%, Recall: 99.75%, F1: 99.49%

### Task B: Face Recognition
- **Optimal Threshold**: 0.975
- **Best Accuracy**: 72.00%
- **Best F1-Score**: 72.44%
- **Precision at Optimal**: 71.32%

## File Structure

```
facealign/
├── models/
│   ├── gender_model.py        # Gender classification model
│   └── face_model.py          # Face embedding model
├── utils/
│   ├── gender_dataset.py      # Gender dataset utilities
│   └── face_dataset.py        # Face dataset utilities
├── checkpoints/
│   ├── gender_model.pt        # Trained gender model weights
│   └── final_model.pth        # Trained face recognition weights
├── test_results/              # Face recognition test outputs
├── test_results_gender/       # Gender classification test outputs
├── outputs/                   # Generated outputs and visualizations
├── train_a.py                 # Task A training script
├── train_b.py                 # Task B training script
├── test_gender_model.py       # Gender classification testing & analysis
├── test_model.py              # Face recognition testing & analysis
├── run_evaluation.py          # Official competition evaluation suite
├── demo_gender.py             # Gender classification demo
├── show_predictions.py        # Face recognition demo
├── run_all_tests.py          # Automated test runner
└── requirements.txt          # Project dependencies
```

## Model Architecture Diagrams

### Task A: Gender Classification
```
Input Image (224×224×3)
    ↓
ResNet34 Backbone (Pretrained)
    ↓
Global Average Pooling
    ↓
Flatten → Dropout(0.3) → Linear(512→2)
    ↓
Binary Classification Output
```

### Task B: Face Recognition
```
Input Image (224×224×3)
    ↓
ResNet Backbone (Pretrained)
    ↓
Embedding Head: Linear→BN→ReLU→Dropout→Linear→BN
    ↓
L2 Normalized Embeddings (256D)
    ↓
Cosine Similarity Matching
    ↓
Threshold-based Decision
```

## Key Innovations

1. **Robust Loss Design**: Combination of triplet, alignment, and uniformity losses
2. **Numerical Stability**: MPS-compatible implementations with NaN handling
3. **Threshold Optimization**: Comprehensive analysis for optimal decision boundaries
4. **Class Balancing**: Weighted training to handle dataset imbalances
5. **Comprehensive Evaluation**: Multi-metric analysis with visualization

## Evaluation Protocol

The models are evaluated using the specified competition metrics:
- **Task A Weight**: 30% (Gender Classification)
- **Task B Weight**: 70% (Face Recognition)

### Official Competition Evaluation
- **For Judges**: Use `python run_evaluation.py` for standardized evaluation
- **Output**: Comprehensive reports with official metrics and visualizations
- **Format**: Timestamped results with detailed analysis and summary reports

### Testing Methodology
- **Competition Evaluation**: `run_evaluation.py` generates official metric reports for judges
- **Automated Testing**: `run_all_tests.py` executes comprehensive analysis for both tasks
- **Individual Testing**: Separate test scripts for detailed analysis and debugging
- **Performance Metrics**: Detailed evaluation with threshold analysis and visualization
- **Demo Scripts**: Interactive demonstration of model predictions with visual feedback

## Technical Highlights

- **Cross-platform Compatibility**: Supports CUDA, MPS, and CPU
- **Robust Training**: Early stopping, learning rate scheduling, gradient clipping
- **Comprehensive Logging**: Detailed training and testing logs
- **Visualization**: Performance plots and analysis charts
- **Reproducible Results**: Fixed random seeds and deterministic operations

## Repository Structure

All code is well-documented, modular, and follows best practices for deep learning projects. The implementation handles edge cases, provides comprehensive error checking, and includes extensive logging for debugging and analysis.

## Contact

For questions or clarifications regarding this implementation, please refer to the competition guidelines or contact the organizing committee at comsyshackathon5@gmail.com.
