# Robust Face Recognition and Gender Classification under Adverse Visual Conditions

## COMSYS Hackathon-5, 2025 Submission
**Author**: Manas Trivedi

## Solution Overview

This solution addresses the challenge of maintaining robust performance in face recognition and gender classification under adverse visual conditions including motion blur, overexposure, fog, rain, low light, and uneven lighting.

**Tasks:**
- **Task A**: Gender Classification (Binary) - 99.79% accuracy achieved
- **Task B**: Face Recognition (Identity Matching) - 72.00% accuracy with embedding-based approach

## Key Innovations

### 1. Advanced Loss Function Design
- **Alignment-Uniformity Loss**: Novel combination of triplet loss with alignment and uniformity objectives for robust embedding learning
- **Weighted Cross-Entropy**: Handles class imbalance in gender classification with computed class weights
- **Numerical Stability**: MPS/GPU-compatible implementations with NaN handling and gradient clipping

### 2. Robust Architecture Choices
- **Task A**: ResNet34 backbone with custom dropout-regularized classifier
- **Task B**: Embedding-based Siamese network with L2-normalized 512D representations
- **Threshold Optimization**: Data-driven threshold selection for optimal precision-recall balance

### 3. Advanced Training Strategies
- **Hard Negative Mining**: Intelligent triplet selection for face recognition
- **Decorrelated Batch Normalization**: Prevents dimensional collapse in embedding space
- **Class-Balanced Sampling**: Ensures fair representation during training

## Model Architecture

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

## Performance Results

### Task A: Gender Classification
- **Overall Accuracy**: 99.79%
- **Overall Precision**: 99.24%
- **Overall Recall**: 99.75%
- **Overall F1-Score**: 99.49%
- **ROC AUC**: 100.00%

**Per-Class Performance:**
- **Male**: Precision: 99.93%, Recall: 99.80%, F1: 99.87%
- **Female**: Precision: 99.24%, Recall: 99.75%, F1: 99.49%

### Task B: Face Recognition
- **Optimal Threshold**: 0.975
- **Best Accuracy**: 72.00%
- **Best F1-Score**: 72.44%
- **Precision at Optimal**: 71.32%

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Official Evaluation (For Judges)
```bash
python run_evaluation.py
```
This generates comprehensive evaluation reports with all required metrics for both tasks.

### Training Models
```bash
# Task A - Gender Classification
python train_a.py

# Task B - Face Recognition
python train_b.py
```

### Individual Testing
```bash
# Comprehensive analysis
python run_all_tests.py

# Task-specific testing
python test_gender_model.py    # Task A analysis
python test_model.py          # Task B analysis
```

## Technical Approach

### Problem Analysis
The core challenge lies in maintaining consistent performance across diverse visual degradations. Traditional approaches fail because they rely on idealized training conditions. Our solution addresses this through:

1. **Robust Feature Learning**: Pretrained ResNet backbones provide strong initial representations
2. **Adaptive Loss Design**: Custom loss functions that explicitly handle adversarial conditions
3. **Embedding-Based Matching**: Face recognition as similarity learning rather than classification
4. **Threshold Optimization**: Data-driven decision boundaries for optimal performance

### Implementation Highlights
- **Cross-platform Compatibility**: Supports CUDA, MPS, and CPU
- **Numerical Stability**: Robust implementations with gradient clipping and NaN handling
- **Reproducible Results**: Fixed random seeds and deterministic operations
- **Comprehensive Logging**: Detailed training and evaluation tracking

## Code Structure
```
models/                    # Model architectures
├── gender_model.py       # Task A model
└── face_model.py         # Task B model

utils/                    # Dataset utilities
├── gender_dataset.py    # Task A data handling
└── face_dataset.py      # Task B data handling

train_a.py               # Gender classification training
train_b.py               # Face recognition training
test_gender_model.py     # Task A evaluation
test_model.py            # Task B evaluation
run_evaluation.py        # Official competition evaluation
checkpoints/             # Trained model weights
```
