#type: ignore
#!/usr/bin/env python3
"""
Quick Gender Classification Demo Script
Shows predictions on sample images with confidence scores.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import argparse
from models.gender_model import GenderClassifier
from utils.gender_dataset import GenderDataset

def show_predictions(model, dataset, device, num_examples=15):
    """Show quick prediction examples"""
    print("=== GENDER CLASSIFICATION PREDICTIONS ===")

    model.eval()
    class_names = ['Male', 'Female']

    # Get random samples
    indices = np.random.choice(len(dataset), num_examples, replace=False)

    correct_count = 0

    with torch.no_grad():
        for i, idx in enumerate(indices):
            image, true_label = dataset[idx]
            image_batch = image.unsqueeze(0).to(device)

            # Get prediction
            output = model(image_batch)
            probabilities = F.softmax(output, dim=1)
            predicted_label = torch.argmax(output, dim=1).item()
            confidence = probabilities[0, predicted_label].item()

            # Check if correct
            is_correct = (predicted_label == true_label)
            if is_correct:
                correct_count += 1

            status = "✅ CORRECT" if is_correct else "❌ WRONG"

            print(f"Example {i+1:2d}: {status}")
            print(f"  True: {class_names[true_label]:6s} | Predicted: {class_names[predicted_label]:6s} | Confidence: {confidence:.4f}")

            # Show probabilities
            male_prob = probabilities[0, 0].item()
            female_prob = probabilities[0, 1].item()
            print(f"  Probabilities -> Male: {male_prob:.4f}, Female: {female_prob:.4f}")
            print()

    accuracy = correct_count / num_examples
    print(f"Accuracy on these {num_examples} examples: {accuracy:.2%} ({correct_count}/{num_examples})")

def main():
    parser = argparse.ArgumentParser(description='Demo Gender Classification')
    parser.add_argument('--model_path', type=str, default='checkpoints/gender_model.pt',
                        help='Path to the trained model')
    parser.add_argument('--data_dir', type=str, default='data/facecom/Task_A/',
                        help='Path to data directory')
    parser.add_argument('--num_examples', type=int, default=15,
                        help='Number of examples to show')
    parser.add_argument('--use_val_set', action='store_true',
                        help='Use validation set instead of train set')

    args = parser.parse_args()

    # Device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print("Loading model...")
    model = GenderClassifier(num_classes=2, dropout_rate=0.3).to(device)

    if not os.path.exists(args.model_path):
        print(f"❌ ERROR: Model file not found: {args.model_path}")
        return

    try:
        state_dict = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state_dict)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ ERROR: Failed to load model: {e}")
        return

    # Load dataset
    print("Loading dataset...")
    split = 'val' if args.use_val_set else 'train'
    dataset = GenderDataset(args.data_dir, split=split, oversample_female=False)

    print(f"Dataset: {split} split with {len(dataset)} samples")
    class_dist = dataset.get_class_distribution()
    print(f"Distribution: Male={class_dist.get(0, 0)}, Female={class_dist.get(1, 0)}")
    print()

    # Show predictions
    show_predictions(model, dataset, device, args.num_examples)

if __name__ == "__main__":
    main()
