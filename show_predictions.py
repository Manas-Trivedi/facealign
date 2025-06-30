#!/usr/bin/env python3
"""
Quick demo script to show model predictions on sample images
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse

from models.face_model import FaceEmbeddingModel, cosine_similarity
from utils.face_dataset import ValidationDataset, get_transforms

def show_prediction_examples(model, val_dataset, device, threshold=0.95, num_examples=10):
    """Show some example predictions with images"""

    print("=== PREDICTION EXAMPLES ===")

    # Get gallery embeddings
    gallery_imgs, gallery_identities = val_dataset.get_gallery()
    gallery_embeddings = []

    model.eval()
    with torch.no_grad():
        for i in range(0, len(gallery_imgs), 32):
            batch = gallery_imgs[i:i+32]
            batch_tensor = torch.stack(batch).to(device)
            batch_embeddings = model(batch_tensor)
            gallery_embeddings.append(batch_embeddings.cpu())

    gallery_embeddings = torch.cat(gallery_embeddings, dim=0)

    # Create some test queries
    query_imgs, query_labels, query_info = val_dataset.create_validation_batch(num_examples)

    # Compute query embeddings
    query_embeddings = []
    with torch.no_grad():
        for i in range(0, len(query_imgs), 32):
            batch = query_imgs[i:i+32]
            batch_tensor = torch.stack(batch).to(device)
            batch_embeddings = model(batch_tensor)
            query_embeddings.append(batch_embeddings.cpu())

    query_embeddings = torch.cat(query_embeddings, dim=0)

    print(f"Using threshold: {threshold}")
    print("=" * 80)

    correct_predictions = 0

    for i in range(num_examples):
        query_emb = query_embeddings[i]
        info = query_info[i]

        # Compute similarities with all gallery images
        similarities = cosine_similarity(
            query_emb.unsqueeze(0).repeat(len(gallery_embeddings), 1),
            gallery_embeddings
        )

        # Best match
        max_similarity = similarities.max().item()
        best_match_idx = similarities.argmax().item()
        best_match_identity = gallery_identities[best_match_idx]

        # Prediction
        prediction = 1 if max_similarity > threshold else 0
        ground_truth = query_labels[i]

        # Check if correct
        is_correct = (prediction == ground_truth)
        if is_correct:
            correct_predictions += 1

        # Status indicator
        status = "✅ CORRECT" if is_correct else "❌ WRONG"

        print(f"Example {i+1:2d}: {status}")
        print(f"  Query ID: {info['identity'][:30]:30s} | Type: {info['type']:9s} | Split: {info['split']}")
        print(f"  Ground Truth: {ground_truth} | Prediction: {prediction} | Similarity: {max_similarity:.4f}")
        print(f"  Best Match: {best_match_identity[:30]:30s}")

        # Additional analysis
        if ground_truth == 1:  # Positive pair
            query_identity = info['identity']
            if query_identity == best_match_identity:
                print(f"  ✅ Correctly matched to same identity")
            else:
                print(f"  ⚠️  Matched to different identity!")
        else:  # Negative pair
            if prediction == 0:
                print(f"  ✅ Correctly rejected (similarity below threshold)")
            else:
                print(f"  ❌ Incorrectly accepted (similarity above threshold)")

        print()

    accuracy = correct_predictions / num_examples
    print(f"Accuracy on these {num_examples} examples: {accuracy:.2%} ({correct_predictions}/{num_examples})")

def main():
    parser = argparse.ArgumentParser(description='Show Face Recognition Predictions')
    parser.add_argument('--model_path', type=str, default='models/final_model.pth',
                        help='Path to the trained model')
    parser.add_argument('--data_dir', type=str, default='data/facecom/Task_B/',
                        help='Path to data directory')
    parser.add_argument('--embedding_dim', type=int, default=256,
                        help='Embedding dimension')
    parser.add_argument('--backbone', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50'],
                        help='Backbone architecture')
    parser.add_argument('--threshold', type=float, default=0.95,
                        help='Similarity threshold for predictions')
    parser.add_argument('--num_examples', type=int, default=20,
                        help='Number of examples to show')

    args = parser.parse_args()

    # Device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print("Loading model...")
    model = FaceEmbeddingModel(
        embedding_dim=args.embedding_dim,
        backbone=args.backbone,
        pretrained=False
    ).to(device)

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
    print("Loading validation dataset...")
    val_transform = get_transforms(train=False)
    val_dataset = ValidationDataset(args.data_dir, transform=val_transform)

    print(f"Gallery size: {len(val_dataset.gallery_images)}")
    print(f"Available queries: {len(val_dataset.positive_queries) + len(val_dataset.negative_queries)}")
    print()

    # Show predictions
    show_prediction_examples(
        model, val_dataset, device,
        threshold=args.threshold,
        num_examples=args.num_examples
    )

if __name__ == "__main__":
    main()
