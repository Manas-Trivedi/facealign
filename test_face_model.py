#!/usr/bin/env python3
"""
Face Verification Model Testing Script
Tests the trained face embedding model for face verification tasks.
Compares clear/reference images with positive distortions and negative samples.
"""

import os
import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import argparse
from tqdm import tqdm
import time
from datetime import datetime

from models.face_model import FaceEmbeddingModel, cosine_similarity
from utils.face_dataset import get_transforms


class FaceVerificationDataset(Dataset):
    """Custom dataset class for face verification testing"""

    def __init__(self, data_dir, num_positive=3, num_negative=5):
        """
        Args:
            data_dir: Directory containing person folders with clear images and distortion subfolders
            num_positive: Number of positive samples per identity (from distortion folder)
            num_negative: Number of negative samples per identity (from other identities)

        Expected structure:
        data_dir/
        ├── person1/
        │   ├── clear_image.jpg (reference)
        │   └── distortion/
        │       ├── distorted1.jpg (positive sample)
        │       ├── distorted2.jpg (positive sample)
        │       └── ...
        ├── person2/
        │   ├── clear_image.jpg (reference)
        │   └── distortion/
        │       ├── distorted1.jpg (positive sample)
        │       └── ...
        └── ...
        """
        self.data_dir = data_dir
        self.num_positive = num_positive
        self.num_negative = num_negative
        self.samples = []

        # Define transforms
        self.transform = get_transforms(train=False)

        # Load verification pairs
        self._load_verification_pairs()

    def _load_verification_pairs(self):
        """Load verification pairs from the dataset"""
        print(f"Loading face verification dataset from: {self.data_dir}")

        # Structure: data_dir/identity_name/clear_image.jpg and data_dir/identity_name/distortion/distorted_images.jpg
        identity_dirs = [d for d in os.listdir(self.data_dir)
                        if os.path.isdir(os.path.join(self.data_dir, d))]

        for identity in identity_dirs:
            identity_path = os.path.join(self.data_dir, identity)

            # Find reference (clear) image in the person folder
            clear_images = [f for f in os.listdir(identity_path)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            if not clear_images:
                print(f"Warning: No clear image found for identity {identity}")
                continue

            # Use first clear image as reference
            reference_img = os.path.join(identity_path, clear_images[0])

            # Look for distortion folder
            distortion_path = os.path.join(identity_path, 'distortion')
            if not os.path.exists(distortion_path):
                print(f"Warning: No distortion folder found for identity {identity}")
                continue

            # Get distorted images for positive pairs
            distorted_images = [f for f in os.listdir(distortion_path)
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            if not distorted_images:
                print(f"Warning: No distorted images found for identity {identity}")
                continue

            # Create positive pairs with distorted images
            positive_imgs = distorted_images[:min(len(distorted_images), self.num_positive)]
            for pos_img in positive_imgs:
                pos_path = os.path.join(distortion_path, pos_img)
                self.samples.append((reference_img, pos_path, 1, identity))  # 1 = positive

            # Create negative pairs with other identities' clear images
            other_identities = [other for other in identity_dirs if other != identity]
            neg_count = 0

            for other_identity in other_identities:
                if neg_count >= self.num_negative:
                    break

                other_path = os.path.join(self.data_dir, other_identity)
                other_clear_images = [f for f in os.listdir(other_path)
                                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

                if other_clear_images:
                    neg_img = os.path.join(other_path, other_clear_images[0])
                    self.samples.append((reference_img, neg_img, 0, f"{identity}_vs_{other_identity}"))  # 0 = negative
                    neg_count += 1

        print(f"Verification dataset loaded:")
        print(f"  Positive pairs: {sum(1 for _, _, label, _ in self.samples if label == 1)}")
        print(f"  Negative pairs: {sum(1 for _, _, label, _ in self.samples if label == 0)}")
        print(f"  Total pairs: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ref_path, test_path, label, identity = self.samples[idx]

        try:
            # Load reference image
            ref_image = Image.open(ref_path).convert('RGB')
            ref_image = self.transform(ref_image)

            # Load test image
            test_image = Image.open(test_path).convert('RGB')
            test_image = self.transform(test_image)

            return ref_image, test_image, label, identity
        except Exception as e:
            print(f"Error loading images {ref_path}, {test_path}: {e}")
            # Return dummy images if loading fails
            dummy_image = torch.zeros(3, 224, 224)
            return dummy_image, dummy_image, label, identity

    def get_class_distribution(self):
        """Get class distribution for logging"""
        dist = {}
        for _, _, label, _ in self.samples:
            dist[label] = dist.get(label, 0) + 1
        return dist


def log_print(message, log_file=None):
    """Print and optionally log message"""
    print(message)
    if log_file:
        with open(log_file, 'a') as f:
            f.write(message + '\n')

def log_task_evaluation_criteria(log_file=None):
    """Log the official evaluation criteria for face verification"""
    log_print("=" * 80, log_file)
    log_print("FACE VERIFICATION TESTING", log_file)
    log_print("=" * 80, log_file)
    log_print("Objective: Verify if two face images belong to the same person", log_file)
    log_print("           by comparing embeddings of reference and test images.", log_file)
    log_print("", log_file)
    log_print("EVALUATION METRICS:", log_file)
    log_print("  • Accuracy    - Overall verification accuracy", log_file)
    log_print("  • Precision   - True positives / (True positives + False positives)", log_file)
    log_print("  • Recall      - True positives / (True positives + False negatives)", log_file)
    log_print("  • F1-Score    - Harmonic mean of Precision and Recall", log_file)
    log_print("  • ROC AUC     - Area under ROC curve", log_file)
    log_print("=" * 80, log_file)

def compute_face_verification(model, test_loader, device, log_file=None):
    """Compute face verification results"""
    log_print("=== FACE VERIFICATION EVALUATION ===", log_file)

    model.eval()
    all_similarities = []
    all_labels = []
    all_identities = []

    log_print(f"Evaluating on {len(test_loader)} batches...", log_file)

    with torch.no_grad():
        for batch_idx, (ref_images, test_images, labels, identities) in enumerate(tqdm(test_loader, desc="Computing similarities")):
            ref_images = ref_images.to(device)
            test_images = test_images.to(device)

            # Get embeddings
            ref_embeddings = model(ref_images)
            test_embeddings = model(test_images)

            # Compute similarities
            similarities = cosine_similarity(ref_embeddings, test_embeddings)

            # Store results
            all_similarities.extend(similarities.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_identities.extend(identities)

    all_similarities = np.array(all_similarities)
    all_labels = np.array(all_labels)

    log_print(f"Verification completed on {len(all_labels)} pairs", log_file)

    return all_similarities, all_labels, all_identities

def analyze_verification_performance(similarities, labels, identities, log_file=None):
    """Analyze face verification performance"""
    log_print("\n" + "=" * 60, log_file)
    log_print("FACE VERIFICATION RESULTS", log_file)
    log_print("=" * 60, log_file)

    # Similarity analysis
    log_print("SIMILARITY ANALYSIS:", log_file)
    log_print(f"  Similarity range: {np.min(similarities):.4f} - {np.max(similarities):.4f}", log_file)
    log_print(f"  Mean similarity: {np.mean(similarities):.4f} ± {np.std(similarities):.4f}", log_file)

    # Separate by ground truth
    positive_similarities = similarities[labels == 1]
    negative_similarities = similarities[labels == 0]

    log_print(f"  Positive pairs: {np.mean(positive_similarities):.4f} ± {np.std(positive_similarities):.4f}", log_file)
    log_print(f"  Negative pairs: {np.mean(negative_similarities):.4f} ± {np.std(negative_similarities):.4f}", log_file)
    log_print(f"  Separation: {np.mean(positive_similarities) - np.mean(negative_similarities):.4f}", log_file)

    # Threshold analysis
    thresholds = np.arange(0.1, 1.0, 0.05)
    results = []

    log_print("\nTHRESHOLD ANALYSIS:", log_file)
    log_print("Threshold | Accuracy | Precision | Recall | F1-Score", log_file)
    log_print("-" * 50, log_file)

    best_f1 = 0.0
    best_threshold = 0.5

    for thresh in thresholds:
        predictions = (similarities >= thresh).astype(int)

        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, average='binary', zero_division=0)
        recall = recall_score(labels, predictions, average='binary', zero_division=0)
        f1 = f1_score(labels, predictions, average='binary', zero_division=0)

        log_print(f"{thresh:8.2f} | {accuracy:8.4f} | {precision:9.4f} | {recall:6.4f} | {f1:8.4f}", log_file)

        results.append({
            'threshold': thresh,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh

    # ROC Analysis
    fpr, tpr, _ = roc_curve(labels, similarities)
    roc_auc = auc(fpr, tpr)

    log_print(f"\nBest threshold: {best_threshold:.2f} with F1: {best_f1:.4f}", log_file)
    log_print(f"ROC AUC: {roc_auc:.4f}", log_file)

    # Confusion Matrix for best threshold
    best_predictions = (similarities >= best_threshold).astype(int)
    cm = confusion_matrix(labels, best_predictions)

    log_print("\nCONFUSION MATRIX (Best Threshold):", log_file)
    log_print(f"              Predicted", log_file)
    log_print(f"           Same  Different", log_file)
    log_print(f"Actual Same  {cm[1,1]:4d}   {cm[1,0]:4d}", log_file)
    log_print(f"   Different {cm[0,1]:4d}   {cm[0,0]:4d}", log_file)

    log_print("=" * 60, log_file)

    return {
        'threshold_results': results,
        'best_threshold': best_threshold,
        'best_f1': best_f1,
        'roc_auc': roc_auc,
        'fpr': fpr,
        'tpr': tpr,
        'confusion_matrix': cm,
        'similarities': similarities,
        'labels': labels
    }

def save_verification_results(results, output_dir, log_file=None):
    """Save verification results and create visualizations"""
    log_print("=== SAVING RESULTS ===", log_file)

    os.makedirs(output_dir, exist_ok=True)

    # Save metrics summary
    metrics_path = os.path.join(output_dir, 'verification_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("FACE VERIFICATION EVALUATION RESULTS\n")
        f.write("=" * 60 + "\n")
        f.write("Objective: Verify if two face images belong to the same person\n")
        f.write("           by comparing embeddings of reference and test images\n\n")

        f.write("PERFORMANCE METRICS:\n")
        best_result = next((r for r in results['threshold_results'] if r['threshold'] == results['best_threshold']), results['threshold_results'][-1])
        f.write(f"Best Threshold: {results['best_threshold']:.2f}\n")
        f.write(f"Accuracy:  {best_result['accuracy']:.4f} ({best_result['accuracy']*100:.2f}%)\n")
        f.write(f"Precision: {best_result['precision']:.4f} ({best_result['precision']*100:.2f}%)\n")
        f.write(f"Recall:    {best_result['recall']:.4f} ({best_result['recall']*100:.2f}%)\n")
        f.write(f"F1-Score:  {best_result['f1']:.4f} ({best_result['f1']*100:.2f}%)\n")
        f.write(f"ROC AUC:   {results['roc_auc']:.4f}\n")

    log_print(f"Verification metrics saved to: {metrics_path}", log_file)

    # Save threshold analysis
    thresh_path = os.path.join(output_dir, 'threshold_analysis.txt')
    with open(thresh_path, 'w') as f:
        f.write("Threshold,Accuracy,Precision,Recall,F1\n")
        for r in results['threshold_results']:
            f.write(f"{r['threshold']:.2f},{r['accuracy']:.4f},{r['precision']:.4f},{r['recall']:.4f},{r['f1']:.4f}\n")

    log_print(f"Threshold analysis saved to: {thresh_path}", log_file)

    # Create visualizations
    try:
        plt.style.use('default')

        # 1. Performance analysis plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Face Verification Performance Analysis', fontsize=16)

        # Threshold vs metrics
        thresholds = [r['threshold'] for r in results['threshold_results']]

        axes[0, 0].plot(thresholds, [r['accuracy'] for r in results['threshold_results']], 'b-o', label='Accuracy')
        axes[0, 0].plot(thresholds, [r['f1'] for r in results['threshold_results']], 'r-o', label='F1-Score')
        axes[0, 0].axvline(results['best_threshold'], color='black', linestyle='--', alpha=0.7, label=f'Best: {results["best_threshold"]:.2f}')
        axes[0, 0].set_title('Performance vs Threshold')
        axes[0, 0].set_xlabel('Threshold')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # ROC Curve
        axes[0, 1].plot(results['fpr'], results['tpr'], 'b-', linewidth=2, label=f'ROC (AUC = {results["roc_auc"]:.3f})')
        axes[0, 1].plot([0, 1], [0, 1], 'r--', alpha=0.7, label='Random')
        axes[0, 1].set_title('ROC Curve')
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Similarity distributions
        pos_similarities = results['similarities'][results['labels'] == 1]
        neg_similarities = results['similarities'][results['labels'] == 0]

        axes[1, 0].hist(pos_similarities, bins=30, alpha=0.7, label='Same Person', color='green')
        axes[1, 0].hist(neg_similarities, bins=30, alpha=0.7, label='Different Person', color='red')
        axes[1, 0].axvline(results['best_threshold'], color='blue', linestyle='--', label=f'Best Threshold: {results["best_threshold"]:.2f}')
        axes[1, 0].set_title('Similarity Distribution')
        axes[1, 0].set_xlabel('Cosine Similarity')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Confusion Matrix
        cm = results['confusion_matrix']
        im = axes[1, 1].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        axes[1, 1].set_title('Confusion Matrix')
        axes[1, 1].set_xlabel('Predicted')
        axes[1, 1].set_ylabel('Actual')
        axes[1, 1].set_xticks([0, 1])
        axes[1, 1].set_yticks([0, 1])
        axes[1, 1].set_xticklabels(['Different', 'Same'])
        axes[1, 1].set_yticklabels(['Different', 'Same'])

        # Add text annotations
        for i in range(2):
            for j in range(2):
                axes[1, 1].text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=14)

        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'verification_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        log_print(f"Verification plots saved to: {plot_path}", log_file)
        plt.close()

        log_print("All visualizations saved successfully", log_file)

    except Exception as e:
        log_print(f"Warning: Could not create visualizations: {e}", log_file)

def main():
    parser = argparse.ArgumentParser(description='Test Face Verification Model')
    parser.add_argument('--model_path', type=str, default='checkpoints/final_model.pt',
                        help='Path to the trained model')
    parser.add_argument('--data_dir', type=str, default='Comsys-Hackathon5/Task_B/val',
                        help='Path to test data directory (containing identity subfolders)')
    parser.add_argument('--embedding_dim', type=int, default=512,
                        help='Embedding dimension')
    parser.add_argument('--backbone', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50'],
                        help='Backbone architecture')
    parser.add_argument('--num_positive', type=int, default=3,
                        help='Number of positive samples per identity')
    parser.add_argument('--num_negative', type=int, default=5,
                        help='Number of negative samples per identity')
    parser.add_argument('--output_dir', type=str, default='test_results_face/',
                        help='Directory to save test results')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')

    args = parser.parse_args()

    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, f'face_verification_log_{timestamp}.txt')

    log_print("=== FACE VERIFICATION MODEL TESTING ===", log_file)
    log_print(f"Model path: {args.model_path}", log_file)
    log_print(f"Data directory: {args.data_dir}", log_file)
    log_print(f"Output directory: {args.output_dir}", log_file)

    # Log the official evaluation criteria
    log_task_evaluation_criteria(log_file)

    # Device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_print(f"Using device: {device}", log_file)

    # Load model
    log_print("Loading model...", log_file)
    model = FaceEmbeddingModel(
        embedding_dim=args.embedding_dim,
        backbone=args.backbone,
        pretrained=False  # We're loading trained weights
    ).to(device)

    if not os.path.exists(args.model_path):
        log_print(f"ERROR: Model file not found: {args.model_path}", log_file)
        return

    try:
        state_dict = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state_dict)
        log_print("Model loaded successfully", log_file)
    except Exception as e:
        log_print(f"ERROR: Failed to load model: {e}", log_file)
        return

    # Load verification dataset
    log_print("Loading face verification dataset...", log_file)

    # Create verification dataset
    verification_dataset = FaceVerificationDataset(
        args.data_dir,
        num_positive=args.num_positive,
        num_negative=args.num_negative
    )

    verification_loader = torch.utils.data.DataLoader(
        verification_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0  # Set to 0 to avoid multiprocessing issues
    )

    log_print(f"Verification dataset loaded", log_file)
    log_print(f"Total verification pairs: {len(verification_dataset)}", log_file)
    class_dist = verification_dataset.get_class_distribution()
    log_print(f"Class distribution: Same={class_dist.get(1, 0)}, Different={class_dist.get(0, 0)}", log_file)

    # Run face verification
    start_time = time.time()

    similarities, labels, identities = compute_face_verification(
        model, verification_loader, device, log_file
    )

    # Analyze verification performance
    results = analyze_verification_performance(similarities, labels, identities, log_file)

    test_time = time.time() - start_time

    log_print(f"\nFace verification completed in {test_time:.1f} seconds", log_file)
    log_print(f"Best performance: F1={results['best_f1']:.4f} at threshold {results['best_threshold']:.2f}", log_file)
    log_print(f"ROC AUC: {results['roc_auc']:.4f}", log_file)

    # Save results
    save_verification_results(results, args.output_dir, log_file)

    log_print(f"\nComplete log saved to: {log_file}", log_file)
    log_print("Face verification testing completed successfully!", log_file)

if __name__ == "__main__":
    main()
