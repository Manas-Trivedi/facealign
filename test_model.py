#type: ignore
#!/usr/bin/env python3
"""
Comprehensive Face Recognition Model Testing Script
Tests the trained face embedding model with detailed logging and analysis.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import roc_curve, auc, confusion_matrix
import argparse
from tqdm import tqdm
import time
import logging
from datetime import datetime

from models.face_model import FaceEmbeddingModel, cosine_similarity
from utils.face_dataset import ValidationDataset, get_transforms

# Setup logging
def setup_logging(log_file):
    """Setup comprehensive logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def compute_embeddings_batch(model, images, device, batch_size=32):
    """Compute embeddings for a list of images with batch processing"""
    model.eval()
    embeddings = []

    logger.info(f"Computing embeddings for {len(images)} images in batches of {batch_size}")

    with torch.no_grad():
        for i in tqdm(range(0, len(images), batch_size), desc="Computing embeddings"):
            batch = images[i:i+batch_size]
            batch_tensor = torch.stack(batch).to(device)
            batch_embeddings = model(batch_tensor)

            # Check for NaN embeddings
            if torch.isnan(batch_embeddings).any():
                logger.error(f"NaN embeddings detected in batch {i//batch_size}")
                return None

            embeddings.append(batch_embeddings.cpu())

    return torch.cat(embeddings, dim=0)

def analyze_embeddings(embeddings, labels, logger):
    """Analyze embedding properties"""
    logger.info("=== EMBEDDING ANALYSIS ===")

    # Basic statistics
    mean_norm = torch.norm(embeddings, p=2, dim=1).mean().item()
    std_norm = torch.norm(embeddings, p=2, dim=1).std().item()
    logger.info(f"Embedding L2 norms - Mean: {mean_norm:.4f}, Std: {std_norm:.4f}")

    # Check for NaN/Inf
    nan_count = torch.isnan(embeddings).sum().item()
    inf_count = torch.isinf(embeddings).sum().item()
    logger.info(f"NaN values: {nan_count}, Inf values: {inf_count}")

    if nan_count > 0 or inf_count > 0:
        logger.error("❌ Embeddings contain NaN or Inf values!")
        return False

    # Embedding distribution
    flat_embeddings = embeddings.flatten()
    logger.info(f"Embedding values - Min: {flat_embeddings.min():.4f}, Max: {flat_embeddings.max():.4f}")
    logger.info(f"Embedding values - Mean: {flat_embeddings.mean():.4f}, Std: {flat_embeddings.std():.4f}")

    # Pairwise similarity analysis
    if len(embeddings) > 1:
        # Sample some pairs for analysis
        sample_size = min(100, len(embeddings))
        sample_indices = torch.randperm(len(embeddings))[:sample_size]
        sample_embeddings = embeddings[sample_indices]

        # Compute pairwise similarities
        similarities = F.cosine_similarity(
            sample_embeddings.unsqueeze(1),
            sample_embeddings.unsqueeze(0),
            dim=2
        )

        # Remove diagonal (self-similarities)
        mask = ~torch.eye(len(sample_embeddings), dtype=torch.bool)
        off_diagonal_sims = similarities[mask]

        logger.info(f"Pairwise similarities - Min: {off_diagonal_sims.min():.4f}, Max: {off_diagonal_sims.max():.4f}")
        logger.info(f"Pairwise similarities - Mean: {off_diagonal_sims.mean():.4f}, Std: {off_diagonal_sims.std():.4f}")

    logger.info("✅ Embedding analysis completed successfully")
    return True

def comprehensive_validation(model, val_dataset, device, thresholds, num_queries=500, logger=None):
    """Comprehensive validation with multiple metrics and threshold analysis"""
    logger.info("=== COMPREHENSIVE VALIDATION ===")

    # Get gallery embeddings
    logger.info("Computing gallery embeddings...") #type: ignore
    gallery_imgs, gallery_identities = val_dataset.get_gallery()
    gallery_embeddings = compute_embeddings_batch(model, gallery_imgs, device)

    if gallery_embeddings is None:
        logger.error("Failed to compute gallery embeddings") #type: ignore
        return None

    logger.info(f"Gallery: {len(gallery_embeddings)} embeddings computed") #type: ignore

    # Analyze gallery embeddings
    if not analyze_embeddings(gallery_embeddings, gallery_identities, logger):
        logger.error("Gallery embeddings failed analysis")
        return None

    # Create validation queries
    logger.info(f"Creating {num_queries} validation queries...")
    query_imgs, query_labels, query_info = val_dataset.create_validation_batch(num_queries)
    query_embeddings = compute_embeddings_batch(model, query_imgs, device)

    if query_embeddings is None:
        logger.error("Failed to compute query embeddings")
        return None

    logger.info(f"Queries: {len(query_embeddings)} embeddings computed")

    # Analyze query embeddings
    if not analyze_embeddings(query_embeddings, query_labels, logger):
        logger.error("Query embeddings failed analysis")
        return None

    # Compute all similarities
    logger.info("Computing similarities between queries and gallery...")
    all_similarities = []
    all_predictions = {thresh: [] for thresh in thresholds}
    best_matches = []

    for i, query_emb in enumerate(tqdm(query_embeddings, desc="Processing queries")):
        # Compute similarities with all gallery images
        similarities = cosine_similarity(
            query_emb.unsqueeze(0).repeat(len(gallery_embeddings), 1),
            gallery_embeddings
        )

        # Check for NaN similarities
        if torch.isnan(similarities).any():
            logger.error(f"NaN similarities detected for query {i}")
            return None

        # Best match
        max_similarity = similarities.max().item()
        best_match_idx = similarities.argmax().item()
        best_match_identity = gallery_identities[best_match_idx]

        all_similarities.append(max_similarity)
        best_matches.append(best_match_identity)

        # Predictions for all thresholds
        for thresh in thresholds:
            pred = 1 if max_similarity > thresh else 0
            all_predictions[thresh].append(pred)

    # Analyze similarities
    logger.info("=== SIMILARITY ANALYSIS ===")
    logger.info(f"Similarity range: {min(all_similarities):.4f} - {max(all_similarities):.4f}")
    logger.info(f"Similarity mean: {np.mean(all_similarities):.4f} ± {np.std(all_similarities):.4f}")

    # Separate by ground truth
    pos_similarities = [all_similarities[i] for i in range(len(all_similarities)) if query_labels[i] == 1]
    neg_similarities = [all_similarities[i] for i in range(len(all_similarities)) if query_labels[i] == 0]

    logger.info(f"Positive pairs similarity: {np.mean(pos_similarities):.4f} ± {np.std(pos_similarities):.4f}")
    logger.info(f"Negative pairs similarity: {np.mean(neg_similarities):.4f} ± {np.std(neg_similarities):.4f}")
    logger.info(f"Similarity gap: {np.mean(pos_similarities) - np.mean(neg_similarities):.4f}")

    # Compute metrics for all thresholds
    results = []
    logger.info("\n=== THRESHOLD ANALYSIS ===")
    logger.info("Threshold | Accuracy | Precision | Recall | F1-Score")
    logger.info("-" * 50)

    best_f1 = 0.0
    best_threshold = thresholds[0]

    for thresh in thresholds:
        predictions = all_predictions[thresh]

        accuracy = accuracy_score(query_labels, predictions)
        precision = precision_score(query_labels, predictions, average='binary', zero_division=0)
        recall = recall_score(query_labels, predictions, average='binary', zero_division=0)
        f1 = f1_score(query_labels, predictions, average='binary', zero_division=0)

        logger.info(f"{thresh:8.3f} | {accuracy:8.4f} | {precision:9.4f} | {recall:6.4f} | {f1:8.4f}")

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

    logger.info(f"\n🏆 Best threshold: {best_threshold} with F1: {best_f1:.4f}")

    # Detailed analysis for best threshold
    logger.info(f"\n=== DETAILED ANALYSIS (Threshold: {best_threshold}) ===")
    best_predictions = all_predictions[best_threshold]

    # Confusion matrix
    cm = confusion_matrix(query_labels, best_predictions)
    tn, fp, fn, tp = cm.ravel()

    logger.info(f"Confusion Matrix:")
    logger.info(f"  True Negatives:  {tn:4d}  |  False Positives: {fp:4d}")
    logger.info(f"  False Negatives: {fn:4d}  |  True Positives:  {tp:4d}")

    # Per-split analysis
    val_indices = [i for i, info in enumerate(query_info) if info['split'] == 'val']
    train_indices = [i for i, info in enumerate(query_info) if info['split'] == 'train']

    if val_indices:
        val_labels = [query_labels[i] for i in val_indices]
        val_preds = [best_predictions[i] for i in val_indices]
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='binary', zero_division=0)
        logger.info(f"Validation split accuracy: {val_acc:.4f}, F1: {val_f1:.4f}")

    if train_indices:
        train_labels = [query_labels[i] for i in train_indices]
        train_preds = [best_predictions[i] for i in train_indices]
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, average='binary', zero_division=0)
        logger.info(f"Train split accuracy: {train_acc:.4f}, F1: {train_f1:.4f}")

    # ROC analysis
    logger.info("\n=== ROC ANALYSIS ===")
    fpr, tpr, roc_thresholds = roc_curve(query_labels, all_similarities)
    roc_auc = auc(fpr, tpr)
    logger.info(f"ROC AUC: {roc_auc:.4f}")

    return {
        'results': results,
        'best_threshold': best_threshold,
        'best_f1': best_f1,
        'similarities': all_similarities,
        'labels': query_labels,
        'query_info': query_info,
        'roc_auc': roc_auc,
        'fpr': fpr,
        'tpr': tpr,
        'roc_thresholds': roc_thresholds
    }

def save_results(results, output_dir, logger):
    """Save results to files"""
    logger.info("=== SAVING RESULTS ===")

    os.makedirs(output_dir, exist_ok=True)

    # Save metrics table as simple text file
    txt_path = os.path.join(output_dir, 'threshold_analysis.txt')
    with open(txt_path, 'w') as f:
        f.write("Threshold,Accuracy,Precision,Recall,F1\n")
        for r in results['results']:
            f.write(f"{r['threshold']:.4f},{r['accuracy']:.4f},{r['precision']:.4f},{r['recall']:.4f},{r['f1']:.4f}\n")
    logger.info(f"Threshold analysis saved to: {txt_path}")

    # Save plots
    try:
        plt.style.use('default')  # Use default style instead of seaborn
    except:
        pass

    # 1. Threshold vs Metrics plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Model Performance Analysis', fontsize=16)

    thresholds = [r['threshold'] for r in results['results']]

    # Accuracy
    axes[0, 0].plot(thresholds, [r['accuracy'] for r in results['results']], 'b-o')
    axes[0, 0].set_title('Accuracy vs Threshold')
    axes[0, 0].set_xlabel('Threshold')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].grid(True)

    # F1 Score
    axes[0, 1].plot(thresholds, [r['f1'] for r in results['results']], 'g-o')
    axes[0, 1].axvline(results['best_threshold'], color='r', linestyle='--', alpha=0.7, label=f"Best: {results['best_threshold']}")
    axes[0, 1].set_title('F1 Score vs Threshold')
    axes[0, 1].set_xlabel('Threshold')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Precision and Recall
    axes[1, 0].plot(thresholds, [r['precision'] for r in results['results']], 'r-o', label='Precision')
    axes[1, 0].plot(thresholds, [r['recall'] for r in results['results']], 'b-o', label='Recall')
    axes[1, 0].set_title('Precision and Recall vs Threshold')
    axes[1, 0].set_xlabel('Threshold')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # ROC Curve
    axes[1, 1].plot(results['fpr'], results['tpr'], 'b-', linewidth=2, label=f'ROC (AUC = {results["roc_auc"]:.3f})')
    axes[1, 1].plot([0, 1], [0, 1], 'r--', alpha=0.7, label='Random')
    axes[1, 1].set_title('ROC Curve')
    axes[1, 1].set_xlabel('False Positive Rate')
    axes[1, 1].set_ylabel('True Positive Rate')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'performance_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Performance plots saved to: {plot_path}")
    plt.close()

    # 2. Similarity distribution plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Separate similarities by ground truth
    pos_similarities = [results['similarities'][i] for i in range(len(results['similarities'])) if results['labels'][i] == 1]
    neg_similarities = [results['similarities'][i] for i in range(len(results['similarities'])) if results['labels'][i] == 0]

    # Histogram
    axes[0].hist(pos_similarities, bins=50, alpha=0.7, label='Positive pairs', color='green')
    axes[0].hist(neg_similarities, bins=50, alpha=0.7, label='Negative pairs', color='red')
    axes[0].axvline(results['best_threshold'], color='blue', linestyle='--', label=f'Best threshold: {results["best_threshold"]}')
    axes[0].set_title('Similarity Distribution')
    axes[0].set_xlabel('Cosine Similarity')
    axes[0].set_ylabel('Count')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Box plot
    axes[1].boxplot([pos_similarities, neg_similarities], labels=['Positive', 'Negative'])
    axes[1].set_title('Similarity Distribution by Ground Truth')
    axes[1].set_ylabel('Cosine Similarity')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    dist_plot_path = os.path.join(output_dir, 'similarity_distribution.png')
    plt.savefig(dist_plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Similarity distribution plots saved to: {dist_plot_path}")
    plt.close()

    logger.info("✅ All results saved successfully")

def main():
    parser = argparse.ArgumentParser(description='Test Face Recognition Model')
    parser.add_argument('--model_path', type=str, default='checkpoints/final_model.pth',
                        help='Path to the trained model')
    parser.add_argument('--data_dir', type=str, default='data/facecom/Task_B/',
                        help='Path to data directory')
    parser.add_argument('--embedding_dim', type=int, default=256,
                        help='Embedding dimension')
    parser.add_argument('--backbone', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50'],
                        help='Backbone architecture')
    parser.add_argument('--num_queries', type=int, default=1000,
                        help='Number of validation queries')
    parser.add_argument('--output_dir', type=str, default='test_results/',
                        help='Directory to save test results')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for embedding computation')

    args = parser.parse_args()

    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, f'test_log_{timestamp}.txt')

    global logger
    logger = setup_logging(log_file)

    logger.info("=== FACE RECOGNITION MODEL TESTING ===")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")

    # Device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load model
    logger.info("Loading model...")
    model = FaceEmbeddingModel(
        embedding_dim=args.embedding_dim,
        backbone=args.backbone,
        pretrained=False  # We're loading trained weights
    ).to(device)

    if not os.path.exists(args.model_path):
        logger.error(f"Model file not found: {args.model_path}")
        return

    try:
        state_dict = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state_dict)
        logger.info("✅ Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # Load dataset
    logger.info("Loading validation dataset...")
    val_transform = get_transforms(train=False)
    val_dataset = ValidationDataset(args.data_dir, transform=val_transform)

    logger.info(f"Gallery size: {len(val_dataset.gallery_images)}")
    logger.info(f"Positive queries available: {len(val_dataset.positive_queries)}")
    logger.info(f"Negative queries available: {len(val_dataset.negative_queries)}")

    # Test thresholds
    thresholds = [0.95, 0.96, 0.97, 0.975, 0.98, 0.985, 0.99, 0.992, 0.994, 0.995, 0.996, 0.997, 0.998, 0.999, 0.9995]

    # Run comprehensive validation
    start_time = time.time()
    results = comprehensive_validation(
        model, val_dataset, device, thresholds,
        num_queries=args.num_queries, logger=logger
    )
    test_time = time.time() - start_time

    if results is None:
        logger.error("❌ Testing failed!")
        return

    logger.info(f"\n🎉 Testing completed in {test_time:.1f} seconds")
    logger.info(f"🏆 Best performance: F1={results['best_f1']:.4f} at threshold={results['best_threshold']}")
    logger.info(f"📊 ROC AUC: {results['roc_auc']:.4f}")

    # Save results
    save_results(results, args.output_dir, logger)

    logger.info(f"\n📝 Complete log saved to: {log_file}")
    logger.info("✅ Testing completed successfully!")

if __name__ == "__main__":
    main()
