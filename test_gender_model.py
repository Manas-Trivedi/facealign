#type: ignore
#!/usr/bin/env python3
"""
Comprehensive Gender Classification Model Testing Script
Tests the trained gender classification model with detailed logging and analysis.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import argparse
from tqdm import tqdm
import time
from datetime import datetime

from models.gender_model import GenderClassifier
from utils.gender_dataset import GenderDataset

def log_print(message, log_file=None):
    """Print and optionally log message"""
    print(message)
    if log_file:
        with open(log_file, 'a') as f:
            f.write(message + '\n')

def log_task_evaluation_criteria(log_file=None):
    """Log the official evaluation criteria for Task A"""
    log_print("=" * 80, log_file)
    log_print("TASK A - GENDER CLASSIFICATION", log_file)
    log_print("=" * 80, log_file)
    log_print("Objective: Predict the gender (Male/Female) of a subject from a face image", log_file)
    log_print("           captured under visually degraded conditions.", log_file)
    log_print("", log_file)
    log_print("OFFICIAL EVALUATION METRICS:", log_file)
    log_print("  • Accuracy    - Overall classification accuracy", log_file)
    log_print("  • Precision   - True positives / (True positives + False positives)", log_file)
    log_print("  • Recall      - True positives / (True positives + False negatives)", log_file)
    log_print("  • F1-Score    - Harmonic mean of Precision and Recall", log_file)
    log_print("=" * 80, log_file)

def evaluate_model_comprehensive(model, test_loader, device, split_name, log_file=None):
    """Comprehensive model evaluation"""
    log_print(f"=== COMPREHENSIVE MODEL EVALUATION - {split_name.upper()} SET ===", log_file)

    model.eval()
    all_predictions = []
    all_probabilities = []
    all_labels = []
    all_features = []

    log_print(f"Evaluating on {len(test_loader)} batches...", log_file)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc=f"Evaluating {split_name}")):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)

            # Extract features for analysis
            features = model.get_features(images)

            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_features.extend(features.cpu().numpy())

    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    all_labels = np.array(all_labels)
    all_features = np.array(all_features)

    log_print(f"Evaluation completed on {len(all_labels)} samples", log_file)

    return all_predictions, all_probabilities, all_labels, all_features

def analyze_performance(predictions, probabilities, labels, split_name, log_file=None):
    """Analyze model performance with detailed metrics"""
    log_print("\n" + "=" * 60, log_file)
    log_print(f"OFFICIAL TASK A EVALUATION RESULTS - {split_name.upper()} SET", log_file)
    log_print("=" * 60, log_file)

    # OFFICIAL METRICS - Task A Requirements
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='binary')
    recall = recall_score(labels, predictions, average='binary')
    f1 = f1_score(labels, predictions, average='binary')

    log_print(f"OFFICIAL EVALUATION METRICS - {split_name.upper()}:", log_file)
    log_print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)", log_file)
    log_print(f"   Precision: {precision:.4f} ({precision*100:.2f}%)", log_file)
    log_print(f"   Recall:    {recall:.4f} ({recall*100:.2f}%)", log_file)
    log_print(f"   F1-Score:  {f1:.4f} ({f1*100:.2f}%)", log_file)
    log_print("", log_file)

    # Additional detailed analysis
    log_print("DETAILED PERFORMANCE ANALYSIS:", log_file)

    # Per-class metrics
    precision_per_class = precision_score(labels, predictions, average=None)
    recall_per_class = recall_score(labels, predictions, average=None)
    f1_per_class = f1_score(labels, predictions, average=None)

    class_names = ['Male', 'Female']
    log_print("\nPER-CLASS BREAKDOWN:", log_file)
    for i, class_name in enumerate(class_names):
        prec = precision_per_class[i] if hasattr(precision_per_class, '__getitem__') else precision_per_class
        rec = recall_per_class[i] if hasattr(recall_per_class, '__getitem__') else recall_per_class
        f1_val = f1_per_class[i] if hasattr(f1_per_class, '__getitem__') else f1_per_class
        log_print(f"   {class_name:6s}: Precision={prec:.4f}, Recall={rec:.4f}, F1={f1_val:.4f}", log_file)

    # Confusion Matrix
    cm = confusion_matrix(labels, predictions)
    log_print("\nCONFUSION MATRIX:", log_file)
    log_print(f"              Predicted", log_file)
    log_print(f"           Male  Female", log_file)
    log_print(f"Actual Male  {cm[0,0]:4d}   {cm[0,1]:4d}", log_file)
    log_print(f"    Female   {cm[1,0]:4d}   {cm[1,1]:4d}", log_file)

    # Classification Report
    log_print("\nDETAILED CLASSIFICATION REPORT:", log_file)
    report = classification_report(labels, predictions, target_names=class_names)
    log_print(report, log_file)

    # ROC Analysis
    female_probs = probabilities[:, 1]  # Probability of being female
    fpr, tpr, thresholds = roc_curve(labels, female_probs)
    roc_auc = auc(fpr, tpr)

    log_print(f"\nROC ANALYSIS:", log_file)
    log_print(f"   ROC AUC: {roc_auc:.4f}", log_file)

    # Confidence analysis
    log_print(f"\nCONFIDENCE ANALYSIS:", log_file)
    max_probs = np.max(probabilities, axis=1)
    log_print(f"   Average confidence: {np.mean(max_probs):.4f} ± {np.std(max_probs):.4f}", log_file)
    log_print(f"   Confidence range: {np.min(max_probs):.4f} - {np.max(max_probs):.4f}", log_file)

    # Confidence by class
    male_indices = labels == 0
    female_indices = labels == 1

    male_confidence = np.mean(max_probs[male_indices])
    female_confidence = np.mean(max_probs[female_indices])

    log_print(f"   Male predictions confidence: {male_confidence:.4f}", log_file)
    log_print(f"   Female predictions confidence: {female_confidence:.4f}", log_file)

    log_print("=" * 60, log_file)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'confusion_matrix': cm,
        'roc_auc': roc_auc,
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'probabilities': probabilities,
        'confidence': max_probs
    }

def analyze_features(features, labels, log_file=None):
    """Analyze learned features"""
    log_print("=== FEATURE ANALYSIS ===", log_file)

    # Feature statistics
    log_print(f"Feature dimensionality: {features.shape[1]}", log_file)
    log_print(f"Feature range: {np.min(features):.4f} - {np.max(features):.4f}", log_file)
    log_print(f"Feature mean: {np.mean(features):.4f} ± {np.std(features):.4f}", log_file)

    # Check for problematic values
    nan_count = np.isnan(features).sum()
    inf_count = np.isinf(features).sum()
    log_print(f"NaN values: {nan_count}, Inf values: {inf_count}", log_file)

    if nan_count > 0 or inf_count > 0:
        log_print("WARNING: Features contain NaN or Inf values!", log_file)
    else:
        log_print("Features are clean (no NaN/Inf values)", log_file)

    # Separate features by class
    male_features = features[labels == 0]
    female_features = features[labels == 1]

    log_print(f"\nMale samples: {len(male_features)}", log_file)
    log_print(f"Female samples: {len(female_features)}", log_file)

    # Feature means by class
    male_mean = np.mean(male_features, axis=0)
    female_mean = np.mean(female_features, axis=0)

    # Feature separation (how different are the class means)
    feature_separation = np.linalg.norm(male_mean - female_mean)
    log_print(f"Feature separation (L2 distance between class means): {feature_separation:.4f}", log_file)

    return {
        'male_features': male_features,
        'female_features': female_features,
        'separation': feature_separation,
        'nan_count': nan_count,
        'inf_count': inf_count
    }

def threshold_analysis(probabilities, labels, log_file=None):
    """Analyze performance at different decision thresholds"""
    log_print("=== THRESHOLD ANALYSIS ===", log_file)

    thresholds = np.arange(0.1, 1.0, 0.05)
    female_probs = probabilities[:, 1]  # Probability of being female

    results = []
    log_print("Threshold | Accuracy | Precision | Recall | F1-Score", log_file)
    log_print("-" * 50, log_file)

    best_f1 = 0.0
    best_threshold = 0.5

    for thresh in thresholds:
        # Make predictions based on threshold
        thresh_predictions = (female_probs >= thresh).astype(int)

        # Calculate metrics
        accuracy = accuracy_score(labels, thresh_predictions)
        precision = precision_score(labels, thresh_predictions, average='binary', zero_division=0)
        recall = recall_score(labels, thresh_predictions, average='binary', zero_division=0)
        f1 = f1_score(labels, thresh_predictions, average='binary', zero_division=0)

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

    log_print(f"\nBest threshold: {best_threshold:.2f} with F1: {best_f1:.4f}", log_file)

    return results, best_threshold, best_f1

def show_prediction_examples(model, dataset, device, num_examples=20, log_file=None):
    """Show individual prediction examples"""
    log_print("=== PREDICTION EXAMPLES ===", log_file)

    model.eval()
    class_names = ['Male', 'Female']

    # Get some random samples
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

            status = "CORRECT" if is_correct else "WRONG"

            log_print(f"Example {i+1:2d}: {status}", log_file)
            log_print(f"  True: {class_names[true_label]:6s} | Predicted: {class_names[predicted_label]:6s} | Confidence: {confidence:.4f}", log_file)

            # Show probabilities
            male_prob = probabilities[0, 0].item()
            female_prob = probabilities[0, 1].item()
            log_print(f"  Probabilities -> Male: {male_prob:.4f}, Female: {female_prob:.4f}", log_file)
            log_print("", log_file)

    accuracy = correct_count / num_examples
    log_print(f"Accuracy on these {num_examples} examples: {accuracy:.2%} ({correct_count}/{num_examples})", log_file)

def save_results(results, output_dir, log_file=None):
    """Save results and create visualizations"""
    log_print("=== SAVING RESULTS ===", log_file)

    os.makedirs(output_dir, exist_ok=True)

    # Save metrics summary
    metrics_path = os.path.join(output_dir, 'performance_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("TASK A - GENDER CLASSIFICATION EVALUATION RESULTS\n")
        f.write("=" * 60 + "\n")
        f.write("Objective: Predict gender (Male/Female) from face images\n")
        f.write("           captured under visually degraded conditions\n\n")

        f.write("OFFICIAL EVALUATION METRICS:\n")
        f.write(f"Accuracy:  {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)\n")
        f.write(f"Precision: {results['precision']:.4f} ({results['precision']*100:.2f}%)\n")
        f.write(f"Recall:    {results['recall']:.4f} ({results['recall']*100:.2f}%)\n")
        f.write(f"F1-Score:  {results['f1']:.4f} ({results['f1']*100:.2f}%)\n\n")

        f.write("ADDITIONAL METRICS:\n")
        f.write(f"ROC AUC: {results['roc_auc']:.4f}\n\n")

        f.write("PER-CLASS METRICS:\n")
        f.write(f"Male   - Precision: {results['precision_per_class'][0]:.4f}, Recall: {results['recall_per_class'][0]:.4f}, F1: {results['f1_per_class'][0]:.4f}\n")
        f.write(f"Female - Precision: {results['precision_per_class'][1]:.4f}, Recall: {results['recall_per_class'][1]:.4f}, F1: {results['f1_per_class'][1]:.4f}\n")

    log_print(f"Performance metrics saved to: {metrics_path}", log_file)

    # Save threshold analysis if available
    if 'threshold_results' in results:
        thresh_path = os.path.join(output_dir, 'threshold_analysis.txt')
        with open(thresh_path, 'w') as f:
            f.write("Threshold,Accuracy,Precision,Recall,F1\n")
            for r in results['threshold_results']:
                f.write(f"{r['threshold']:.2f},{r['accuracy']:.4f},{r['precision']:.4f},{r['recall']:.4f},{r['f1']:.4f}\n")
        log_print(f"Threshold analysis saved to: {thresh_path}", log_file)

    # Create visualizations
    try:
        plt.style.use('default')

        # 1. Performance metrics plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Gender Classification Model Performance', fontsize=16)

        # Confusion Matrix
        cm = results['confusion_matrix']
        im = axes[0, 0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        axes[0, 0].set_xticks([0, 1])
        axes[0, 0].set_yticks([0, 1])
        axes[0, 0].set_xticklabels(['Male', 'Female'])
        axes[0, 0].set_yticklabels(['Male', 'Female'])

        # Add text annotations
        for i in range(2):
            for j in range(2):
                axes[0, 0].text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=14)

        # ROC Curve
        axes[0, 1].plot(results['fpr'], results['tpr'], 'b-', linewidth=2, label=f'ROC (AUC = {results["roc_auc"]:.3f})')
        axes[0, 1].plot([0, 1], [0, 1], 'r--', alpha=0.7, label='Random')
        axes[0, 1].set_title('ROC Curve')
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Confidence Distribution
        axes[1, 0].hist(results['confidence'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 0].set_title('Prediction Confidence Distribution')
        axes[1, 0].set_xlabel('Confidence')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].grid(True, alpha=0.3)

        # Per-class metrics bar plot
        classes = ['Male', 'Female']
        metrics = ['Precision', 'Recall', 'F1-Score']
        x = np.arange(len(classes))
        width = 0.25

        axes[1, 1].bar(x - width, results['precision_per_class'], width, label='Precision', alpha=0.8)
        axes[1, 1].bar(x, results['recall_per_class'], width, label='Recall', alpha=0.8)
        axes[1, 1].bar(x + width, results['f1_per_class'], width, label='F1-Score', alpha=0.8)

        axes[1, 1].set_title('Per-Class Performance')
        axes[1, 1].set_xlabel('Class')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(classes)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'performance_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        log_print(f"Performance plots saved to: {plot_path}", log_file)
        plt.close()

        # 2. Threshold analysis plot (if available)
        if 'threshold_results' in results:
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))

            thresholds = [r['threshold'] for r in results['threshold_results']]

            # Metrics vs Threshold
            axes[0].plot(thresholds, [r['accuracy'] for r in results['threshold_results']], 'b-o', label='Accuracy')
            axes[0].plot(thresholds, [r['precision'] for r in results['threshold_results']], 'r-o', label='Precision')
            axes[0].plot(thresholds, [r['recall'] for r in results['threshold_results']], 'g-o', label='Recall')
            axes[0].plot(thresholds, [r['f1'] for r in results['threshold_results']], 'm-o', label='F1-Score')

            if 'best_threshold' in results:
                axes[0].axvline(results['best_threshold'], color='black', linestyle='--', alpha=0.7,
                              label=f'Best: {results["best_threshold"]:.2f}')

            axes[0].set_title('Performance vs Decision Threshold')
            axes[0].set_xlabel('Threshold')
            axes[0].set_ylabel('Score')
            axes[0].legend()
            axes[0].grid(True)

            # F1 Score focus
            axes[1].plot(thresholds, [r['f1'] for r in results['threshold_results']], 'g-o', linewidth=2)
            if 'best_threshold' in results:
                axes[1].axvline(results['best_threshold'], color='red', linestyle='--', alpha=0.7,
                              label=f'Best F1: {results.get("best_f1", 0):.4f}')
            axes[1].set_title('F1-Score vs Threshold')
            axes[1].set_xlabel('Threshold')
            axes[1].set_ylabel('F1-Score')
            axes[1].legend()
            axes[1].grid(True)

            plt.tight_layout()
            thresh_plot_path = os.path.join(output_dir, 'threshold_analysis.png')
            plt.savefig(thresh_plot_path, dpi=300, bbox_inches='tight')
            log_print(f"Threshold analysis plots saved to: {thresh_plot_path}", log_file)
            plt.close()

        log_print("All visualizations saved successfully", log_file)

    except Exception as e:
        log_print(f"Warning: Could not create visualizations: {e}", log_file)

def main():
    parser = argparse.ArgumentParser(description='Test Gender Classification Model')
    parser.add_argument('--model_path', type=str, default='checkpoints/gender_model.pt',
                        help='Path to the trained model')
    parser.add_argument('--data_dir', type=str, default='Comsys-Hackathon5/Task_A/',
                        help='Path to data directory')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--output_dir', type=str, default='test_results_gender/',
                        help='Directory to save test results')
    parser.add_argument('--num_examples', type=int, default=20,
                        help='Number of prediction examples to show')

    args = parser.parse_args()

    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, f'gender_test_log_{timestamp}.txt')

    log_print("=== GENDER CLASSIFICATION MODEL TESTING ===", log_file)
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
    model = GenderClassifier(num_classes=2, dropout_rate=0.3).to(device)

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

    # Evaluate on both training and validation sets
    all_results = {}

    for split in ['train', 'val']:
        log_print(f"\n{'='*80}", log_file)
        log_print(f"EVALUATING ON {split.upper()} SET", log_file)
        log_print(f"{'='*80}", log_file)

        # Load dataset
        log_print(f"Loading {split} dataset...", log_file)
        dataset = GenderDataset(args.data_dir, split=split, oversample_female=False)

        test_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2
        )

        log_print(f"Dataset split: {split}", log_file)
        log_print(f"Total samples: {len(dataset)}", log_file)
        class_dist = dataset.get_class_distribution()
        log_print(f"Class distribution: Male={class_dist.get(0, 0)}, Female={class_dist.get(1, 0)}", log_file)

        # Run comprehensive evaluation
        start_time = time.time()

        predictions, probabilities, labels, features = evaluate_model_comprehensive(
            model, test_loader, device, split, log_file
        )

        # Analyze performance
        perf_results = analyze_performance(predictions, probabilities, labels, split, log_file)

        # Analyze features
        feature_results = analyze_features(features, labels, log_file)

        # Threshold analysis
        thresh_results, best_thresh, best_f1 = threshold_analysis(probabilities, labels, log_file)

        # Show prediction examples
        show_prediction_examples(model, dataset, device, args.num_examples, log_file)

        test_time = time.time() - start_time

        # Store results for this split
        all_results[split] = {
            **perf_results,
            'threshold_results': thresh_results,
            'best_threshold': best_thresh,
            'best_f1': best_f1,
            'feature_analysis': feature_results,
            'test_time': test_time
        }

        log_print(f"\n{split.upper()} evaluation completed in {test_time:.1f} seconds", log_file)
        log_print(f"Best performance: Accuracy={perf_results['accuracy']:.4f}, F1={perf_results['f1']:.4f}", log_file)
        log_print(f"ROC AUC: {perf_results['roc_auc']:.4f}", log_file)
        log_print(f"Optimal threshold: {best_thresh:.2f} (F1={best_f1:.4f})", log_file)

    # Print summary comparison
    log_print(f"\n{'='*80}", log_file)
    log_print("SUMMARY COMPARISON", log_file)
    log_print(f"{'='*80}", log_file)

    log_print("TRAINING SET RESULTS:", log_file)
    train_results = all_results['train']
    log_print(f"  Training Accuracy: {train_results['accuracy']:.4f} ({train_results['accuracy']*100:.2f}%)", log_file)
    log_print(f"  Training F1 Score: {train_results['f1']:.4f} ({train_results['f1']*100:.2f}%)", log_file)
    log_print(f"  Training Precision: {train_results['precision']:.4f} ({train_results['precision']*100:.2f}%)", log_file)
    log_print(f"  Training Recall: {train_results['recall']:.4f} ({train_results['recall']*100:.2f}%)", log_file)

    log_print("\nVALIDATION SET RESULTS:", log_file)
    val_results = all_results['val']
    log_print(f"  Validation Accuracy: {val_results['accuracy']:.4f} ({val_results['accuracy']*100:.2f}%)", log_file)
    log_print(f"  Validation F1 Score: {val_results['f1']:.4f} ({val_results['f1']*100:.2f}%)", log_file)
    log_print(f"  Validation Precision: {val_results['precision']:.4f} ({val_results['precision']*100:.2f}%)", log_file)
    log_print(f"  Validation Recall: {val_results['recall']:.4f} ({val_results['recall']*100:.2f}%)", log_file)

    # Save results for both splits
    for split in ['train', 'val']:
        split_output_dir = os.path.join(args.output_dir, f'{split}_results')
        save_results(all_results[split], split_output_dir, log_file)

    log_print(f"\nComplete log saved to: {log_file}", log_file)
    log_print("Testing completed successfully!", log_file)

if __name__ == "__main__":
    main()
