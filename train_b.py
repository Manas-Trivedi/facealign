import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import argparse
from tqdm import tqdm
import time

from models.face_model import FaceEmbeddingModel, TripletLoss, AlignmentUniformityLoss, cosine_similarity
from utils.face_dataset import TripletDataset, ValidationDataset, get_transforms

def compute_embeddings(model, images, device, batch_size=32):
    """Compute embeddings for a list of images"""
    model.eval()
    embeddings = []

    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            batch_tensor = torch.stack(batch).to(device)
            batch_embeddings = model(batch_tensor)
            embeddings.append(batch_embeddings.cpu())

    return torch.cat(embeddings, dim=0)

def validate_model(model, val_dataset, device, threshold=0.5, num_queries=200, debug=False):
    """Validate model performance"""
    model.eval()

    # Get gallery embeddings
    gallery_imgs, gallery_identities = val_dataset.get_gallery()
    gallery_embeddings = compute_embeddings(model, gallery_imgs, device)

    # Create validation queries
    query_imgs, query_labels, query_info = val_dataset.create_validation_batch(num_queries)
    query_embeddings = compute_embeddings(model, query_imgs, device)

    predictions = []
    similarities_list = []

    # For each query, find closest gallery match
    for i, query_emb in enumerate(query_embeddings):
        # Compute similarities with all gallery images
        similarities = cosine_similarity(
            query_emb.unsqueeze(0).repeat(len(gallery_embeddings), 1),
            gallery_embeddings
        )

        # Best match
        max_similarity = similarities.max().item()
        best_match_idx = similarities.argmax().item()
        best_match_identity = gallery_identities[best_match_idx]

        # Threshold decision
        pred = 1 if max_similarity > threshold else 0
        predictions.append(pred)
        similarities_list.append(max_similarity)

        # Debug information for first few queries or when debug=True
        if debug and i < 20:  # Show more queries
            info = query_info[i]

            # Check if query identity exists in gallery
            query_identity = info['identity']
            identity_in_gallery = query_identity in gallery_identities

            print(f"Query {i+1:2d}: {info['split']:5s} | {info['type']:9s} | GT:{info['ground_truth']} | Pred:{pred} | Sim:{max_similarity:.4f}")
            print(f"    Query ID: {query_identity[:20]:20s} | In Gallery: {identity_in_gallery} | Best Match: {best_match_identity[:20]:20s}")
            if not identity_in_gallery and info['split'] == 'val':
                print(f"    ⚠️  ERROR: Val query identity '{query_identity}' NOT FOUND in gallery!")

    # Calculate metrics
    accuracy = accuracy_score(query_labels, predictions)
    f1 = f1_score(query_labels, predictions, average='binary')

    if debug:
        # Print summary statistics
        print(f"\n=== VALIDATION SUMMARY ===")
        print(f"Total queries: {len(query_labels)}")
        print(f"Positive queries (val): {sum(1 for info in query_info if info['ground_truth'] == 1)}")
        print(f"Negative queries (train): {sum(1 for info in query_info if info['ground_truth'] == 0)}")
        print(f"Correct predictions: {sum(1 for i in range(len(predictions)) if predictions[i] == query_labels[i])}")
        print(f"Average similarity for positives: {np.mean([similarities_list[i] for i in range(len(similarities_list)) if query_labels[i] == 1]):.4f}")
        print(f"Average similarity for negatives: {np.mean([similarities_list[i] for i in range(len(similarities_list)) if query_labels[i] == 0]):.4f}")
        print(f"Threshold: {threshold}")
        print(f"Similarity range: {min(similarities_list):.4f} - {max(similarities_list):.4f}")

        # Count predictions by type
        val_correct = sum(1 for i in range(len(predictions)) if query_info[i]['split'] == 'val' and predictions[i] == 1)
        val_total = sum(1 for info in query_info if info['split'] == 'val')
        train_correct = sum(1 for i in range(len(predictions)) if query_info[i]['split'] == 'train' and predictions[i] == 0)
        train_total = sum(1 for info in query_info if info['split'] == 'train')

        print(f"Val accuracy: {val_correct}/{val_total} = {val_correct/val_total:.4f}")
        print(f"Train accuracy: {train_correct}/{train_total} = {train_correct/train_total:.4f}")

    return accuracy, f1, predictions, query_labels

def train_epoch(model, dataloader, criterion, optimizer, device, debug_training=False):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    pos_distances = []
    neg_distances = []

    progress_bar = tqdm(dataloader, desc="Training")

    for batch_idx, (anchor, positive, negative) in enumerate(progress_bar):
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        optimizer.zero_grad()

        # Get embeddings
        anchor_emb = model(anchor)
        positive_emb = model(positive)
        negative_emb = model(negative)

        # Compute loss
        loss = criterion(anchor_emb, positive_emb, negative_emb)

        # Debug: track distances
        if debug_training and batch_idx == 0:
            with torch.no_grad():
                pos_dist = torch.nn.functional.pairwise_distance(anchor_emb, positive_emb, p=2)
                neg_dist = torch.nn.functional.pairwise_distance(anchor_emb, negative_emb, p=2)
                pos_distances.extend(pos_dist.cpu().tolist())
                neg_distances.extend(neg_dist.cpu().tolist())

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Avg Loss': f'{total_loss/num_batches:.4f}'
        })

    if debug_training and pos_distances:
        print(f"  Avg Positive Distance: {np.mean(pos_distances):.4f}")
        print(f"  Avg Negative Distance: {np.mean(neg_distances):.4f}")
        print(f"  Distance Margin: {np.mean(neg_distances) - np.mean(pos_distances):.4f}")

    return total_loss / num_batches

def main():
    parser = argparse.ArgumentParser(description='Train Face Recognition Model')
    parser.add_argument('--data_dir', type=str, default='data/facecom/Task_B/',
                        help='Path to data directory')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--embedding_dim', type=int, default=256,
                        help='Embedding dimension')
    parser.add_argument('--margin', type=float, default=2.0,
                        help='Triplet loss margin')
    parser.add_argument('--backbone', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50'],
                        help='Backbone architecture')
    parser.add_argument('--threshold', type=float, default=0.9995,
                        help='Similarity threshold for validation')
    parser.add_argument('--save_dir', type=str, default='models/',
                        help='Directory to save models')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--val_queries', type=int, default=200,
                        help='Number of validation queries per epoch')

    args = parser.parse_args()

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Datasets
    print("Loading datasets...")
    train_transform = get_transforms(train=True)
    val_transform = get_transforms(train=False)

    train_dataset = TripletDataset(args.data_dir, 'train', train_transform)
    val_dataset = ValidationDataset(args.data_dir, transform=val_transform)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Gallery size: {len(val_dataset.gallery_images)}")
    print(f"Positive queries available: {len(val_dataset.positive_queries)} (distorted: {len(val_dataset.positive_queries_distorted)}, clear: {len(val_dataset.positive_queries_clear)})")
    print(f"Negative queries available: {len(val_dataset.negative_queries)}")

    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    # Model
    print("Creating model...")
    model = FaceEmbeddingModel(
        embedding_dim=args.embedding_dim,
        backbone=args.backbone,
        pretrained=True
    ).to(device)

    # Loss and optimizer
    criterion = AlignmentUniformityLoss(margin=args.margin, uniform_weight=0.5)  # Use new loss!
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)  # Reduced weight decay
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)  # More gradual decay

    # Training loop
    print("Starting training...")
    best_f1 = 0.0
    train_losses = []
    val_accuracies = []
    val_f1s = []

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        # Train
        start_time = time.time()
        debug_training = (epoch == 0)  # Debug first epoch
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, debug_training)
        train_time = time.time() - start_time

        # Validate
        print("Validating...")
        start_time = time.time()

        # Enable debug for first epoch to see what's happening
        debug_mode = (epoch == 0)

        val_acc, val_f1, _, _ = validate_model(
            model, val_dataset, device,
            threshold=args.threshold,
            num_queries=args.val_queries,
            debug=debug_mode
        )
        val_time = time.time() - start_time

        # Learning rate decay
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Log results
        train_losses.append(train_loss)
        val_accuracies.append(val_acc)
        val_f1s.append(val_f1)

        print(f"Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
        print(f"LR: {current_lr:.6f} | Train Time: {train_time:.1f}s | Val Time: {val_time:.1f}s")

        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pt'))
            print(f"New best F1: {best_f1:.4f} - Model saved!")

    # Final evaluation with different thresholds
    print("\nFinal evaluation with threshold tuning...")
    thresholds = [0.98, 0.985, 0.99, 0.992, 0.994, 0.995, 0.996, 0.997, 0.998, 0.999]
    best_threshold = args.threshold
    best_final_f1 = 0.0

    for thresh in thresholds:
        val_acc, val_f1, _, _ = validate_model(
            model, val_dataset, device,
            threshold=thresh,
            num_queries=args.val_queries * 2,  # More queries for final eval
            debug=False
        )
        print(f"Threshold {thresh}: Acc={val_acc:.4f}, F1={val_f1:.4f}")

        if val_f1 > best_final_f1:
            best_final_f1 = val_f1
            best_threshold = thresh

    print(f"\nBest threshold: {best_threshold} with F1: {best_final_f1:.4f}")

    # Save final model with best threshold
    torch.save(model.state_dict(), os.path.join(args.save_dir, 'final_model.pt'))

    print("Training completed!")

if __name__ == "__main__":
    main()