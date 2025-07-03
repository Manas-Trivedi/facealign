import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import time
from datetime import datetime

# Import our modules
from utils.gender_dataset import get_dataloaders
from models.gender_model import create_model, count_parameters


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    print(f"\nEpoch {epoch+1} - Training:")
    print("-" * 50)

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Print progress every 20 batches
        if (batch_idx + 1) % 20 == 0:
            print(f"Batch [{batch_idx+1}/{len(train_loader)}] - Loss: {loss.item():.4f}")

    # Calculate epoch metrics
    avg_loss = total_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

    # Calculate per-class metrics (male=0, female=1)
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=[0, 1]
    )

    print(f"\nTrain Results:")
    print(f"Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    print(f"Female (Class 1) - Precision: {precision_per_class[1]:.4f}, Recall: {recall_per_class[1]:.4f}, F1: {f1_per_class[1]:.4f}") # type: ignore

    return avg_loss, accuracy, f1


def validate_epoch(model, val_loader, criterion, device, epoch):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    print(f"\nEpoch {epoch+1} - Validation:")
    print("-" * 50)

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Statistics
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate epoch metrics
    avg_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

    # Calculate per-class metrics
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=[0, 1]
    )

    print(f"\nValidation Results:")
    print(f"Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    print(f"Female (Class 1) - Precision: {precision_per_class[1]:.4f}, Recall: {recall_per_class[1]:.4f}, F1: {f1_per_class[1]:.4f}") # type: ignore

    # Detailed classification report
    print(f"\nDetailed Classification Report:")
    target_names = ['Male', 'Female']
    print(classification_report(all_labels, all_preds, target_names=target_names))

    return avg_loss, accuracy, f1


def train_model():
    """Main training function"""
    print("=" * 60)
    print("GENDER CLASSIFICATION TRAINING")
    print("=" * 60)

    # Configuration
    DATA_DIR = "Comsys-Hackathon5/Task_A"
    BATCH_SIZE = 16
    NUM_WORKERS = 0
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 25
    PATIENCE = 7
    CHECKPOINT_DIR = "checkpoints"

    # Create checkpoint directory
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Device setup
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print(f"\nLoading data from: {DATA_DIR}")
    train_loader, val_loader, class_weights = get_dataloaders(
        DATA_DIR, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )

    # Create model
    print(f"\nCreating model...")
    model = create_model(device=(str)(device))
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss and optimizer
    print(f"\nSetting up training...")
    print(f"Class weights: {class_weights}")
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    # Training setup
    best_accuracy = 0.0
    patience_counter = 0
    start_time = time.time()

    print(f"\nTraining Configuration:")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Early Stopping Patience: {PATIENCE}")
    print(f"Scheduler: ReduceLROnPlateau (patience=3)")

    print(f"\nStarting training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Training loop
    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()

        # Train
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss, val_acc, val_f1 = validate_epoch(
            model, val_loader, criterion, device, epoch
        )

        # Scheduler step
        scheduler.step(val_acc)

        # Epoch summary
        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']

        print(f"\n" + "="*60)
        print(f"EPOCH {epoch+1}/{NUM_EPOCHS} SUMMARY")
        print(f"Time: {epoch_time:.2f}s | LR: {current_lr:.6f}")
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

        # Save best model (based on accuracy)
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            patience_counter = 0

            checkpoint_path = os.path.join(CHECKPOINT_DIR, "gender_model.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✅ New best accuracy: {best_accuracy:.4f} - Model saved!")
        else:
            patience_counter += 1
            print(f"❌ No improvement. Patience: {patience_counter}/{PATIENCE}")

        print("="*60)

        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"\n🛑 Early stopping triggered after {epoch+1} epochs!")
            print(f"Best validation accuracy: {best_accuracy:.4f}")
            break

    # Training completed
    total_time = time.time() - start_time
    print(f"\n" + "="*60)
    print("TRAINING COMPLETED!")
    print(f"Total time: {total_time/60:.2f} minutes")
    print(f"Best validation accuracy: {best_accuracy:.4f}")
    print(f"Best model saved at: {os.path.join(CHECKPOINT_DIR, 'gender_model.pt')}")
    print("="*60)


if __name__ == "__main__":
    train_model()