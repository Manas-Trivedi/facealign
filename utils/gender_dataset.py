import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from collections import defaultdict


class GenderDataset(Dataset):
    def __init__(self, data_dir, split='train', oversample_female=True, target_female_count=850):
        """
        Gender Classification Dataset with class balancing

        Args:
            data_dir (str): Path to data directory (e.g., 'Comsys-Hackathon5/Task_A')
            split (str): 'train' or 'val'
            oversample_female (bool): Whether to oversample female images
            target_female_count (int): Target number of female samples after oversampling
        """
        self.data_dir = data_dir
        self.split = split
        self.oversample_female = oversample_female and split == 'train'  # Only oversample for training

        # Define transforms
        self.base_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])

        # Light augmentation transforms for oversampling
        self.aug_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        # Load dataset
        self.samples = self._load_samples()

        print(f"{split.upper()} Dataset loaded:")
        print(f"  Male: {sum(1 for _, label, _ in self.samples if label == 0)}")
        print(f"  Female: {sum(1 for _, label, _ in self.samples if label == 1)}")
        print(f"  Total: {len(self.samples)}")

    def _load_samples(self):
        """Load all samples and handle class balancing"""
        samples = []

        # Class mapping: male=0, female=1 (female as positive class)
        class_to_idx = {'male': 0, 'female': 1}

        split_dir = os.path.join(self.data_dir, self.split)

        # Collect original samples
        original_samples = {'male': [], 'female': []}

        for class_name in ['male', 'female']:
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.exists(class_dir):
                raise ValueError(f"Directory not found: {class_dir}")

            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    original_samples[class_name].append(img_path)

        # Add all male samples (no augmentation needed)
        for img_path in original_samples['male']:
            samples.append((img_path, class_to_idx['male'], False))  # False = no augmentation

        # Handle female samples with oversampling
        female_paths = original_samples['female']

        if self.oversample_female and len(female_paths) > 0:
            # Add original female samples
            for img_path in female_paths:
                samples.append((img_path, class_to_idx['female'], False))

            # Calculate how many augmented samples we need
            original_female_count = len(female_paths)
            target_count = min(850, len(original_samples['male']))  # Don't exceed male count
            additional_needed = max(0, target_count - original_female_count)

            # Create additional augmented samples
            if additional_needed > 0:
                # Cycle through female images to create augmented versions
                for i in range(additional_needed):
                    img_path = female_paths[i % len(female_paths)]
                    samples.append((img_path, class_to_idx['female'], True))  # True = use augmentation
        else:
            # No oversampling - just add original female samples
            for img_path in female_paths:
                samples.append((img_path, class_to_idx['female'], False))

        # Shuffle samples for training
        if self.split == 'train':
            random.shuffle(samples)

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, use_augmentation = self.samples[idx]

        try:
            # Load image
            image = Image.open(img_path).convert('RGB')

            # Apply appropriate transform
            if use_augmentation:
                image = self.aug_transform(image)
            else:
                image = self.base_transform(image)

            return image, label

        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            fallback_image = torch.zeros(3, 224, 224)
            return fallback_image, label

    def get_class_distribution(self):
        """Get class distribution for analysis"""
        distribution = defaultdict(int)
        for _, label, _ in self.samples:
            distribution[label] += 1
        return dict(distribution)

    def get_class_weights(self):
        """Calculate class weights for weighted loss"""
        distribution = self.get_class_distribution()
        total_samples = sum(distribution.values())

        # Calculate weights (inverse frequency)
        weights = {}
        for class_idx, count in distribution.items():
            weights[class_idx] = total_samples / (len(distribution) * count)

        # Return as tensor in class order [male_weight, female_weight]
        return torch.tensor([weights[0], weights[1]], dtype=torch.float32)


def get_dataloaders(data_dir, batch_size=32, num_workers=2):
    """
    Create train and validation dataloaders

    Args:
        data_dir (str): Path to data directory
        batch_size (int): Batch size
        num_workers (int): Number of worker processes

    Returns:
        train_loader, val_loader, class_weights
    """
    # Create datasets
    train_dataset = GenderDataset(data_dir, split='train', oversample_female=True)
    val_dataset = GenderDataset(data_dir, split='val', oversample_female=False)

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # Get class weights for loss function
    class_weights = train_dataset.get_class_weights()

    print(f"\nDataloaders created:")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Class weights: {class_weights}")

    return train_loader, val_loader, class_weights


if __name__ == "__main__":
    # Test the dataset
    data_dir = "Comsys-Hackathon5/Task_A"

    # Test dataset creation
    train_dataset = GenderDataset(data_dir, split='train')
    val_dataset = GenderDataset(data_dir, split='val')

    print(f"\nClass distributions:")
    print(f"Train: {train_dataset.get_class_distribution()}")
    print(f"Val: {val_dataset.get_class_distribution()}")

    # Test dataloader
    train_loader, val_loader, weights = get_dataloaders(data_dir, batch_size=16)

    # Test a batch
    for images, labels in train_loader:
        print(f"\nFirst batch:")
        print(f"Images shape: {images.shape}")
        print(f"Labels: {labels}")
        print(f"Male count: {(labels == 0).sum().item()}")
        print(f"Female count: {(labels == 1).sum().item()}")
        break