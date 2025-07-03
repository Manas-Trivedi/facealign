import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class FaceDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        """
        Args:
            data_dir: Path to data directory (e.g., 'Comsys-Hackathon5/Task_B/')
            split: 'train' or 'val'
            transform: Image transformations
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform

        # Load all images and their identity labels
        self.images, self.identities = self._load_images()
        self.identity_to_images = self._group_by_identity()
        self.unique_identities = list(self.identity_to_images.keys())

    def _load_images(self):
        """Load all images (clear + distorted) and map to identities"""
        images = []
        identities = []

        split_dir = os.path.join(self.data_dir, self.split)

        for identity_folder in os.listdir(split_dir):
            identity_path = os.path.join(split_dir, identity_folder)
            if not os.path.isdir(identity_path):
                continue

            # Load clear images
            for img_file in os.listdir(identity_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(identity_path, img_file)
                    images.append(img_path)
                    identities.append(identity_folder)

            # Load distorted images
            distortion_dir = os.path.join(identity_path, 'distortion')
            if os.path.exists(distortion_dir):
                for img_file in os.listdir(distortion_dir):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(distortion_dir, img_file)
                        images.append(img_path)
                        identities.append(identity_folder)

        return images, identities

    def _group_by_identity(self):
        """Group image paths by identity"""
        identity_to_images = {}
        for img_path, identity in zip(self.images, self.identities):
            if identity not in identity_to_images:
                identity_to_images[identity] = []
            identity_to_images[identity].append(img_path)
        return identity_to_images

    def _load_image(self, img_path):
        """Load and transform image"""
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image as fallback
            blank = Image.new('RGB', (224, 224), color=(128, 128, 128))
            if self.transform:
                blank = self.transform(blank)
            return blank

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """Return single image and identity for basic usage"""
        img_path = self.images[idx]
        identity = self.identities[idx]
        image = self._load_image(img_path)
        return image, identity

class TripletDataset(FaceDataset):
    def __init__(self, data_dir, split='train', transform=None, hard_negative_prob=0.5, model=None):
        super().__init__(data_dir, split, transform)
        self.hard_negative_prob = hard_negative_prob  # Probability of selecting hard negatives
        self.model = model  # Model for computing embeddings (optional)
        self.negative_cache = {}  # Cache hard negatives

    def set_model(self, model):
        """Set the model for hard negative mining"""
        self.model = model

    def _get_hard_negative(self, anchor_img, anchor_identity):
        """Get a hard negative using the current model"""
        if self.model is None:
            # Fallback to random negative
            return self._get_random_negative(anchor_identity)

        self.model.eval()
        with torch.no_grad():
            # Get anchor embedding
            if isinstance(anchor_img, torch.Tensor):
                anchor_tensor = anchor_img.unsqueeze(0)
            else:
                # Convert PIL image to tensor if needed
                from torchvision import transforms
                to_tensor = transforms.ToTensor()
                anchor_tensor = to_tensor(anchor_img).unsqueeze(0)

            if torch.cuda.is_available():
                anchor_tensor = anchor_tensor.cuda()
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                anchor_tensor = anchor_tensor.to('mps')

            anchor_emb = self.model(anchor_tensor).cpu()

            # Sample candidates from different identities
            candidate_identities = [id for id in self.unique_identities if id != anchor_identity]
            candidates = random.sample(candidate_identities, min(20, len(candidate_identities)))

            similarities = []
            candidate_imgs = []

            for neg_identity in candidates:
                neg_path = random.choice(self.identity_to_images[neg_identity])
                neg_img = self._load_image(neg_path)
                candidate_imgs.append((neg_img, neg_path))

                # Get negative embedding
                if isinstance(neg_img, torch.Tensor):
                    neg_tensor = neg_img.unsqueeze(0)
                else:
                    # Convert PIL image to tensor if needed
                    from torchvision import transforms
                    to_tensor = transforms.ToTensor()
                    neg_tensor = to_tensor(neg_img).unsqueeze(0)

                if torch.cuda.is_available():
                    neg_tensor = neg_tensor.cuda()
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    neg_tensor = neg_tensor.to('mps')

                neg_emb = self.model(neg_tensor).cpu()

                # Compute similarity (higher = harder negative)
                sim = torch.nn.functional.cosine_similarity(anchor_emb, neg_emb, dim=1).item()
                similarities.append(sim)

            # Select the hardest negative (highest similarity)
            hardest_idx = max(range(len(similarities)), key=lambda i: similarities[i])
            return candidate_imgs[hardest_idx][0]

    def _get_random_negative(self, anchor_identity):
        """Get a random negative (fallback method)"""
        negative_identity = random.choice([id for id in self.unique_identities
                                         if id != anchor_identity])
        negative_path = random.choice(self.identity_to_images[negative_identity])
        return self._load_image(negative_path)

    def __getitem__(self, idx):
        """Generate a triplet: anchor, positive, negative with hard negative mining"""
        # Select anchor
        anchor_path = self.images[idx]
        anchor_identity = self.identities[idx]
        anchor_img = self._load_image(anchor_path)

        # Select positive (same identity, different image)
        positive_candidates = [img for img in self.identity_to_images[anchor_identity]
                             if img != anchor_path]
        if len(positive_candidates) == 0:
            # If only one image for this identity, use the same image
            positive_path = anchor_path
        else:
            positive_path = random.choice(positive_candidates)
        positive_img = self._load_image(positive_path)

        # Select negative with hard mining
        if random.random() < self.hard_negative_prob:
            negative_img = self._get_hard_negative(anchor_img, anchor_identity)
        else:
            negative_img = self._get_random_negative(anchor_identity)

        return anchor_img, positive_img, negative_img

class ValidationDataset:
    def __init__(self, data_dir, val_split='val', train_split='train', transform=None):
        """
        Dataset for validation queries
        Args:
            data_dir: Path to data directory
            val_split: Split to use for gallery (clear images) and positive queries (distorted + clear)
            train_split: Split to use for negative queries
            transform: Image transformations
        """
        self.data_dir = data_dir
        self.transform = transform

        # Load gallery images (clear images from val)
        self.gallery_images, self.gallery_identities = self._load_gallery(val_split)

        # Load positive queries: distorted images from val + some clear images from val
        self.positive_queries_distorted = self._load_distorted_images(val_split)
        self.positive_queries_clear = self._load_clear_images_for_queries(val_split)

        # Combine all positive queries
        self.positive_queries = self.positive_queries_distorted + self.positive_queries_clear

        # Load negative queries (clear + distorted images from train)
        self.negative_queries = self._load_negative_queries(train_split)

    def _load_gallery(self, split):
        """Load clear images for gallery"""
        gallery_images = []
        gallery_identities = []

        split_dir = os.path.join(self.data_dir, split)

        for identity_folder in os.listdir(split_dir):
            identity_path = os.path.join(split_dir, identity_folder)
            if not os.path.isdir(identity_path):
                continue

            # Only load clear images (not from distortion folder)
            for img_file in os.listdir(identity_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(identity_path, img_file)
                    gallery_images.append(img_path)
                    gallery_identities.append(identity_folder)

        return gallery_images, gallery_identities

    def _load_distorted_images(self, split):
        """Load distorted images for positive queries"""
        distorted_images = []

        split_dir = os.path.join(self.data_dir, split)

        for identity_folder in os.listdir(split_dir):
            identity_path = os.path.join(split_dir, identity_folder)
            distortion_dir = os.path.join(identity_path, 'distortion')

            if os.path.exists(distortion_dir):
                for img_file in os.listdir(distortion_dir):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(distortion_dir, img_file)
                        # Extract identity from path
                        distorted_images.append((img_path, identity_folder))

        return distorted_images

    def _load_clear_images_for_queries(self, split):
        """Load clear images from val for positive queries (sanity check)"""
        clear_images = []

        split_dir = os.path.join(self.data_dir, split)

        for identity_folder in os.listdir(split_dir):
            identity_path = os.path.join(split_dir, identity_folder)
            if not os.path.isdir(identity_path):
                continue

            # Load clear images (not from distortion folder)
            for img_file in os.listdir(identity_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(identity_path, img_file)
                    clear_images.append((img_path, identity_folder))

        return clear_images

    def _load_negative_queries(self, split):
        """Load images from different split for negative queries"""
        negative_images = []

        split_dir = os.path.join(self.data_dir, split)

        for identity_folder in os.listdir(split_dir):
            identity_path = os.path.join(split_dir, identity_folder)
            if not os.path.isdir(identity_path):
                continue

            # Load clear images
            for img_file in os.listdir(identity_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(identity_path, img_file)
                    negative_images.append(img_path)

            # Load distorted images
            distortion_dir = os.path.join(identity_path, 'distortion')
            if os.path.exists(distortion_dir):
                for img_file in os.listdir(distortion_dir):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(distortion_dir, img_file)
                        negative_images.append(img_path)

        return negative_images

    def _load_image(self, img_path):
        """Load and transform image"""
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            blank = Image.new('RGB', (224, 224), color=(128, 128, 128))
            if self.transform:
                blank = self.transform(blank)
            return blank

    def create_validation_batch(self, num_queries=100):
        """Create a balanced validation batch"""
        queries = []
        labels = []
        query_info = []  # Store debugging info

        # Sample positive queries (50%) - mix of distorted and clear from val
        num_positive = num_queries // 2

        # Ensure we have enough positive queries
        if len(self.positive_queries) < num_positive:
            print(f"Warning: Only {len(self.positive_queries)} positive queries available, requested {num_positive}")
            num_positive = len(self.positive_queries)

        positive_samples = random.sample(self.positive_queries, num_positive)

        for img_path, identity in positive_samples:
            queries.append(self._load_image(img_path))
            labels.append(1)  # Positive match

            # Determine if it's distorted or clear
            query_type = "distorted" if "/distortion/" in img_path else "clear"
            query_info.append({
                'path': img_path,
                'identity': identity,
                'split': 'val',
                'type': query_type,
                'ground_truth': 1
            })

        # Sample negative queries (50%) - from train (different identities)
        num_negative = num_queries - len(positive_samples)

        if len(self.negative_queries) < num_negative:
            print(f"Warning: Only {len(self.negative_queries)} negative queries available, requested {num_negative}")
            num_negative = len(self.negative_queries)

        negative_samples = random.sample(self.negative_queries, num_negative)

        for img_path in negative_samples:
            queries.append(self._load_image(img_path))
            labels.append(0)  # Negative match

            # Extract identity from path
            identity = img_path.split('/')[-2] if '/distortion/' in img_path else img_path.split('/')[-1].split('.')[0]
            query_type = "distorted" if "/distortion/" in img_path else "clear"
            query_info.append({
                'path': img_path,
                'identity': identity,
                'split': 'train',
                'type': query_type,
                'ground_truth': 0
            })

        return queries, labels, query_info

    def get_gallery(self):
        """Return all gallery images and identities"""
        gallery_imgs = [self._load_image(img_path) for img_path in self.gallery_images]
        return gallery_imgs, self.gallery_identities

def get_transforms(train=True, img_size=224):
    """Get image transformations"""
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])