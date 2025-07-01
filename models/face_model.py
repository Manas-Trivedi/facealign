import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class DecorrelatedBatchNorm1d(nn.Module):
    """Decorrelated Batch Normalization to prevent dimensional collapse"""
    def __init__(self, num_features, eps=1e-3, momentum=0.1):
        super(DecorrelatedBatchNorm1d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

        # Running statistics
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_cov', torch.eye(num_features))

    def forward(self, x):
        # Check for NaN inputs
        if torch.isnan(x).any():
            print("Warning: NaN input to DecorrelatedBatchNorm1d")
            # Return normalized input as fallback
            return F.batch_norm(x.unsqueeze(0).unsqueeze(-1), None, None,
                              self.weight, self.bias, self.training, self.momentum, self.eps).squeeze(-1).squeeze(0)

        if self.training and x.size(0) > 1:  # Need at least 2 samples for covariance
            # Compute batch statistics
            batch_mean = x.mean(dim=0)
            centered = x - batch_mean

            # Add small regularization to prevent singular matrices
            batch_cov = torch.mm(centered.t(), centered) / max(x.size(0) - 1, 1)
            batch_cov = batch_cov + self.eps * torch.eye(self.num_features, device=x.device)

            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_cov = (1 - self.momentum) * self.running_cov + self.momentum * batch_cov

            # Decorrelate with more stable computation
            cov_inv_sqrt = self._matrix_power(batch_cov, -0.5)
            if cov_inv_sqrt is None:
                # Fallback to regular batch norm if matrix is too unstable
                return F.batch_norm(x.unsqueeze(0).unsqueeze(-1), None, None,
                                  self.weight, self.bias, self.training, self.momentum, self.eps).squeeze(-1).squeeze(0)

            decorrelated = torch.mm(centered, cov_inv_sqrt)
        else:
            # Use running statistics during inference or single sample
            if x.size(0) == 1:
                # Single sample case - just normalize
                centered = x - self.running_mean
                decorrelated = centered / torch.sqrt(torch.diag(self.running_cov) + self.eps)
            else:
                centered = x - self.running_mean
                cov_inv_sqrt = self._matrix_power(self.running_cov + self.eps * torch.eye(self.num_features, device=x.device), -0.5)
                if cov_inv_sqrt is None:
                    decorrelated = centered / torch.sqrt(torch.diag(self.running_cov) + self.eps)
                else:
                    decorrelated = torch.mm(centered, cov_inv_sqrt)

        # Check for NaN outputs
        if torch.isnan(decorrelated).any():
            print("Warning: NaN output from decorrelation, using fallback")
            return F.batch_norm(x.unsqueeze(0).unsqueeze(-1), None, None,
                              self.weight, self.bias, self.training, self.momentum, self.eps).squeeze(-1).squeeze(0)

        # Scale and shift
        return decorrelated * self.weight + self.bias

    def _matrix_power(self, matrix, power):
        """Compute matrix^power using eigendecomposition - MPS friendly with better stability"""
        try:
            # Move to CPU for eigendecomposition if on MPS
            device = matrix.device
            if device.type == 'mps':
                matrix_cpu = matrix.cpu()
                eigenvalues, eigenvectors = torch.linalg.eigh(matrix_cpu)
            else:
                eigenvalues, eigenvectors = torch.linalg.eigh(matrix)

            # Much more aggressive clamping for stability
            eigenvalues = torch.clamp(eigenvalues, min=1e-2)  # Increased from 1e-5

            # Check for invalid eigenvalues
            if torch.isnan(eigenvalues).any() or torch.isinf(eigenvalues).any():
                return None

            powered_eigenvalues = eigenvalues.pow(power)

            # Check for invalid powered eigenvalues
            if torch.isnan(powered_eigenvalues).any() or torch.isinf(powered_eigenvalues).any():
                return None

            result = torch.mm(torch.mm(eigenvectors, torch.diag(powered_eigenvalues)), eigenvectors.t())

            if device.type == 'mps':
                result = result.to(device)

            # Final NaN check
            if torch.isnan(result).any():
                return None

            return result
        except Exception as e:
            print(f"Matrix power computation failed: {e}")
            return None

class FaceEmbeddingModel(nn.Module):
    def __init__(self, embedding_dim=256, backbone='resnet18', pretrained=True):
        super(FaceEmbeddingModel, self).__init__()

        self.embedding_dim = embedding_dim

        # Load backbone
        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            backbone_dim = 512
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            backbone_dim = 512
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            backbone_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Remove final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        # Add embedding head with conservative design for stability
        self.embedding_head = nn.Sequential(
            nn.Linear(backbone_dim, embedding_dim * 2),
            nn.BatchNorm1d(embedding_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.BatchNorm1d(embedding_dim)  # Use regular BN for now to ensure stability
        )

    def forward(self, x):
        # Extract features from backbone
        features = self.backbone(x)
        features = features.view(features.size(0), -1)  # Flatten

        # Get embeddings
        embeddings = self.embedding_head(features)

        # L2 normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings

class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Compute distances
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)

        # Triplet loss with margin
        loss = F.relu(pos_dist - neg_dist + self.margin)

        return loss.mean()

class AlignmentUniformityLoss(nn.Module):
    def __init__(self, margin=0.3, align_alpha=2, uniform_t=2, uniform_weight=1.0):
        super(AlignmentUniformityLoss, self).__init__()
        self.margin = margin
        self.align_alpha = align_alpha
        self.uniform_t = uniform_t
        self.uniform_weight = uniform_weight

    def lalign(self, x, y, alpha=2):
        """Alignment loss - pull positive pairs together"""
        return (x - y).norm(dim=1).pow(alpha).mean()

    def lunif(self, x, t=2):
        """Uniformity loss - spread embeddings uniformly on hypersphere"""
        # MPS-compatible implementation of pairwise distances
        n = x.size(0)
        if n < 2:
            return torch.tensor(0.0, device=x.device)

        # Compute pairwise squared distances manually
        # ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2*x_i^T*x_j
        x_norm_sq = (x * x).sum(dim=1, keepdim=True)  # [n, 1]
        pairwise_sq_dist = x_norm_sq + x_norm_sq.t() - 2 * torch.mm(x, x.t())  # [n, n]

        # Get upper triangular part (excluding diagonal) to avoid duplicates
        mask = torch.triu(torch.ones(n, n, device=x.device), diagonal=1).bool()
        sq_pdist = pairwise_sq_dist[mask]

        # Clamp to avoid numerical issues and prevent extreme values
        sq_pdist = torch.clamp(sq_pdist, min=1e-8, max=10.0)  # Prevent extreme distances

        # Use more stable computation to avoid overflow/underflow
        # Instead of exp(-t * dist^2), use more stable version
        neg_t_dist = -t * sq_pdist

        # Clamp the exponential argument to prevent overflow/underflow
        neg_t_dist = torch.clamp(neg_t_dist, min=-50, max=10)  # exp(-50) ≈ 0, exp(10) is manageable

        exp_vals = torch.exp(neg_t_dist)

        # Check for numerical issues
        if torch.isnan(exp_vals).any() or torch.isinf(exp_vals).any():
            print("Warning: NaN or Inf in uniformity loss computation")
            return torch.tensor(0.0, device=x.device)

        mean_exp = exp_vals.mean()

        # Prevent log(0) or log of very small numbers
        if mean_exp < 1e-8:
            return torch.tensor(-18.0, device=x.device)  # log(1e-8) ≈ -18

        result = mean_exp.log()

        # Final NaN check
        if torch.isnan(result) or torch.isinf(result):
            print("Warning: NaN or Inf in uniformity loss result")
            return torch.tensor(0.0, device=x.device)

        return result

    def forward(self, anchor, positive, negative):
        # Check for NaN inputs
        if torch.isnan(anchor).any() or torch.isnan(positive).any() or torch.isnan(negative).any():
            print("Warning: NaN inputs to AlignmentUniformityLoss")
            return torch.tensor(0.0, device=anchor.device, requires_grad=True)

        # Original triplet loss
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        triplet_loss = F.relu(pos_dist - neg_dist + self.margin).mean()

        # Check triplet loss
        if torch.isnan(triplet_loss) or torch.isinf(triplet_loss):
            print("Warning: Invalid triplet loss")
            triplet_loss = torch.tensor(0.0, device=anchor.device, requires_grad=True)

        # Alignment loss - pull positive pairs together
        align_loss = self.lalign(anchor, positive, self.align_alpha)

        # Check alignment loss
        if torch.isnan(align_loss) or torch.isinf(align_loss):
            print("Warning: Invalid alignment loss")
            align_loss = torch.tensor(0.0, device=anchor.device, requires_grad=True)

        # Uniformity loss - spread all embeddings uniformly
        all_embeddings = torch.cat([anchor, positive, negative], dim=0)
        uniform_loss = self.lunif(all_embeddings, self.uniform_t)

        # Check uniformity loss
        if torch.isnan(uniform_loss) or torch.isinf(uniform_loss):
            print("Warning: Invalid uniformity loss")
            uniform_loss = torch.tensor(0.0, device=anchor.device, requires_grad=True)

        # Combined loss
        total_loss = triplet_loss + align_loss + self.uniform_weight * uniform_loss

        # Final check
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"Warning: Invalid total loss - triplet: {triplet_loss}, align: {align_loss}, uniform: {uniform_loss}")
            return torch.tensor(0.0, device=anchor.device, requires_grad=True)

        return total_loss

class ContrastiveLoss(nn.Module):
    """Contrastive Loss for better positive/negative separation"""
    def __init__(self, margin=1.0, temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.temperature = temperature

    def forward(self, anchor, positive, negative):
        # Normalize embeddings
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        negative = F.normalize(negative, p=2, dim=1)

        # Positive pairs - minimize distance
        pos_sim = F.cosine_similarity(anchor, positive, dim=1)
        pos_loss = (1 - pos_sim).mean()

        # Negative pairs - maximize distance (with margin)
        neg_sim = F.cosine_similarity(anchor, negative, dim=1)
        neg_loss = F.relu(neg_sim - self.margin).mean()

        return pos_loss + neg_loss

def cosine_similarity(embedding1, embedding2):
    """Compute cosine similarity between two embeddings"""
    return F.cosine_similarity(embedding1, embedding2, dim=1)

def euclidean_distance(embedding1, embedding2):
    """Compute euclidean distance between two embeddings"""
    return F.pairwise_distance(embedding1, embedding2, p=2)

# Helper function to create model
def create_face_model(embedding_dim=256, backbone='resnet18', pretrained=True):
    """Create and return face embedding model"""
    model = FaceEmbeddingModel(
        embedding_dim=embedding_dim,
        backbone=backbone,
        pretrained=pretrained
    )
    return model