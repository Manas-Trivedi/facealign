import torch
import torch.nn as nn
import torchvision.models as models


class GenderClassifier(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.3):
        """
        Gender Classification model based on ResNet34

        Args:
            num_classes (int): Number of output classes (default: 2 for male/female)
            dropout_rate (float): Dropout rate before final classifier
        """
        super(GenderClassifier, self).__init__()

        # Load pretrained ResNet34
        self.backbone = models.resnet34(pretrained=True)

        # Get the number of features from the last layer
        num_features = self.backbone.fc.in_features

        # Remove the original classifier
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        # Add our custom classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, num_classes)
        )

        # Initialize the classifier weights
        self._initialize_classifier()

    def _initialize_classifier(self):
        """Initialize the classifier layer with proper weights"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 224, 224)

        Returns:
            torch.Tensor: Raw logits of shape (batch_size, num_classes)
        """
        # Extract features using backbone
        features = self.backbone(x)

        # Classify
        logits = self.classifier(features)

        return logits

    def get_features(self, x):
        """
        Extract feature embeddings (useful for analysis)

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 224, 224)

        Returns:
            torch.Tensor: Feature embeddings before classification
        """
        with torch.no_grad():
            features = self.backbone(x)
            features = features.view(features.size(0), -1)  # Flatten
        return features


def create_model(num_classes=2, dropout_rate=0.3, device='cpu'):
    """
    Create and initialize the gender classification model

    Args:
        num_classes (int): Number of output classes
        dropout_rate (float): Dropout rate
        device (str): Device to move model to

    Returns:
        GenderClassifier: Initialized model
    """
    model = GenderClassifier(num_classes=num_classes, dropout_rate=dropout_rate)
    model = model.to(device)

    return model


def count_parameters(model):
    """
    Count total and trainable parameters in the model

    Args:
        model (nn.Module): PyTorch model

    Returns:
        tuple: (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total_params, trainable_params


if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model
    model = create_model(device=(str)(device))

    # Print model info
    total_params, trainable_params = count_parameters(model)
    print(f"\nModel created successfully!")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test forward pass
    batch_size = 4
    test_input = torch.randn(batch_size, 3, 224, 224).to(device)

    print(f"\nTesting forward pass...")
    print(f"Input shape: {test_input.shape}")

    with torch.no_grad():
        output = model(test_input)
        print(f"Output shape: {output.shape}")
        print(f"Output (raw logits): {output}")

        # Test probabilities
        probabilities = torch.softmax(output, dim=1)
        print(f"Probabilities: {probabilities}")

        # Test predictions
        predictions = torch.argmax(output, dim=1)
        print(f"Predictions: {predictions}")

        # Test feature extraction
        features = model.get_features(test_input)
        print(f"Feature embeddings shape: {features.shape}")

    print(f"\nModel test completed successfully!")

    # Print model architecture summary
    print(f"\nModel Architecture:")
    print(f"Backbone: ResNet34 (pretrained)")
    print(f"Features: 512 (ResNet34)")
    print(f"Classifier: Dropout({model.classifier[1].p}) -> Linear(512, 2)")
    print(f"Output: Raw logits (no activation)")