import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import efficientnet_b0
import math

# DirectML import for Windows AMD GPU support
try:
    import torch_directml
    DIRECTML_AVAILABLE = torch_directml.is_available()
except ImportError:
    DIRECTML_AVAILABLE = False

class BilinearPooling(nn.Module):
    """Bilinear pooling layer for combining two feature maps"""
    def __init__(self):
        super(BilinearPooling, self).__init__()
        
    def forward(self, x1, x2):
        batch_size, height, width, depth1 = x1.size()
        _, _, _, depth2 = x2.size()
        
        # Reshape to [batch_size, height*width, depth]
        x1_flat = x1.view(batch_size, height*width, depth1)
        x2_flat = x2.view(batch_size, height*width, depth2)
        
        # Compute bilinear pooling
        phi_I = torch.bmm(x1_flat.transpose(1, 2), x2_flat)  # [batch_size, depth1, depth2]
        phi_I = phi_I.view(batch_size, depth1*depth2)
        phi_I = phi_I / (height * width)
        
        # Signed square root normalization
        y_sqrt = torch.sign(phi_I) * torch.sqrt(torch.abs(phi_I) + 1e-12)
        z_12 = F.normalize(y_sqrt, p=2, dim=1)
        
        return z_12

class EfficientNetB0FeatureExtractor(nn.Module):
    """EfficientNetB0 feature extractor with frozen/unfrozen layers"""
    def __init__(self, pretrained_weights="imagenet", freeze_layers=150, is_full=False):
        super(EfficientNetB0FeatureExtractor, self).__init__()
        
        # Load EfficientNetB0
        if pretrained_weights == "imagenet":
            self.backbone = efficientnet_b0(pretrained=True)
        elif pretrained_weights == "noisy-student":
            # For noisy-student weights, we'll use imagenet as fallback
            # In practice, you would load the actual noisy-student weights
            self.backbone = efficientnet_b0(pretrained=True)
        else:
            self.backbone = efficientnet_b0(pretrained=False)
            
        # Remove classifier
        self.features = nn.Sequential(*list(self.backbone.features.children()))
        
        # Set layer trainability based on is_full parameter
        for i, layer in enumerate(self.features):
            if i < freeze_layers:
                # First 150 layers are always frozen
                for param in layer.parameters():
                    param.requires_grad = False
            else:
                # Layers after 150: trainable for crop model, frozen for full model
                for param in layer.parameters():
                    param.requires_grad = not is_full
                    
    def forward(self, x):
        return self.features(x)

class CRBLDualModel(nn.Module):
    """CRBL Anomaly Detection Model with dual EfficientNetB0 and bilinear pooling
    Supports both crop and full image models based on is_full parameter"""
    def __init__(self, input_shape=(3, 128, 128), num_classes=1, is_full=False):
        super(CRBLDualModel, self).__init__()
        
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.is_full = is_full
        
        # Two EfficientNetB0 feature extractors
        self.feature_extractor1 = EfficientNetB0FeatureExtractor(
            pretrained_weights="imagenet", 
            freeze_layers=150,
            is_full=is_full
        )
        self.feature_extractor2 = EfficientNetB0FeatureExtractor(
            pretrained_weights="noisy-student", 
            freeze_layers=150,
            is_full=is_full
        )
        
        # Bilinear pooling
        self.bilinear_pooling = BilinearPooling()
        
        # Classifier
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(1280 * 1280, num_classes)  # EfficientNetB0 output is 1280
        
    def forward(self, x):
        # Extract features from both models
        features1 = self.feature_extractor1(x)  # [batch, 1280, 4, 4] for 128x128 input
        features2 = self.feature_extractor2(x)  # [batch, 1280, 4, 4] for 128x128 input
        
        # Transpose to match TensorFlow format [batch, height, width, channels]
        features1 = features1.permute(0, 2, 3, 1)
        features2 = features2.permute(0, 2, 3, 1)
        
        # Apply bilinear pooling
        bilinear_features = self.bilinear_pooling(features1, features2)
        
        # Classification
        x = self.dropout(bilinear_features)
        x = self.classifier(x)
        x = torch.sigmoid(x)
        
        return x

def get_device():
    """Get the best available device (DirectML > CUDA > CPU)"""
    if DIRECTML_AVAILABLE:
        device = torch_directml.device()
        print(f"Using DirectML device: {device}")
        return device
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA device: {device}")
        return device
    else:
        device = torch.device('cpu')
        print(f"Using CPU device: {device}")
        return device

def create_crop_model(input_shape=(3, 128, 128), num_classes=1):
    """Create CRBL model for cropped images (is_full=False)"""
    model = CRBLDualModel(input_shape=input_shape, num_classes=num_classes, is_full=False)
    return model

def create_full_model(input_shape=(3, 128, 128), num_classes=1):
    """Create CRBL model for full images (is_full=True)"""
    model = CRBLDualModel(input_shape=input_shape, num_classes=num_classes, is_full=True)
    return model

def create_model(input_shape=(3, 128, 128), num_classes=1, is_full=False):
    """Create CRBL model with specified type"""
    model = CRBLDualModel(input_shape=input_shape, num_classes=num_classes, is_full=is_full)
    return model

# Utility functions for model operations
def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def freeze_model_layers(model, freeze_layers=150, is_full=False):
    """Freeze specific layers in the model"""
    for i, layer in enumerate(model.feature_extractor1.features):
        if i < freeze_layers:
            for param in layer.parameters():
                param.requires_grad = False
        else:
            for param in layer.parameters():
                param.requires_grad = not is_full
                
    for i, layer in enumerate(model.feature_extractor2.features):
        if i < freeze_layers:
            for param in layer.parameters():
                param.requires_grad = False
        else:
            for param in layer.parameters():
                param.requires_grad = not is_full

def unfreeze_all_layers(model):
    """Unfreeze all layers in the model"""
    for param in model.parameters():
        param.requires_grad = True

def print_model_info(model, model_type="Unknown"):
    """Print model information including trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{model_type} Model Information:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters: {total_params - trainable_params:,}")
    print(f"  Trainable ratio: {trainable_params/total_params:.2%}")
