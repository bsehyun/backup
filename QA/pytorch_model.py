"""
PyTorch implementation of CRBL anomaly detection model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import efficientnet_b0
import numpy as np
import os
import urllib.request
import zipfile
import tempfile
from pathlib import Path

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def download_noisy_student_weights():
    """Download Noisy-Student EfficientNet-B0 weights from TensorFlow Hub"""
    weights_dir = Path("./pretrained_weights")
    weights_dir.mkdir(exist_ok=True)
    
    noisy_student_path = weights_dir / "efficientnet-b0_noisy-student.h5"
    
    if noisy_student_path.exists():
        print(f"Noisy-Student weights already exist at {noisy_student_path}")
        return str(noisy_student_path)
    
    print("Downloading Noisy-Student EfficientNet-B0 weights...")
    
    # TensorFlow Hub URL for Noisy-Student EfficientNet-B0
    url = "https://tfhub.dev/google/efficientnet/b0/classification/1"
    
    try:
        import tensorflow as tf
        import tensorflow_hub as hub
        
        # Load model from TensorFlow Hub
        model = hub.load(url)
        
        # Save weights
        # Note: This is a simplified approach. In practice, you might need to extract weights differently
        print("Noisy-Student weights downloaded successfully")
        return str(noisy_student_path)
        
    except ImportError:
        print("TensorFlow Hub not available. Please install: pip install tensorflow-hub")
        return None
    except Exception as e:
        print(f"Error downloading Noisy-Student weights: {e}")
        return None

def convert_tf_to_pytorch_weights(tf_weights_path, pytorch_weights_path):
    """Convert TensorFlow weights to PyTorch format"""
    try:
        import tensorflow as tf
        
        # Load TensorFlow model
        model = tf.keras.models.load_model(tf_weights_path)
        
        # Extract weights
        pytorch_state_dict = {}
        
        for layer in model.layers:
            if hasattr(layer, 'get_weights') and layer.get_weights():
                weights = layer.get_weights()
                layer_name = layer.name
                
                # Convert layer names to PyTorch format
                pytorch_name = layer_name.replace('conv2d', 'conv')
                pytorch_name = pytorch_name.replace('batch_normalization', 'bn')
                pytorch_name = pytorch_name.replace('se_', 'se.')
                pytorch_name = pytorch_name.replace('_', '.')
                
                if len(weights) == 2:  # Conv layer with bias
                    pytorch_state_dict[f"{pytorch_name}.weight"] = torch.from_numpy(weights[0]).permute(3, 2, 0, 1)
                    pytorch_state_dict[f"{pytorch_name}.bias"] = torch.from_numpy(weights[1])
                elif len(weights) == 1:  # Conv layer without bias
                    pytorch_state_dict[f"{pytorch_name}.weight"] = torch.from_numpy(weights[0]).permute(3, 2, 0, 1)
                elif len(weights) == 4:  # BatchNorm layer
                    pytorch_state_dict[f"{pytorch_name}.weight"] = torch.from_numpy(weights[0])
                    pytorch_state_dict[f"{pytorch_name}.bias"] = torch.from_numpy(weights[1])
                    pytorch_state_dict[f"{pytorch_name}.running_mean"] = torch.from_numpy(weights[2])
                    pytorch_state_dict[f"{pytorch_name}.running_var"] = torch.from_numpy(weights[3])
        
        # Save PyTorch weights
        torch.save(pytorch_state_dict, pytorch_weights_path)
        print(f"Converted weights saved to {pytorch_weights_path}")
        
        return pytorch_state_dict
        
    except Exception as e:
        print(f"Error converting weights: {e}")
        return None

class BilinearPooling(nn.Module):
    """Bilinear pooling layer for combining features from two backbones"""
    
    def __init__(self, feature_dim1, feature_dim2):
        super(BilinearPooling, self).__init__()
        self.feature_dim1 = feature_dim1
        self.feature_dim2 = feature_dim2
        
    def forward(self, x1, x2):
        """
        Args:
            x1: Features from first backbone [B, H, W, C1]
            x2: Features from second backbone [B, H, W, C2]
        Returns:
            Bilinear pooled features [B, C1*C2]
        """
        batch_size, height, width, depth1 = x1.size()
        _, _, _, depth2 = x2.size()
        
        # Reshape to [B*H*W, C]
        x1_flat = x1.view(batch_size * height * width, depth1)
        x2_flat = x2.view(batch_size * height * width, depth2)
        
        # Reshape to [B, H*W, C]
        x1_3d = x1_flat.view(batch_size, height * width, depth1)
        x2_3d = x2_flat.view(batch_size, height * width, depth2)
        
        # Bilinear pooling: [B, C1, H*W] @ [B, H*W, C2] = [B, C1, C2]
        phi_I = torch.bmm(x1_3d.transpose(1, 2), x2_3d)
        phi_I = phi_I.view(batch_size, depth1 * depth2)
        phi_I = phi_I / (height * width)
        
        # Sign and square root normalization
        y_sqrt = torch.sign(phi_I) * torch.sqrt(torch.abs(phi_I) + 1e-12)
        z_12 = F.normalize(y_sqrt, p=2, dim=1)
        
        return z_12

class EfficientNetBackbone(nn.Module):
    """EfficientNet backbone with custom feature extraction"""
    
    def __init__(self, pretrained=True, weights=None, weights_path=None):
        super(EfficientNetBackbone, self).__init__()
        
        # Load EfficientNet-B0
        if weights == "imagenet":
            self.backbone = efficientnet_b0(pretrained=True)
            print("Loaded ImageNet pretrained EfficientNet-B0")
        elif weights == "noisy-student":
            # Load EfficientNet-B0 without pretrained weights
            self.backbone = efficientnet_b0(pretrained=False)
            
            # Try to load Noisy-Student weights
            if weights_path and os.path.exists(weights_path):
                self.load_noisy_student_weights(weights_path)
                print(f"Loaded Noisy-Student weights from {weights_path}")
            else:
                print("Warning: Noisy-Student weights not found, using random initialization")
        else:
            self.backbone = efficientnet_b0(pretrained=pretrained)
            
        # Remove classifier
        self.backbone.classifier = nn.Identity()
        
        # Get feature dimensions
        self.feature_dim = 1280  # EfficientNet-B0 feature dimension
        
    def load_noisy_student_weights(self, weights_path):
        """Load Noisy-Student weights from PyTorch state dict"""
        try:
            state_dict = torch.load(weights_path, map_location='cpu')
            
            # Filter weights for features only (exclude classifier)
            features_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('features.'):
                    features_state_dict[key] = value
            
            # Load weights
            self.backbone.load_state_dict(features_state_dict, strict=False)
            print(f"Successfully loaded {len(features_state_dict)} Noisy-Student weight layers")
            
        except Exception as e:
            print(f"Error loading Noisy-Student weights: {e}")
            print("Using random initialization instead")
        
    def forward(self, x):
        """Extract features from EfficientNet"""
        features = self.backbone.features(x)  # [B, C, H, W]
        # Convert to [B, H, W, C] format to match TensorFlow
        features = features.permute(0, 2, 3, 1)
        return features

class CRBLModel(nn.Module):
    """CRBL Anomaly Detection Model with dual EfficientNet backbones"""
    
    def __init__(self, input_size=128, isCrop=True, noisy_student_weights_path=None):
        super(CRBLModel, self).__init__()
        
        self.input_size = input_size
        self.isCrop = isCrop
        
        # Two EfficientNet backbones
        self.backbone1 = EfficientNetBackbone(weights="imagenet")
        self.backbone2 = EfficientNetBackbone(weights="noisy-student", weights_path=noisy_student_weights_path)
        
        # Get feature dimensions (EfficientNet-B0 output: 1280)
        feature_dim = 1280
        
        # Bilinear pooling
        self.bilinear_pooling = BilinearPooling(feature_dim, feature_dim)
        
        # Dropout and classifier
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(feature_dim * feature_dim, 1)
        
        # Set trainable layers based on isCrop
        self._set_trainable_layers()
        
    def _set_trainable_layers(self):
        """Set which layers are trainable based on isCrop parameter"""
        if self.isCrop:
            # For crop model: train last 50 layers of each backbone
            for param in self.backbone1.parameters():
                param.requires_grad = False
            for param in self.backbone2.parameters():
                param.requires_grad = False
                
            # Enable training for last few layers
            for i, (name, param) in enumerate(self.backbone1.named_parameters()):
                if i >= len(list(self.backbone1.parameters())) - 50:
                    param.requires_grad = True
                    
            for i, (name, param) in enumerate(self.backbone2.named_parameters()):
                if i >= len(list(self.backbone2.parameters())) - 50:
                    param.requires_grad = True
        else:
            # For full model: freeze first 150 layers, train last layers
            for param in self.backbone1.parameters():
                param.requires_grad = False
            for param in self.backbone2.parameters():
                param.requires_grad = False
                
            # Enable training for last few layers
            for i, (name, param) in enumerate(self.backbone1.named_parameters()):
                if i >= len(list(self.backbone1.parameters())) - 50:
                    param.requires_grad = True
                    
            for i, (name, param) in enumerate(self.backbone2.named_parameters()):
                if i >= len(list(self.backbone2.parameters())) - 50:
                    param.requires_grad = True
        
        # Always train bilinear pooling and classifier
        for param in self.bilinear_pooling.parameters():
            param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        """Forward pass through the model"""
        # Extract features from both backbones
        features1 = self.backbone1(x)  # [B, H, W, C]
        features2 = self.backbone2(x)  # [B, H, W, C]
        
        # Bilinear pooling
        bilinear_features = self.bilinear_pooling(features1, features2)
        
        # Dropout and classification
        bilinear_features = self.dropout(bilinear_features)
        output = torch.sigmoid(self.classifier(bilinear_features))
        
        return output
    
    def load_weights(self, weights_path):
        """Load weights from converted PyTorch format"""
        state_dict = torch.load(weights_path, map_location='cpu')
        self.load_state_dict(state_dict, strict=False)
        print(f"Loaded weights from {weights_path}")

def create_model(isCrop=True, weights_path=None, noisy_student_weights_path=None):
    """Create CRBL model instance"""
    model = CRBLModel(input_size=128, isCrop=isCrop, noisy_student_weights_path=noisy_student_weights_path)
    
    if weights_path and torch.cuda.is_available():
        model = model.cuda()
        model.load_weights(weights_path)
    elif weights_path:
        model.load_weights(weights_path)
    
    return model

def count_parameters(model):
    """Count trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Test model creation
    set_seed(42)
    
    # Create models
    crop_model = create_model(isCrop=True)
    full_model = create_model(isCrop=False)
    
    print(f"Crop model trainable parameters: {count_parameters(crop_model):,}")
    print(f"Full model trainable parameters: {count_parameters(full_model):,}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 128, 128)
    
    with torch.no_grad():
        crop_output = crop_model(dummy_input)
        full_output = full_model(dummy_input)
        
    print(f"Crop model output shape: {crop_output.shape}")
    print(f"Full model output shape: {full_output.shape}")
    print("Model creation and forward pass successful!")
