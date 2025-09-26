"""
Download and convert Noisy-Student EfficientNet-B0 weights for PyTorch
"""
import os
import torch
import numpy as np
from pathlib import Path
import urllib.request
import tempfile
import zipfile

def download_noisy_student_from_tfhub():
    """Download Noisy-Student weights from TensorFlow Hub"""
    print("Downloading Noisy-Student EfficientNet-B0 weights from TensorFlow Hub...")
    
    try:
        import tensorflow as tf
        import tensorflow_hub as hub
        
        # Create weights directory
        weights_dir = Path("./pretrained_weights")
        weights_dir.mkdir(exist_ok=True)
        
        # TensorFlow Hub URL for Noisy-Student EfficientNet-B0
        # Note: This is the official Noisy-Student model
        url = "https://tfhub.dev/google/efficientnet/b0/classification/1"
        
        # Load model from TensorFlow Hub
        print(f"Loading model from: {url}")
        model = hub.load(url)
        
        # Save the model
        tf_model_path = weights_dir / "efficientnet-b0_noisy-student_tf"
        tf.saved_model.save(model, str(tf_model_path))
        
        print(f"TensorFlow model saved to: {tf_model_path}")
        return str(tf_model_path)
        
    except ImportError as e:
        print(f"Required packages not installed: {e}")
        print("Please install: pip install tensorflow tensorflow-hub")
        return None
    except Exception as e:
        print(f"Error downloading from TensorFlow Hub: {e}")
        return None

def download_noisy_student_from_github():
    """Download Noisy-Student weights from GitHub releases"""
    print("Downloading Noisy-Student EfficientNet-B0 weights from GitHub...")
    
    try:
        import tensorflow as tf
        
        # Create weights directory
        weights_dir = Path("./pretrained_weights")
        weights_dir.mkdir(exist_ok=True)
        
        # GitHub release URL for Noisy-Student weights
        # This is a direct download of the .h5 weights file
        url = "https://github.com/tensorflow/tpu/releases/download/v1/efficientnet-b0_noisy-student.h5"
        
        weights_path = weights_dir / "efficientnet-b0_noisy-student.h5"
        
        if weights_path.exists():
            print(f"Weights already exist at: {weights_path}")
            return str(weights_path)
        
        print(f"Downloading from: {url}")
        urllib.request.urlretrieve(url, weights_path)
        
        print(f"Weights downloaded to: {weights_path}")
        return str(weights_path)
        
    except Exception as e:
        print(f"Error downloading from GitHub: {e}")
        return None

def convert_tf_weights_to_pytorch(tf_weights_path, pytorch_weights_path):
    """Convert TensorFlow weights to PyTorch format"""
    print(f"Converting TensorFlow weights to PyTorch format...")
    
    try:
        import tensorflow as tf
        
        # Load TensorFlow model
        if tf_weights_path.endswith('.h5'):
            # Load from .h5 file
            model = tf.keras.models.load_model(tf_weights_path)
        else:
            # Load from saved model
            model = tf.saved_model.load(tf_weights_path)
        
        # Extract weights and convert to PyTorch format
        pytorch_state_dict = {}
        
        # Get all layers with weights
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
                    # Convert from HWCK to NCHW format
                    pytorch_state_dict[f"features.{pytorch_name}.weight"] = torch.from_numpy(weights[0]).permute(3, 2, 0, 1)
                    pytorch_state_dict[f"features.{pytorch_name}.bias"] = torch.from_numpy(weights[1])
                elif len(weights) == 1:  # Conv layer without bias
                    pytorch_state_dict[f"features.{pytorch_name}.weight"] = torch.from_numpy(weights[0]).permute(3, 2, 0, 1)
                elif len(weights) == 4:  # BatchNorm layer
                    pytorch_state_dict[f"features.{pytorch_name}.weight"] = torch.from_numpy(weights[0])
                    pytorch_state_dict[f"features.{pytorch_name}.bias"] = torch.from_numpy(weights[1])
                    pytorch_state_dict[f"features.{pytorch_name}.running_mean"] = torch.from_numpy(weights[2])
                    pytorch_state_dict[f"features.{pytorch_name}.running_var"] = torch.from_numpy(weights[3])
        
        # Save PyTorch weights
        torch.save(pytorch_state_dict, pytorch_weights_path)
        print(f"PyTorch weights saved to: {pytorch_weights_path}")
        print(f"Converted {len(pytorch_state_dict)} weight tensors")
        
        return pytorch_state_dict
        
    except Exception as e:
        print(f"Error converting weights: {e}")
        return None

def create_pytorch_efficientnet_weights():
    """Create PyTorch EfficientNet weights from ImageNet pretrained model"""
    print("Creating PyTorch EfficientNet-B0 weights from ImageNet pretrained model...")
    
    try:
        from torchvision.models import efficientnet_b0
        
        # Load ImageNet pretrained model
        model = efficientnet_b0(pretrained=True)
        
        # Extract features weights
        features_state_dict = {}
        for name, param in model.features.named_parameters():
            features_state_dict[f"features.{name}"] = param.data
        
        # Save weights
        weights_dir = Path("./pretrained_weights")
        weights_dir.mkdir(exist_ok=True)
        
        pytorch_weights_path = weights_dir / "efficientnet-b0_imagenet.pth"
        torch.save(features_state_dict, pytorch_weights_path)
        
        print(f"ImageNet weights saved to: {pytorch_weights_path}")
        return str(pytorch_weights_path)
        
    except Exception as e:
        print(f"Error creating ImageNet weights: {e}")
        return None

def main():
    """Main function to download and convert Noisy-Student weights"""
    print("="*80)
    print("Noisy-Student EfficientNet-B0 Weight Downloader")
    print("="*80)
    
    # Create weights directory
    weights_dir = Path("./pretrained_weights")
    weights_dir.mkdir(exist_ok=True)
    
    pytorch_weights_path = weights_dir / "efficientnet-b0_noisy-student.pth"
    
    # Check if PyTorch weights already exist
    if pytorch_weights_path.exists():
        print(f"PyTorch weights already exist at: {pytorch_weights_path}")
        print("Skipping download and conversion.")
        return str(pytorch_weights_path)
    
    # Try to download from GitHub first (more reliable)
    tf_weights_path = download_noisy_student_from_github()
    
    if tf_weights_path is None:
        # Fallback to TensorFlow Hub
        print("GitHub download failed, trying TensorFlow Hub...")
        tf_weights_path = download_noisy_student_from_tfhub()
    
    if tf_weights_path is None:
        print("Failed to download Noisy-Student weights from both sources.")
        print("Creating ImageNet pretrained weights as fallback...")
        return create_pytorch_efficientnet_weights()
    
    # Convert to PyTorch format
    pytorch_state_dict = convert_tf_weights_to_pytorch(tf_weights_path, pytorch_weights_path)
    
    if pytorch_state_dict is None:
        print("Failed to convert weights. Creating ImageNet pretrained weights as fallback...")
        return create_pytorch_efficientnet_weights()
    
    print("="*80)
    print("Noisy-Student weights download and conversion completed!")
    print(f"PyTorch weights saved to: {pytorch_weights_path}")
    print("="*80)
    
    return str(pytorch_weights_path)

if __name__ == "__main__":
    main()
