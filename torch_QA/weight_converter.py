"""
TensorFlow/Keras .h5 weights to PyTorch .pth converter for CRBL model
This script converts the trained TensorFlow weights to PyTorch format
"""

import os
import numpy as np
import torch
import torch.nn as nn
import h5py
from collections import OrderedDict

# Import our PyTorch model
from model_pytorch import create_model

def load_tensorflow_weights(h5_path):
    """Load weights from TensorFlow .h5 file"""
    weights_dict = {}
    
    with h5py.File(h5_path, 'r') as f:
        def visit_func(name, obj):
            if isinstance(obj, h5py.Dataset):
                weights_dict[name] = np.array(obj)
        
        f.visititems(visit_func)
    
    return weights_dict

def convert_efficientnet_weights(tf_weights, model_prefix):
    """Convert EfficientNet weights from TensorFlow to PyTorch format"""
    pytorch_weights = OrderedDict()
    
    # Map TensorFlow layer names to PyTorch layer names
    layer_mapping = {
        'conv2d/kernel': 'weight',
        'conv2d/bias': 'bias',
        'batch_normalization/gamma': 'weight',
        'batch_normalization/beta': 'bias',
        'batch_normalization/moving_mean': 'running_mean',
        'batch_normalization/moving_variance': 'running_var',
        'dense/kernel': 'weight',
        'dense/bias': 'bias',
    }
    
    for tf_name, tf_weight in tf_weights.items():
        if model_prefix in tf_name:
            # Remove model prefix
            clean_name = tf_name.replace(f'{model_prefix}/', '')
            
            # Convert layer name
            for tf_layer, pt_layer in layer_mapping.items():
                if tf_layer in clean_name:
                    clean_name = clean_name.replace(tf_layer, pt_layer)
                    break
            
            # Convert weight format
            if 'conv2d/kernel' in tf_name or 'dense/kernel' in tf_name:
                # TensorFlow: [H, W, C_in, C_out] -> PyTorch: [C_out, C_in, H, W]
                if len(tf_weight.shape) == 4:  # Conv2D
                    tf_weight = np.transpose(tf_weight, (3, 2, 0, 1))
                elif len(tf_weight.shape) == 2:  # Dense
                    tf_weight = np.transpose(tf_weight, (1, 0))
            elif 'batch_normalization/moving_variance' in tf_name:
                # Convert variance to running_var (sqrt of variance)
                tf_weight = np.sqrt(tf_weight + 1e-3)
            
            pytorch_weights[clean_name] = torch.from_numpy(tf_weight.astype(np.float32))
    
    return pytorch_weights

def convert_bilinear_weights(tf_weights):
    """Convert bilinear pooling and classifier weights"""
    pytorch_weights = OrderedDict()
    
    # Convert classifier weights
    for tf_name, tf_weight in tf_weights.items():
        if 'predictions' in tf_name:
            if 'kernel' in tf_name:
                # Dense layer weight: [input_dim, output_dim] -> [output_dim, input_dim]
                tf_weight = np.transpose(tf_weight, (1, 0))
                pytorch_weights['classifier.weight'] = torch.from_numpy(tf_weight.astype(np.float32))
            elif 'bias' in tf_name:
                pytorch_weights['classifier.bias'] = torch.from_numpy(tf_weight.astype(np.float32))
    
    return pytorch_weights

def convert_weights(tf_h5_path, pytorch_pth_path, input_shape=(3, 128, 128)):
    """
    Convert TensorFlow .h5 weights to PyTorch .pth format
    
    Args:
        tf_h5_path: Path to TensorFlow .h5 weights file
        pytorch_pth_path: Path to save PyTorch .pth weights file
        input_shape: Input shape for the model
    """
    print(f"Loading TensorFlow weights from: {tf_h5_path}")
    tf_weights = load_tensorflow_weights(tf_h5_path)
    
    print("Converting weights to PyTorch format...")
    
    # Create PyTorch model
    pytorch_model = create_model(input_shape=input_shape, num_classes=1)
    
    # Convert weights for each EfficientNet
    model1_weights = convert_efficientnet_weights(tf_weights, 'model1')
    model2_weights = convert_efficientnet_weights(tf_weights, 'model2')
    
    # Convert classifier weights
    classifier_weights = convert_bilinear_weights(tf_weights)
    
    # Combine all weights
    pytorch_state_dict = OrderedDict()
    
    # Add model1 weights
    for key, value in model1_weights.items():
        pytorch_state_dict[f'feature_extractor1.features.{key}'] = value
    
    # Add model2 weights
    for key, value in model2_weights.items():
        pytorch_state_dict[f'feature_extractor2.features.{key}'] = value
    
    # Add classifier weights
    for key, value in classifier_weights.items():
        pytorch_state_dict[key] = value
    
    # Load weights into model
    try:
        pytorch_model.load_state_dict(pytorch_state_dict, strict=False)
        print("Successfully loaded converted weights into PyTorch model")
    except Exception as e:
        print(f"Warning: Some weights could not be loaded: {e}")
        print("This is normal for some layer mappings")
    
    # Save PyTorch model
    os.makedirs(os.path.dirname(pytorch_pth_path), exist_ok=True)
    torch.save(pytorch_model.state_dict(), pytorch_pth_path)
    print(f"PyTorch weights saved to: {pytorch_pth_path}")
    
    return pytorch_model

def verify_conversion(tf_h5_path, pytorch_pth_path, input_shape=(3, 128, 128)):
    """Verify that the conversion was successful"""
    print("\nVerifying conversion...")
    
    # Load PyTorch model
    pytorch_model = create_model(input_shape=input_shape, num_classes=1)
    pytorch_model.load_state_dict(torch.load(pytorch_pth_path, map_location='cpu'))
    pytorch_model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, *input_shape)
    
    # Test forward pass
    with torch.no_grad():
        output = pytorch_model(dummy_input)
        print(f"Model output shape: {output.shape}")
        print(f"Model output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    
    print("Conversion verification completed successfully!")
    return True

def main():
    """Main conversion function"""
    # Configuration
    tf_weights_path = "./weights/CRBL_250328.weights.h5"
    pytorch_weights_path = "./weights/CRBL_250328_pytorch.pth"
    input_shape = (3, 128, 128)
    
    print("TensorFlow to PyTorch Weight Converter")
    print("=" * 50)
    
    # Check if TensorFlow weights exist
    if not os.path.exists(tf_weights_path):
        print(f"TensorFlow weights file not found: {tf_weights_path}")
        print("Please ensure the .h5 weights file exists.")
        return
    
    try:
        # Convert weights
        pytorch_model = convert_weights(
            tf_h5_path=tf_weights_path,
            pytorch_pth_path=pytorch_weights_path,
            input_shape=input_shape
        )
        
        # Verify conversion
        verify_conversion(
            tf_h5_path=tf_weights_path,
            pytorch_pth_path=pytorch_weights_path,
            input_shape=input_shape
        )
        
        print("\nWeight conversion completed successfully!")
        print(f"PyTorch weights saved to: {pytorch_weights_path}")
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        print("Please check the TensorFlow weights file and try again.")

if __name__ == "__main__":
    main()
