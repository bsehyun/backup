"""
TensorFlow weights to PyTorch weights converter for CRBL anomaly detection model
"""
import os
import numpy as np
import torch
import torch.nn as nn
from tensorflow.keras.models import load_model
import tensorflow as tf
import efficientnet.tfkeras as efn
from tensorflow.keras.layers import Dense, Dropout, Input, Lambda
from tensorflow.keras.models import Model
import h5py

def outer_product(inputs):
    """TensorFlow outer product function for bilinear pooling"""
    x1, x2 = inputs 
    batch_size = tf.shape(x1)[0]
    height = tf.shape(x1)[1]
    width = tf.shape(x1)[2] 
    depth1 = x1.shape[3] 
    depth2 = x2.shape[3] 

    x1_flat = tf.reshape(x1, [batch_size*height*width, depth1])
    x2_flat = tf.reshape(x2, [batch_size*height*width, depth2])

    x1_3d = tf.reshape(x1_flat, [batch_size, height*width, depth1])
    x2_3d = tf.reshape(x2_flat, [batch_size, height*width, depth2])

    phi_I = tf.matmul(tf.transpose(x1_3d, [0,2,1]), x2_3d)
    phi_I = tf.reshape(phi_I, [batch_size, depth1*depth2])
    phi_I = phi_I / tf.cast(height*width, tf.float32) 

    y_sqrt = tf.sign(phi_I)*tf.sqrt(tf.abs(phi_I)+1e-12)
    z_12 = tf.nn.l2_normalize(y_sqrt, axis=1)

    return z_12

def get_tf_model(input_shape, isCrop):
    """Recreate TensorFlow model architecture"""
    input_tensor = Input(shape=input_shape)

    base_model1 = efn.EfficientNetB0(weights="imagenet", include_top=False)
    base_model2 = efn.EfficientNetB0(weights="noisy-student", include_top=False)

    base_model1._name = "EfficientNetB0_imagenetWeight"
    base_model2._name = "EfficientNetB0_noisy-studentWeight"

    for layer in base_model1.layers:
        layer._name = "model1_" + layer._name
    
    for layer in base_model1.layers[150:]:
        if isCrop:
            layer.trainable = True 
        else:
            layer.trainable = False
    for layer in base_model1.layers[:150]:
        layer.trainable = False

    for layer in base_model2.layers:
        layer._name = "model2_" + layer._name

    for layer in base_model2.layers[150:]:
        if isCrop:
            layer.trainable = True 
        else:
            layer.trainable = False
    for layer in base_model2.layers[:150]:
        layer.trainable = False

    d1 = base_model1(input_tensor)
    d2 = base_model2(input_tensor)

    bilinear = Lambda(outer_product)([d1, d2])
    bilinear = Dropout(0.5)(bilinear)
    predictions = Dense(1, activation="sigmoid", name="predictions")(bilinear)

    model = Model(inputs=input_tensor, outputs=predictions)
    return model

def convert_tf_to_pytorch_weights(tf_weights_path, pytorch_weights_path, isCrop=True):
    """
    Convert TensorFlow weights to PyTorch format
    
    Args:
        tf_weights_path: Path to TensorFlow .weights.h5 file
        pytorch_weights_path: Path to save PyTorch weights
        isCrop: Whether this is for crop model or full model
    """
    print(f"Converting weights from {tf_weights_path} to {pytorch_weights_path}")
    
    # Create TensorFlow model and load weights
    input_shape = (128, 128, 3)
    tf_model = get_tf_model(input_shape, isCrop)
    tf_model.load_weights(tf_weights_path)
    
    # Extract weights from TensorFlow model
    pytorch_state_dict = {}
    
    # Process EfficientNet weights
    for layer in tf_model.layers:
        if hasattr(layer, 'get_weights') and layer.get_weights():
            weights = layer.get_weights()
            layer_name = layer.name
            
            if 'model1_' in layer_name:
                # EfficientNet ImageNet weights
                pytorch_name = layer_name.replace('model1_', 'backbone1.')
                pytorch_name = pytorch_name.replace('conv2d', 'conv')
                pytorch_name = pytorch_name.replace('batch_normalization', 'bn')
                pytorch_name = pytorch_name.replace('se_', 'se.')
                pytorch_name = pytorch_name.replace('_', '.')
                
                if len(weights) == 2:  # Conv layer with bias
                    pytorch_state_dict[f"{pytorch_name}.weight"] = torch.from_numpy(weights[0]).permute(3, 2, 0, 1)  # HWCK -> NCHW
                    pytorch_state_dict[f"{pytorch_name}.bias"] = torch.from_numpy(weights[1])
                elif len(weights) == 1:  # Conv layer without bias
                    pytorch_state_dict[f"{pytorch_name}.weight"] = torch.from_numpy(weights[0]).permute(3, 2, 0, 1)
                elif len(weights) == 4:  # BatchNorm layer
                    pytorch_state_dict[f"{pytorch_name}.weight"] = torch.from_numpy(weights[0])
                    pytorch_state_dict[f"{pytorch_name}.bias"] = torch.from_numpy(weights[1])
                    pytorch_state_dict[f"{pytorch_name}.running_mean"] = torch.from_numpy(weights[2])
                    pytorch_state_dict[f"{pytorch_name}.running_var"] = torch.from_numpy(weights[3])
                    
            elif 'model2_' in layer_name:
                # EfficientNet Noisy-Student weights
                pytorch_name = layer_name.replace('model2_', 'backbone2.')
                pytorch_name = pytorch_name.replace('conv2d', 'conv')
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
                    
            elif layer_name == 'predictions':
                # Final classification layer
                pytorch_state_dict['classifier.weight'] = torch.from_numpy(weights[0]).T  # Transpose for PyTorch
                pytorch_state_dict['classifier.bias'] = torch.from_numpy(weights[1])
    
    # Save PyTorch state dict
    torch.save(pytorch_state_dict, pytorch_weights_path)
    print(f"PyTorch weights saved to {pytorch_weights_path}")
    
    return pytorch_state_dict

def main():
    """Convert both crop and full model weights"""
    # Create weights directory if it doesn't exist
    os.makedirs("./weights_pytorch", exist_ok=True)
    
    # Convert crop model weights
    crop_tf_path = "./weights/efficientnet_CRBL_origin_ExtraData_250401.weights.h5"
    crop_pytorch_path = "./weights_pytorch/crop_model_weights.pth"
    
    if os.path.exists(crop_tf_path):
        convert_tf_to_pytorch_weights(crop_tf_path, crop_pytorch_path, isCrop=True)
    else:
        print(f"Warning: {crop_tf_path} not found")
    
    # Convert full model weights
    full_tf_path = "./weights/efficientnet_CRBL_250328.weights.h5"
    full_pytorch_path = "./weights_pytorch/full_model_weights.pth"
    
    if os.path.exists(full_tf_path):
        convert_tf_to_pytorch_weights(full_tf_path, full_pytorch_path, isCrop=False)
    else:
        print(f"Warning: {full_tf_path} not found")

if __name__ == "__main__":
    main()
