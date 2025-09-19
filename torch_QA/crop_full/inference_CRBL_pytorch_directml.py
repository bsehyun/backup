import os 
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import cv2 
from pathlib import Path

# Import our custom modules
from model_pytorch_dual_directml import (
    create_crop_model, create_full_model, create_model, get_device
)
from loadDataset_generator_pytorch import crop_image, origin_image, TestDataset

# Configuration
INPUT_DIM = 128
INPUT_SHAPE = (3, INPUT_DIM, INPUT_DIM)
CROPPED_THRESHOLD = 0.01 
FULL_THRESHOLD = 0.5 

# Model paths
crop_weight_path = "./weights/CRBL_250328_pytorch_directml_crop.pth"
full_weight_path = "./weights/CRBL_250328_pytorch_directml_full.pth"
test_csv_path = "./data/csv/valid.csv"
image_path = "./data/images"

# Set device (DirectML > CUDA > CPU)
device = get_device()

# Model type selection
MODEL_TYPE = "crop"  # "crop" or "full"
weight_path = crop_weight_path if MODEL_TYPE == "crop" else full_weight_path

print(f"Using {MODEL_TYPE} model for inference with DirectML...")
print(f"Model path: {weight_path}")

class InferenceModel:
    """Inference wrapper for CRBL model with DirectML support and crop/full models"""
    def __init__(self, model_path, input_shape, device, model_type="crop"):
        self.device = device
        self.input_shape = input_shape
        self.model_type = model_type
        
        # Load model based on type
        if model_type == "crop":
            self.model = create_crop_model(input_shape=input_shape, num_classes=1)
        elif model_type == "full":
            self.model = create_full_model(input_shape=input_shape, num_classes=1)
        else:
            raise ValueError("model_type must be 'crop' or 'full'")
            
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])
        
        print(f"Loaded {model_type} model successfully with DirectML")
        
    def preprocess_image(self, image):
        """Preprocess single image for inference"""
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor
        if isinstance(image, np.ndarray):
            image = self.transform(image)
        
        # Add batch dimension
        image = image.unsqueeze(0)
        return image.to(self.device)
    
    def predict_single(self, image):
        """Predict anomaly for single image"""
        with torch.no_grad():
            image_tensor = self.preprocess_image(image)
            output = self.model(image_tensor)
            probability = output.squeeze().cpu().item()
            return probability
    
    def predict_batch(self, images):
        """Predict anomaly for batch of images"""
        with torch.no_grad():
            if isinstance(images, list):
                # Process list of images
                batch_tensors = []
                for img in images:
                    if isinstance(img, str):
                        img = cv2.imread(img)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    batch_tensors.append(self.transform(img))
                batch_tensor = torch.stack(batch_tensors).to(self.device)
            else:
                # Process numpy array batch
                batch_tensor = torch.from_numpy(images.transpose(0, 3, 1, 2)).float().to(self.device)
            
            outputs = self.model(batch_tensor)
            probabilities = outputs.squeeze().cpu().numpy()
            return probabilities

def get_boxImage(image):
    """Extract bounding box from image (same as original)"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    mask = np.zeros_like(image[:,:,0])
    color_range = {
        "black": [(0), (50)],
        "gray": [(50), (150)],
    }

    for color, (lower, upper) in color_range.items():
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        color_mask = cv2.inRange(gray, lower, upper)
        mask = cv2.bitwise_or(mask, color_mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image  # Return original if no contours found
        
    largest_contour = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(largest_contour)

    if (y+h>89):
        h = 89-y
    
    image = image[y+1:y+h-1, x+1:x+w-1]

    return image

def resize_with_aspect_ratio(image, target_size=224):
    """Resize image with aspect ratio preservation"""
    resized_image = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return resized_image

def crop_image_for_inference(test_csv_path, image_path, input_dim):
    """Crop images for inference"""
    test_label_df = pd.read_csv(test_csv_path)
    test_images = [] 
    test_labels = [] 

    for _, row in test_label_df.iterrows():
        path = Path(image_path).joinpath(row["image_path"])

        image = cv2.imread(str(path))
        cropped_image = get_boxImage(image)
        image = resize_with_aspect_ratio(cropped_image, 100)
        image = cv2.resize(image, (input_dim, input_dim), interpolation=cv2.INTER_CUBIC)

        image_array = np.array(image, dtype=np.float32)
        test_images.append(image_array)
        test_labels.append(int(row["impurity"]))
        
    X = np.array(test_images, dtype=np.float32)/255.0
    return X, test_labels

def origin_image_for_inference(test_csv_path, image_path, input_dim):
    """Load original images for inference"""
    test_label_df = pd.read_csv(test_csv_path)
    test_images = []
    test_labels = []

    for _, row in test_label_df.iterrows():
        path = Path(image_path).joinpath(row["image_path"])

        image = cv2.imread(str(path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = image[4:, 4:]
        image = cv2.resize(image, (input_dim, input_dim), interpolation=cv2.INTER_CUBIC)

        image_array = np.array(image, dtype=np.float32)
        test_images.append(image_array)
        test_labels.append(int(row["impurity"]))

    X = np.array(test_images, dtype=np.float32)/255.0
    return X, test_labels

def evaluate_model(model_path, test_csv_path, image_path, input_dim, model_type="crop", threshold=0.5):
    """Evaluate model on test dataset"""
    print("Loading inference model...")
    inference_model = InferenceModel(model_path, INPUT_SHAPE, device, model_type)
    
    print("Loading test data...")
    # Load cropped images
    X_cropped, y_cropped = crop_image_for_inference(test_csv_path, image_path, input_dim)
    
    # Load original images
    X_original, y_original = origin_image_for_inference(test_csv_path, image_path, input_dim)
    
    print(f"Test samples: {len(X_cropped)}")
    
    # Select appropriate data based on model type
    if model_type == "crop":
        X_test, y_test = X_cropped, y_cropped
        print("Using cropped images for inference")
    else:
        X_test, y_test = X_original, y_original
        print("Using original images for inference")
    
    # Predict on test data
    print(f"Predicting on {model_type} images...")
    predictions = inference_model.predict_batch(X_test)
    
    # Calculate metrics
    pred_binary = (predictions > threshold).astype(int)
    accuracy = np.mean(pred_binary == y_test)
    
    print(f"\nResults for {model_type} model with threshold {threshold}:")
    print(f"Accuracy: {accuracy:.4f}")
    
    # Also test on both types for comparison
    print("\nTesting on both image types for comparison:")
    predictions_cropped = inference_model.predict_batch(X_cropped)
    predictions_original = inference_model.predict_batch(X_original)
    
    pred_binary_cropped = (predictions_cropped > CROPPED_THRESHOLD).astype(int)
    accuracy_cropped = np.mean(pred_binary_cropped == y_cropped)
    
    pred_binary_original = (predictions_original > FULL_THRESHOLD).astype(int)
    accuracy_original = np.mean(pred_binary_original == y_original)
    
    print(f"Cropped images - Accuracy: {accuracy_cropped:.4f}")
    print(f"Original images - Accuracy: {accuracy_original:.4f}")
    
    # Detailed results
    print(f"\nDetailed Results:")
    print(f"Cropped images:")
    for i, (pred, true) in enumerate(zip(predictions_cropped, y_cropped)):
        print(f"  Sample {i}: Prediction={pred:.4f}, True={true}, Correct={pred_binary_cropped[i]==true}")
    
    print(f"\nOriginal images:")
    for i, (pred, true) in enumerate(zip(predictions_original, y_original)):
        print(f"  Sample {i}: Prediction={pred:.4f}, True={true}, Correct={pred_binary_original[i]==true}")
    
    return {
        'primary': {
            'predictions': predictions,
            'accuracy': accuracy,
            'binary_predictions': pred_binary,
            'model_type': model_type
        },
        'cropped': {
            'predictions': predictions_cropped,
            'accuracy': accuracy_cropped,
            'binary_predictions': pred_binary_cropped
        },
        'original': {
            'predictions': predictions_original,
            'accuracy': accuracy_original,
            'binary_predictions': pred_binary_original
        }
    }

def visualize_predictions(model_path, test_csv_path, image_path, input_dim, model_type="crop", num_samples=10):
    """Visualize model predictions"""
    print("Loading inference model...")
    inference_model = InferenceModel(model_path, INPUT_SHAPE, device, model_type)
    
    print("Loading test data...")
    X_cropped, y_cropped = crop_image_for_inference(test_csv_path, image_path, input_dim)
    X_original, y_original = origin_image_for_inference(test_csv_path, image_path, input_dim)
    
    # Predict
    predictions_cropped = inference_model.predict_batch(X_cropped)
    predictions_original = inference_model.predict_batch(X_original)
    
    # Visualize
    fig, axes = plt.subplots(2, num_samples, figsize=(20, 6))
    
    for i in range(min(num_samples, len(X_cropped))):
        # Cropped images
        axes[0, i].imshow(X_cropped[i])
        axes[0, i].set_title(f'Cropped\nPred: {predictions_cropped[i]:.3f}\nTrue: {y_cropped[i]}')
        axes[0, i].axis('off')
        
        # Original images
        axes[1, i].imshow(X_original[i])
        axes[1, i].set_title(f'Original\nPred: {predictions_original[i]:.3f}\nTrue: {y_original[i]}')
        axes[1, i].axis('off')
    
    plt.suptitle(f'{model_type.title()} Model Predictions (DirectML)', fontsize=16)
    plt.tight_layout()
    plt.show()

def main():
    """Main inference function"""
    print("CRBL Anomaly Detection - PyTorch Inference with DirectML")
    print("=" * 60)
    
    # Check if model exists
    if not os.path.exists(weight_path):
        print(f"Model file not found: {weight_path}")
        print("Please train the model first or check the path.")
        return
    
    # Select appropriate threshold
    threshold = CROPPED_THRESHOLD if MODEL_TYPE == "crop" else FULL_THRESHOLD
    
    # Evaluate model
    results = evaluate_model(
        model_path=weight_path,
        test_csv_path=test_csv_path,
        image_path=image_path,
        input_dim=INPUT_DIM,
        model_type=MODEL_TYPE,
        threshold=threshold
    )
    
    # Visualize predictions
    visualize_predictions(
        model_path=weight_path,
        test_csv_path=test_csv_path,
        image_path=image_path,
        input_dim=INPUT_DIM,
        model_type=MODEL_TYPE,
        num_samples=10
    )
    
    print("\nInference completed successfully!")

if __name__ == "__main__":
    main()

