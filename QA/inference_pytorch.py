"""
PyTorch inference script for CRBL anomaly detection model
"""
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

from pytorch_model import CRBLModel, set_seed
from pytorch_dataset import CRBLValidationDataset, get_transforms

class CRBLInference:
    """CRBL model inference class with batch processing"""
    
    def __init__(self, crop_model_path=None, full_model_path=None, device=None):
        """
        Initialize inference class
        
        Args:
            crop_model_path: Path to crop model weights
            full_model_path: Path to full model weights
            device: Device to run inference on
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load models
        self.crop_model = None
        self.full_model = None
        
        if crop_model_path and os.path.exists(crop_model_path):
            self.crop_model = self._load_model(crop_model_path, isCrop=True)
            print(f"Loaded crop model from {crop_model_path}")
        
        if full_model_path and os.path.exists(full_model_path):
            self.full_model = self._load_model(full_model_path, isCrop=False)
            print(f"Loaded full model from {full_model_path}")
    
    def _load_model(self, model_path, isCrop=True):
        """Load PyTorch model"""
        # Check for Noisy-Student weights
        noisy_student_weights_path = "./pretrained_weights/efficientnet-b0_noisy-student.pth"
        if not os.path.exists(noisy_student_weights_path):
            print("Noisy-Student weights not found. Using ImageNet pretrained weights.")
            noisy_student_weights_path = None
        
        model = CRBLModel(input_size=128, isCrop=isCrop, noisy_student_weights_path=noisy_student_weights_path)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()
        return model
    
    def predict_single_image(self, image_path, isCrop=True, threshold=0.5):
        """
        Predict anomaly for a single image
        
        Args:
            image_path: Path to image
            isCrop: Whether to use crop model or full model
            threshold: Classification threshold
            
        Returns:
            prediction: 0 or 1
            confidence: prediction confidence
        """
        model = self.crop_model if isCrop else self.full_model
        if model is None:
            raise ValueError(f"{'Crop' if isCrop else 'Full'} model not loaded")
        
        # Load and preprocess image
        image = self._preprocess_image(image_path, isCrop)
        image = image.unsqueeze(0).to(self.device)  # Add batch dimension
        
        # Predict
        with torch.no_grad():
            output = model(image)
            confidence = output.item()
            prediction = 1 if confidence > threshold else 0
        
        return prediction, confidence
    
    def predict_batch(self, image_paths, isCrop=True, threshold=0.5, batch_size=32):
        """
        Predict anomalies for a batch of images
        
        Args:
            image_paths: List of image paths
            isCrop: Whether to use crop model or full model
            threshold: Classification threshold
            batch_size: Batch size for processing
            
        Returns:
            predictions: List of predictions (0 or 1)
            confidences: List of confidence scores
        """
        model = self.crop_model if isCrop else self.full_model
        if model is None:
            raise ValueError(f"{'Crop' if isCrop else 'Full'} model not loaded")
        
        predictions = []
        confidences = []
        
        # Process in batches
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing batches"):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []
            
            # Load and preprocess batch
            for path in batch_paths:
                try:
                    image = self._preprocess_image(path, isCrop)
                    batch_images.append(image)
                except Exception as e:
                    print(f"Error loading image {path}: {e}")
                    # Use zero image as fallback
                    batch_images.append(torch.zeros(3, 128, 128))
            
            # Stack into batch tensor
            batch_tensor = torch.stack(batch_images).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = model(batch_tensor)
                batch_confidences = outputs.squeeze().cpu().numpy()
                batch_predictions = (batch_confidences > threshold).astype(int)
            
            predictions.extend(batch_predictions.tolist())
            confidences.extend(batch_confidences.tolist())
        
        return predictions, confidences
    
    def predict_from_csv(self, csv_path, image_dir, isCrop=True, threshold=0.5, 
                        batch_size=32, save_results=True, output_path=None):
        """
        Predict anomalies for images listed in CSV file
        
        Args:
            csv_path: Path to CSV file with image paths and labels
            image_dir: Directory containing images
            isCrop: Whether to use crop model or full model
            threshold: Classification threshold
            batch_size: Batch size for processing
            save_results: Whether to save results to file
            output_path: Path to save results (optional)
            
        Returns:
            results_df: DataFrame with predictions and ground truth
        """
        # Load CSV
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} images from {csv_path}")
        
        # Get image paths
        image_paths = [os.path.join(image_dir, row['image_path']) for _, row in df.iterrows()]
        
        # Predict
        predictions, confidences = self.predict_batch(
            image_paths, isCrop=isCrop, threshold=threshold, batch_size=batch_size
        )
        
        # Create results DataFrame
        results_df = df.copy()
        results_df['prediction'] = predictions
        results_df['confidence'] = confidences
        results_df['correct'] = (results_df['prediction'] == results_df['impurity']).astype(int)
        
        # Calculate metrics
        accuracy = results_df['correct'].mean()
        precision = results_df[results_df['prediction'] == 1]['correct'].mean() if (results_df['prediction'] == 1).any() else 0
        recall = results_df[results_df['impurity'] == 1]['correct'].mean() if (results_df['impurity'] == 1).any() else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\nResults for {'crop' if isCrop else 'full'} model:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        # Save results
        if save_results:
            if output_path is None:
                model_type = 'crop' if isCrop else 'full'
                output_path = f"./results/{model_type}_predictions.csv"
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            results_df.to_csv(output_path, index=False)
            print(f"Results saved to {output_path}")
        
        return results_df
    
    def predict_combined(self, csv_path, image_dir, crop_threshold=0.01, full_threshold=0.5,
                        batch_size=32, save_results=True, output_path=None):
        """
        Predict using both crop and full models with combined results
        
        Args:
            csv_path: Path to CSV file with image paths and labels
            image_dir: Directory containing images
            crop_threshold: Threshold for crop model
            full_threshold: Threshold for full model
            batch_size: Batch size for processing
            save_results: Whether to save results to file
            output_path: Path to save results (optional)
            
        Returns:
            results_df: DataFrame with combined predictions
        """
        if self.crop_model is None or self.full_model is None:
            raise ValueError("Both crop and full models must be loaded for combined prediction")
        
        print("Running combined prediction with both models...")
        
        # Load CSV
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} images from {csv_path}")
        
        # Get image paths
        image_paths = [os.path.join(image_dir, row['image_path']) for _, row in df.iterrows()]
        
        # Predict with crop model
        print("Predicting with crop model...")
        crop_predictions, crop_confidences = self.predict_batch(
            image_paths, isCrop=True, threshold=crop_threshold, batch_size=batch_size
        )
        
        # Predict with full model
        print("Predicting with full model...")
        full_predictions, full_confidences = self.predict_batch(
            image_paths, isCrop=False, threshold=full_threshold, batch_size=batch_size
        )
        
        # Combine predictions (OR logic: if either model predicts anomaly, final prediction is anomaly)
        combined_predictions = np.logical_or(
            np.array(crop_predictions) == 1, 
            np.array(full_predictions) == 1
        ).astype(int)
        
        # Create results DataFrame
        results_df = df.copy()
        results_df['crop_prediction'] = crop_predictions
        results_df['crop_confidence'] = crop_confidences
        results_df['full_prediction'] = full_predictions
        results_df['full_confidence'] = full_confidences
        results_df['combined_prediction'] = combined_predictions
        results_df['correct'] = (results_df['combined_prediction'] == results_df['impurity']).astype(int)
        
        # Calculate metrics
        accuracy = results_df['correct'].mean()
        precision = results_df[results_df['combined_prediction'] == 1]['correct'].mean() if (results_df['combined_prediction'] == 1).any() else 0
        recall = results_df[results_df['impurity'] == 1]['correct'].mean() if (results_df['impurity'] == 1).any() else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\nCombined Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        # Save results
        if save_results:
            if output_path is None:
                output_path = "./results/combined_predictions.csv"
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            results_df.to_csv(output_path, index=False)
            print(f"Results saved to {output_path}")
        
        return results_df
    
    def _preprocess_image(self, image_path, isCrop=True):
        """Preprocess single image for inference"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if isCrop:
            # Crop image (same logic as in dataset)
            from pytorch_dataset import get_box_WithWhitebackground, resize_with_aspect_ratio
            image = get_box_WithWhitebackground(image, dim=100)
            image = resize_with_aspect_ratio(image, 100)
        else:
            # Use full image with border removal
            image = image[4:, 4:]
        
        # Resize to target size
        image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_CUBIC)
        
        # Normalize and convert to tensor
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)  # HWC to CHW
        
        # Apply normalization (ImageNet stats)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = (image - mean) / std
        
        return image
    
    def visualize_predictions(self, results_df, num_samples=10, save_path=None):
        """Visualize prediction results"""
        # Get sample images
        correct_samples = results_df[results_df['correct'] == 1].head(num_samples // 2)
        incorrect_samples = results_df[results_df['correct'] == 0].head(num_samples // 2)
        
        fig, axes = plt.subplots(2, num_samples // 2, figsize=(15, 6))
        fig.suptitle('Prediction Results', fontsize=16)
        
        for i, (_, row) in enumerate(correct_samples.iterrows()):
            if num_samples // 2 > 1:
                ax = axes[0, i]
            else:
                ax = axes[0]
            
            # Load and display image
            image_path = os.path.join("./data/images", row['image_path'])
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            ax.imshow(image)
            ax.set_title(f"Correct\nPred: {row['combined_prediction']}, GT: {row['impurity']}")
            ax.axis('off')
        
        for i, (_, row) in enumerate(incorrect_samples.iterrows()):
            if num_samples // 2 > 1:
                ax = axes[1, i]
            else:
                ax = axes[1]
            
            # Load and display image
            image_path = os.path.join("./data/images", row['image_path'])
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            ax.imshow(image)
            ax.set_title(f"Incorrect\nPred: {row['combined_prediction']}, GT: {row['impurity']}")
            ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

def main():
    """Main inference function"""
    # Set seed for reproducibility
    set_seed(42)
    
    # Model paths
    crop_model_path = "./weights_pytorch/crop_model_pytorch.pth"
    full_model_path = "./weights_pytorch/full_model_pytorch.pth"
    
    # Data paths
    test_csv_path = "./data/csv/valid.csv"
    image_dir = "./data/images"
    
    # Initialize inference
    inference = CRBLInference(
        crop_model_path=crop_model_path,
        full_model_path=full_model_path
    )
    
    # Run combined prediction
    results_df = inference.predict_combined(
        csv_path=test_csv_path,
        image_dir=image_dir,
        crop_threshold=0.01,
        full_threshold=0.5,
        batch_size=32,
        save_results=True
    )
    
    # Visualize results
    inference.visualize_predictions(results_df, num_samples=10, save_path="./results/prediction_visualization.png")
    
    print("Inference completed!")

if __name__ == "__main__":
    main()
