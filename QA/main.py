"""
Main script for CRBL PyTorch implementation
Handles weight conversion, training, and inference
"""
import os
import argparse
import sys
from pathlib import Path

from utils import set_seed, setup_logging, print_environment_info, create_directories, print_gpu_memory_info
from weight_converter import main as convert_weights
from train_pytorch import train_model
from inference_pytorch import CRBLInference

def convert_tensorflow_weights():
    """Convert TensorFlow weights to PyTorch format"""
    print("Converting TensorFlow weights to PyTorch format...")
    try:
        convert_weights()
        print("Weight conversion completed successfully!")
        return True
    except Exception as e:
        print(f"Error during weight conversion: {e}")
        return False

def train_models(epochs=30, batch_size=32, learning_rate=0.0001):
    """Train both crop and full models"""
    print("Training models...")
    
    try:
        # Train crop model
        print("\n" + "="*50)
        print("TRAINING CROP MODEL")
        print("="*50)
        crop_model, crop_history = train_model(
            isCrop=True,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            class0_ratio=0.5,
            patience=10
        )
        
        # Train full model
        print("\n" + "="*50)
        print("TRAINING FULL MODEL")
        print("="*50)
        full_model, full_history = train_model(
            isCrop=False,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            class0_ratio=0.5,
            patience=10
        )
        
        print("Training completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during training: {e}")
        return False

def run_inference(crop_model_path=None, full_model_path=None, test_csv_path=None, image_dir=None):
    """Run inference with trained models"""
    print("Running inference...")
    
    try:
        # Set default paths if not provided
        if crop_model_path is None:
            crop_model_path = "./weights_pytorch/crop_model_pytorch.pth"
        if full_model_path is None:
            full_model_path = "./weights_pytorch/full_model_pytorch.pth"
        if test_csv_path is None:
            test_csv_path = "./data/csv/valid.csv"
        if image_dir is None:
            image_dir = "./data/images"
        
        # Check if model files exist
        if not os.path.exists(crop_model_path):
            print(f"Warning: Crop model not found at {crop_model_path}")
            crop_model_path = None
        if not os.path.exists(full_model_path):
            print(f"Warning: Full model not found at {full_model_path}")
            full_model_path = None
        
        if crop_model_path is None and full_model_path is None:
            print("Error: No trained models found. Please train models first.")
            return False
        
        # Initialize inference
        inference = CRBLInference(
            crop_model_path=crop_model_path,
            full_model_path=full_model_path
        )
        
        # Run combined prediction if both models are available
        if crop_model_path and full_model_path:
            results_df = inference.predict_combined(
                csv_path=test_csv_path,
                image_dir=image_dir,
                crop_threshold=0.01,
                full_threshold=0.5,
                batch_size=32,
                save_results=True
            )
            
            # Visualize results
            inference.visualize_predictions(
                results_df, 
                num_samples=10, 
                save_path="./results/prediction_visualization.png"
            )
        
        # Run individual model predictions
        elif crop_model_path:
            results_df = inference.predict_from_csv(
                csv_path=test_csv_path,
                image_dir=image_dir,
                isCrop=True,
                threshold=0.01,
                batch_size=32,
                save_results=True,
                output_path="./results/crop_predictions.csv"
            )
        
        elif full_model_path:
            results_df = inference.predict_from_csv(
                csv_path=test_csv_path,
                image_dir=image_dir,
                isCrop=False,
                threshold=0.5,
                batch_size=32,
                save_results=True,
                output_path="./results/full_predictions.csv"
            )
        
        print("Inference completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during inference: {e}")
        return False

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='CRBL PyTorch Implementation')
    parser.add_argument('--mode', type=str, choices=['convert', 'train', 'inference', 'all'], 
                       default='all', help='Mode to run')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--crop_model_path', type=str, help='Path to crop model weights')
    parser.add_argument('--full_model_path', type=str, help='Path to full model weights')
    parser.add_argument('--test_csv_path', type=str, help='Path to test CSV file')
    parser.add_argument('--image_dir', type=str, help='Path to image directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Log directory')
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Setup logging
    logger = setup_logging(args.log_dir)
    logger.info(f"Starting CRBL PyTorch implementation with mode: {args.mode}")
    
    # Print environment information
    print_environment_info()
    
    # Print GPU memory information
    print_gpu_memory_info()
    
    # Create output directories
    dirs = create_directories()
    logger.info(f"Created output directories: {dirs}")
    
    success = True
    
    try:
        if args.mode in ['convert', 'all']:
            logger.info("Starting weight conversion...")
            success &= convert_tensorflow_weights()
        
        if args.mode in ['train', 'all'] and success:
            logger.info("Starting model training...")
            success &= train_models(
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate
            )
        
        if args.mode in ['inference', 'all'] and success:
            logger.info("Starting inference...")
            success &= run_inference(
                crop_model_path=args.crop_model_path,
                full_model_path=args.full_model_path,
                test_csv_path=args.test_csv_path,
                image_dir=args.image_dir
            )
        
        if success:
            logger.info("All operations completed successfully!")
            print("\n" + "="*80)
            print("CRBL PyTorch implementation completed successfully!")
            print("="*80)
            print("Output files:")
            print(f"  - Weights: {dirs['weights']}")
            print(f"  - Results: {dirs['results']}")
            print(f"  - Logs: {dirs['logs']}")
            print(f"  - Plots: {dirs['plots']}")
            print("="*80)
        else:
            logger.error("Some operations failed. Check logs for details.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
