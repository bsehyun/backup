"""
PyTorch dataset and data loader for CRBL anomaly detection
"""
import os
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import random
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_box_WithWhitebackground(image, dim):
    """Extract bounding box from image with white background detection"""
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

class CRBLDataset(Dataset):
    """PyTorch Dataset for CRBL anomaly detection"""
    
    def __init__(self, data_dir, isCrop=True, target_size=128, transform=None, 
                 sampling_strategy=0.5, mode='train'):
        """
        Args:
            data_dir: Directory containing class_0 and class_1 folders
            isCrop: Whether to crop images or use full images
            target_size: Target image size
            transform: Albumentations transform pipeline
            sampling_strategy: Ratio of class 0 samples in each batch
            mode: 'train' or 'val'
        """
        self.data_dir = data_dir
        self.isCrop = isCrop
        self.target_size = target_size
        self.transform = transform
        self.sampling_strategy = sampling_strategy
        self.mode = mode
        
        # Load image paths
        self.class_0_images, self.class_1_images = self._load_image_paths()
        
        # Create stratified indices for balanced sampling
        self._create_stratified_indices()
        
    def _load_image_paths(self):
        """Load image paths from class directories"""
        class_0_dir = os.path.join(self.data_dir, "class_0")
        class_1_dir = os.path.join(self.data_dir, "class_1")

        class_0_images = []
        class_1_images = []
        
        if os.path.exists(class_0_dir):
            class_0_images = [os.path.join(class_0_dir, f) for f in os.listdir(class_0_dir) 
                            if f.endswith((".png", ".jpg", ".jpeg"))]
        
        if os.path.exists(class_1_dir):
            class_1_images = [os.path.join(class_1_dir, f) for f in os.listdir(class_1_dir) 
                            if f.endswith((".png", ".jpg", ".jpeg"))]
        
        return class_0_images, class_1_images
    
    def _create_stratified_indices(self):
        """Create stratified indices for balanced sampling"""
        self.class_0_count = len(self.class_0_images)
        self.class_1_count = len(self.class_1_images)
        
        # Calculate how many samples we can create
        min_class_count = min(self.class_0_count, self.class_1_count)
        self.total_samples = int(min_class_count / self.sampling_strategy)
        
        # Create indices
        self.class_0_indices = list(range(self.class_0_count)) * (self.total_samples // self.class_0_count + 1)
        self.class_1_indices = list(range(self.class_1_count)) * (self.total_samples // self.class_1_count + 1)
        
        # Shuffle indices
        random.shuffle(self.class_0_indices)
        random.shuffle(self.class_1_indices)
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        """Get a single sample"""
        # Determine which class to sample
        if idx % 2 == 0:  # Even indices: class 0
            class_idx = idx // 2 % len(self.class_0_indices)
            img_path = self.class_0_images[self.class_0_indices[class_idx]]
            label = 0
        else:  # Odd indices: class 1
            class_idx = idx // 2 % len(self.class_1_indices)
            img_path = self.class_1_images[self.class_1_indices[class_idx]]
            label = 1
        
        # Load and process image
        image = cv2.imread(img_path)
        if image is None:
            # If image loading fails, return a black image
            image = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if self.isCrop:
                # Crop image
                image = get_box_WithWhitebackground(image, dim=100)
                image = resize_with_aspect_ratio(image, 100)
            else:
                # Use full image with border removal
                image = image[4:, 4:]
            
            # Resize to target size
            image = cv2.resize(image, (self.target_size, self.target_size), 
                             interpolation=cv2.INTER_CUBIC)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image=image)['image']
        else:
            # Default normalization
            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)  # HWC to CHW
        
        return image, torch.tensor(label, dtype=torch.float32)

class CRBLValidationDataset(Dataset):
    """Dataset for validation/test data from CSV"""
    
    def __init__(self, csv_path, image_dir, isCrop=True, target_size=128, transform=None):
        """
        Args:
            csv_path: Path to CSV file with image paths and labels
            image_dir: Directory containing images
            isCrop: Whether to crop images or use full images
            target_size: Target image size
            transform: Albumentations transform pipeline
        """
        self.csv_path = csv_path
        self.image_dir = image_dir
        self.isCrop = isCrop
        self.target_size = target_size
        self.transform = transform
        
        # Load data from CSV
        self.df = pd.read_csv(csv_path)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """Get a single sample"""
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row['image_path'])
        label = int(row['impurity'])
        
        # Load and process image
        image = cv2.imread(img_path)
        if image is None:
            # If image loading fails, return a black image
            image = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if self.isCrop:
                # Crop image
                image = get_box_WithWhitebackground(image, dim=100)
                image = resize_with_aspect_ratio(image, 100)
            else:
                # Use full image with border removal
                image = image[4:, 4:]
            
            # Resize to target size
            image = cv2.resize(image, (self.target_size, self.target_size), 
                             interpolation=cv2.INTER_CUBIC)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image=image)['image']
        else:
            # Default normalization
            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)  # HWC to CHW
        
        return image, torch.tensor(label, dtype=torch.float32)

def get_transforms(mode='train', target_size=128):
    """Get data augmentation transforms"""
    if mode == 'train':
        transform = A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.GaussNoise(p=0.1),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    return transform

def create_data_loaders(class_datadir, datadir, isCrop=True, batch_size=32, 
                       class0_ratio=0.5, train_csv_path=None, test_csv_path=None,
                       num_workers=4, pin_memory=True):
    """Create PyTorch data loaders"""
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Get transforms
    train_transform = get_transforms(mode='train', target_size=128)
    val_transform = get_transforms(mode='val', target_size=128)
    
    # Create datasets
    train_dataset = CRBLDataset(
        data_dir=class_datadir,
        isCrop=isCrop,
        target_size=128,
        transform=train_transform,
        sampling_strategy=class0_ratio,
        mode='train'
    )
    
    # Create validation dataset from CSV
    val_dataset = None
    if test_csv_path and os.path.exists(test_csv_path):
        val_dataset = CRBLValidationDataset(
            csv_path=test_csv_path,
            image_dir=datadir,
            isCrop=isCrop,
            target_size=128,
            transform=val_transform
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    
    return train_loader, val_loader

if __name__ == "__main__":
    # Test data loader creation
    set_seed(42)
    
    # Test parameters
    class_datadir = "./data/images/classed_image"
    datadir = "./data/images"
    test_csv_path = "./data/csv/test_CRBL.csv"
    
    # Test crop data loader
    print("Testing crop data loader...")
    crop_train_loader, crop_val_loader = create_data_loaders(
        class_datadir=class_datadir,
        datadir=datadir,
        isCrop=True,
        batch_size=4,
        test_csv_path=test_csv_path
    )
    
    if crop_train_loader:
        print(f"Crop train loader: {len(crop_train_loader)} batches")
        for batch_idx, (images, labels) in enumerate(crop_train_loader):
            print(f"Batch {batch_idx}: images shape {images.shape}, labels shape {labels.shape}")
            break
    
    if crop_val_loader:
        print(f"Crop val loader: {len(crop_val_loader)} batches")
    
    # Test full data loader
    print("\nTesting full data loader...")
    full_train_loader, full_val_loader = create_data_loaders(
        class_datadir=class_datadir,
        datadir=datadir,
        isCrop=False,
        batch_size=4,
        test_csv_path=test_csv_path
    )
    
    if full_train_loader:
        print(f"Full train loader: {len(full_train_loader)} batches")
    
    if full_val_loader:
        print(f"Full val loader: {len(full_val_loader)} batches")
    
    print("Data loader test completed!")
