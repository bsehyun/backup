import os 
import numpy as np 
import pandas as pd 
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2 
from pathlib import Path
import random
from PIL import Image

class StratifiedDataset(Dataset):
    def __init__(self, data_dir, sampling_strategy, target_size=(224, 224), transform=None):
        self.data_dir = data_dir 
        self.target_size = target_size 
        self.sampling_strategy = sampling_strategy 
        self.transform = transform
        self.class_0_images, self.class_1_images = self._load_image_paths()
        
        # Create balanced dataset indices
        self.indices = self._create_balanced_indices()
        
    def _load_image_paths(self):
        class_0_dir = os.path.join(self.data_dir, "class_0")
        class_1_dir = os.path.join(self.data_dir, "class_1")

        class_0_images = [os.path.join(class_0_dir, f) for f in os.listdir(class_0_dir) if f.endswith(".png")]
        class_1_images = [os.path.join(class_1_dir, f) for f in os.listdir(class_1_dir) if f.endswith(".png")]

        return class_0_images, class_1_images
    
    def _create_balanced_indices(self):
        # Create balanced sampling indices
        class_0_count = len(self.class_0_images)
        class_1_count = len(self.class_1_images)
        
        # Calculate how many samples we need from each class
        total_samples = min(class_0_count, class_1_count) * 2  # Balanced dataset
        class_0_samples = int(total_samples * self.sampling_strategy)
        class_1_samples = total_samples - class_0_samples
        
        # Create indices
        indices = []
        indices.extend([(0, i) for i in range(class_0_samples)])  # (class, index)
        indices.extend([(1, i) for i in range(class_1_samples)])
        
        return indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        class_type, img_idx = self.indices[idx]
        
        if class_type == 0:
            img_path = self.class_0_images[img_idx % len(self.class_0_images)]
            label = 0
        else:
            img_path = self.class_1_images[img_idx % len(self.class_1_images)]
            label = 1
            
        # Load image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_CUBIC)
        
        # Apply transforms if specified
        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
            
        return img, torch.tensor(label, dtype=torch.float32)

def get_box_WithWhitebackground(image, dim):
    """Extract bounding box from image with white background"""
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
    h, w, _ = image.shape 
    resized_image = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_AREA)
    
    return resized_image

class TestDataset(Dataset):
    def __init__(self, csv_path, image_dir, input_dim, transform=None):
        self.csv_path = csv_path
        self.image_dir = image_dir
        self.input_dim = input_dim
        self.transform = transform
        
        # Load test data
        self.test_label_df = pd.read_csv(csv_path)
        
    def __len__(self):
        return len(self.test_label_df)
    
    def __getitem__(self, idx):
        row = self.test_label_df.iloc[idx]
        image_path = Path(self.image_dir).joinpath(row["image_path"])

        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        cropped_image = get_box_WithWhitebackground(image, dim=100)
        image = resize_with_aspect_ratio(cropped_image, 100)
        image = cv2.resize(image, (self.input_dim, self.input_dim), interpolation=cv2.INTER_CUBIC)

        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            
        label = torch.tensor(int(row["impurity"]), dtype=torch.float32)
        
        return image, label

def load_data(input_dim, datadir, batch_size, class0_ratio, test_csv_path, image_dir, num_workers=4):
    """Load training and validation data"""
    
    # Define transforms for training data (data augmentation)
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(90),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomAffine(degrees=0, scale=(0.8, 1.2)),
        transforms.ToTensor(),
    ])
    
    # Define transforms for validation data
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    
    # Create datasets
    train_dataset = StratifiedDataset(
        data_dir=datadir,
        sampling_strategy=class0_ratio,
        target_size=(input_dim, input_dim),
        transform=train_transform
    )
    
    val_dataset = TestDataset(
        csv_path=test_csv_path,
        image_dir=image_dir,
        input_dim=input_dim,
        transform=val_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

def crop_image(test_csv_path, image_path, input_dim):
    """Crop images for inference (legacy function for compatibility)"""
    test_label_df = pd.read_csv(test_csv_path)
    test_images = [] 
    test_labels = [] 

    for _, row in test_label_df.iterrows():
        path = Path(image_path).joinpath(row["image_path"])

        image = cv2.imread(str(path))
        cropped_image = get_box_WithWhitebackground(image, 100)
        image = resize_with_aspect_ratio(cropped_image, 100)
        image = cv2.resize(image, (input_dim, input_dim), interpolation=cv2.INTER_CUBIC)

        image_array = np.array(image, dtype=np.float32)
        test_images.append(image_array)
        test_labels.append(int(row["impurity"]))
        
    X = np.array(test_images, dtype=np.float32)/255.0
    return X, test_labels

def origin_image(test_csv_path, image_path, input_dim):
    """Load original images for inference (legacy function for compatibility)"""
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
