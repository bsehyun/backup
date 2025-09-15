import os 
import numpy as np 
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
from pathlib import Path

# Import our custom modules
from model_pytorch_directml import create_model, count_parameters, get_device
from loadDataset_generator_pytorch import load_data

# Set device (DirectML > CUDA > CPU)
device = get_device()

# Configuration
model_save_name = "CRBL_250328_pytorch_directml"
learning_rate = 0.0001 
weight_decay = 1e-6
label_smoothing = 0.125

num_classes = 1  # Binary classification
input_dim = 128
channel = 3 
batch_size = 32 
input_shape = (channel, input_dim, input_dim)

class0_ratio = 0.5 
c0 = 0.88
c1 = 1 - c0
class_weights = torch.tensor([(c1+c0)/(2*c0), (c1+c0)/(2*c1)], dtype=torch.float32).to(device)

# Early stopping parameters
patience = 10
min_delta = 1e-6

class EarlyStopping:
    """Early stopping utility"""
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.squeeze(), target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        pred = (output.squeeze() > 0.5).float()
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output.squeeze(), target)
            
            running_loss += loss.item()
            pred = (output.squeeze() > 0.5).float()
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def calculate_metrics(model, val_loader, device):
    """Calculate precision, recall, and F1 score"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = (output.squeeze() > 0.5).float()
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Calculate metrics
    tp = np.sum((all_preds == 1) & (all_targets == 1))
    fp = np.sum((all_preds == 1) & (all_targets == 0))
    fn = np.sum((all_preds == 0) & (all_targets == 1))
    tn = np.sum((all_preds == 0) & (all_targets == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

def main():
    # Load data
    print("Loading data...")
    train_loader, val_loader = load_data(
        input_dim=input_dim,
        datadir="./data/images",
        batch_size=batch_size,
        class0_ratio=class0_ratio,
        test_csv_path="./data/csv/test_CRBL.csv",
        image_dir="./data/images",
        num_workers=4
    )
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Create model
    print("Creating model...")
    model = create_model(input_shape=input_shape, num_classes=num_classes)
    model = model.to(device)
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Loss function with class weights
    criterion = nn.BCELoss(weight=class_weights)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
    
    # Training history
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    print("Starting training...")
    start_time = time.time()
    
    for epoch in range(30):  # Max 30 epochs
        print(f"\nEpoch {epoch+1}/30")
        print("-" * 50)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        # Calculate additional metrics
        precision, recall, f1 = calculate_metrics(model, val_loader, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Store history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Print epoch results
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping check
        if early_stopping(val_loss, model):
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    # Save model
    os.makedirs("./weights", exist_ok=True)
    torch.save(model.state_dict(), f"./weights/{model_save_name}.pth")
    print(f"Model saved to ./weights/{model_save_name}.pth")
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot([optimizer.param_groups[0]['lr']] * len(train_losses))
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    
    plt.tight_layout()
    plt.savefig(f"./weights/{model_save_name}_training_history.png")
    plt.show()
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
