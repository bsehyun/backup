"""
PyTorch training script for CRBL anomaly detection model
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
from pathlib import Path

from pytorch_model import CRBLModel, set_seed, count_parameters
from pytorch_dataset import create_data_loaders

class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience=10, min_delta=0, restore_best_weights=True):
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
        """Save model checkpoint"""
        self.best_weights = model.state_dict().copy()

class ReduceLROnPlateau:
    """Learning rate scheduler"""
    
    def __init__(self, optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6):
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.best = None
        self.num_bad_epochs = 0
        
    def step(self, metrics):
        if self.best is None:
            self.best = metrics
        elif self._is_better(metrics, self.best):
            self.best = metrics
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
            
        if self.num_bad_epochs >= self.patience:
            self._reduce_lr()
            self.num_bad_epochs = 0
    
    def _is_better(self, current, best):
        if self.mode == 'min':
            return current < best
        else:
            return current > best
    
    def _reduce_lr(self):
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)
            param_group['lr'] = new_lr
            print(f"Reducing learning rate from {old_lr:.2e} to {new_lr:.2e}")

def calculate_metrics(y_true, y_pred, threshold=0.5):
    """Calculate classification metrics"""
    y_pred_binary = (y_pred > threshold).astype(int)
    
    accuracy = accuracy_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary, zero_division=0)
    recall = recall_score(y_true, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    
    return accuracy, precision, recall, f1

def train_epoch(model, train_loader, criterion, optimizer, device, class_weights=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images).squeeze()
        
        # Calculate weighted loss
        if class_weights is not None:
            weights = torch.where(labels == 0, class_weights[0], class_weights[1]).to(device)
            loss = criterion(outputs, labels)
            loss = (loss * weights).mean()
        else:
            loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        all_predictions.extend(outputs.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
    
    avg_loss = total_loss / len(train_loader)
    accuracy, precision, recall, f1 = calculate_metrics(all_labels, all_predictions)
    
    return avg_loss, accuracy, precision, recall, f1

def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            all_predictions.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    accuracy, precision, recall, f1 = calculate_metrics(all_labels, all_predictions)
    
    return avg_loss, accuracy, precision, recall, f1

def train_model(isCrop=True, epochs=30, batch_size=32, learning_rate=0.0001, 
                class0_ratio=0.5, weight_decay=1e-6, patience=10):
    """Train CRBL model"""
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data paths
    class_datadir = "./data/images/classed_image"
    datadir = "./data/images"
    train_csv_path = "./data/csv/train_CRBL.csv"
    test_csv_path = "./data/csv/test_CRBL.csv"
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(
        class_datadir=class_datadir,
        datadir=datadir,
        isCrop=isCrop,
        batch_size=batch_size,
        class0_ratio=class0_ratio,
        train_csv_path=train_csv_path,
        test_csv_path=test_csv_path,
        num_workers=4,
        pin_memory=True
    )
    
    if train_loader is None:
        print("Error: Could not create train loader")
        return
    
    print(f"Train loader: {len(train_loader)} batches")
    if val_loader:
        print(f"Val loader: {len(val_loader)} batches")
    
    # Create model
    print("Creating model...")
    
    # Check for Noisy-Student weights
    noisy_student_weights_path = "./pretrained_weights/efficientnet-b0_noisy-student.pth"
    if not os.path.exists(noisy_student_weights_path):
        print("Noisy-Student weights not found. Downloading...")
        try:
            from download_noisy_student import main as download_noisy_student
            noisy_student_weights_path = download_noisy_student()
        except Exception as e:
            print(f"Warning: Could not download Noisy-Student weights: {e}")
            noisy_student_weights_path = None
    
    model = CRBLModel(input_size=128, isCrop=isCrop, noisy_student_weights_path=noisy_student_weights_path)
    model = model.to(device)
    
    print(f"Model trainable parameters: {count_parameters(model):,}")
    
    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Class weights (from original TensorFlow code)
    c0 = 0.88
    c1 = 1 - c0
    class_weights = torch.tensor([
        (c1 + c0) / (2 * c0),  # class 0 weight
        (c1 + c0) / (2 * c1)   # class 1 weight
    ])
    
    # Callbacks
    early_stopping = EarlyStopping(patience=patience)
    lr_scheduler = ReduceLROnPlateau(optimizer, patience=2, factor=0.5, min_lr=1e-6)
    
    # TensorBoard logging
    log_dir = f"./logs/{'crop' if isCrop else 'full'}_model"
    writer = SummaryWriter(log_dir)
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [], 'train_precision': [], 'train_recall': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_precision': [], 'val_recall': [], 'val_f1': []
    }
    
    print("Starting training...")
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc, train_precision, train_recall, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device, class_weights
        )
        
        # Validate
        if val_loader:
            val_loss, val_acc, val_precision, val_recall, val_f1 = validate_epoch(
                model, val_loader, criterion, device
            )
        else:
            val_loss = val_acc = val_precision = val_recall = val_f1 = 0
        
        # Update learning rate
        lr_scheduler.step(val_loss if val_loader else train_loss)
        
        # Log metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_precision'].append(train_precision)
        history['train_recall'].append(train_recall)
        history['train_f1'].append(train_f1)
        
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)
        
        # TensorBoard logging
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Val', val_acc, epoch)
        writer.add_scalar('Precision/Train', train_precision, epoch)
        writer.add_scalar('Precision/Val', val_precision, epoch)
        writer.add_scalar('Recall/Train', train_recall, epoch)
        writer.add_scalar('Recall/Val', val_recall, epoch)
        writer.add_scalar('F1/Train', train_f1, epoch)
        writer.add_scalar('F1/Val', val_f1, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        epoch_time = time.time() - epoch_start
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}')
        if val_loader:
            print(f'  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}')
        print(f'  Time: {epoch_time:.2f}s, LR: {optimizer.param_groups[0]["lr"]:.2e}')
        print('-' * 80)
        
        # Early stopping
        if val_loader and early_stopping(val_loss, model):
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f}s")
    
    # Save model
    os.makedirs("./weights_pytorch", exist_ok=True)
    model_name = f"{'crop' if isCrop else 'full'}_model_pytorch.pth"
    model_path = f"./weights_pytorch/{model_name}"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Save training history
    history_path = f"./weights_pytorch/{'crop' if isCrop else 'full'}_training_history.npy"
    np.save(history_path, history)
    print(f"Training history saved to {history_path}")
    
    writer.close()
    
    return model, history

def plot_training_history(history, isCrop=True):
    """Plot training history"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Training History - {"Crop" if isCrop else "Full"} Model', fontsize=16)
    
    metrics = ['loss', 'acc', 'precision', 'recall', 'f1']
    for i, metric in enumerate(metrics):
        row = i // 3
        col = i % 3
        
        axes[row, col].plot(history[f'train_{metric}'], label=f'Train {metric}')
        axes[row, col].plot(history[f'val_{metric}'], label=f'Val {metric}')
        axes[row, col].set_title(f'{metric.title()}')
        axes[row, col].set_xlabel('Epoch')
        axes[row, col].set_ylabel(metric.title())
        axes[row, col].legend()
        axes[row, col].grid(True)
    
    # Learning rate plot
    axes[1, 2].plot(history.get('lr', []), label='Learning Rate')
    axes[1, 2].set_title('Learning Rate')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Learning Rate')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig(f"./weights_pytorch/{'crop' if isCrop else 'full'}_training_plots.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Train crop model
    print("Training crop model...")
    crop_model, crop_history = train_model(
        isCrop=True,
        epochs=30,
        batch_size=32,
        learning_rate=0.0001,
        class0_ratio=0.5,
        patience=10
    )
    
    # Plot training history
    plot_training_history(crop_history, isCrop=True)
    
    # Train full model
    print("\nTraining full model...")
    full_model, full_history = train_model(
        isCrop=False,
        epochs=30,
        batch_size=32,
        learning_rate=0.0001,
        class0_ratio=0.5,
        patience=10
    )
    
    # Plot training history
    plot_training_history(full_history, isCrop=False)
    
    print("Training completed for both models!")
