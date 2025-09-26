"""
Utility functions for CRBL PyTorch implementation
"""
import os
import torch
import numpy as np
import random
import json
from pathlib import Path
import logging
from datetime import datetime

def set_seed(seed=42):
    """
    Set random seeds for reproducibility across all libraries
    
    Args:
        seed: Random seed value
    """
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # PyTorch deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables for deterministic behavior
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    # AMD GPU specific optimizations
    if torch.cuda.is_available():
        try:
            # Check if it's AMD GPU with ROCm
            if 'rocm' in torch.version.cuda.lower() or 'hip' in torch.version.cuda.lower():
                # ROCm specific optimizations
                os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'  # For compatibility
                os.environ['ROCM_PATH'] = '/opt/rocm'
                print("AMD GPU optimizations applied")
        except:
            pass
    
    # Enable deterministic algorithms (if available)
    try:
        torch.use_deterministic_algorithms(True)
    except:
        pass  # Not available in older PyTorch versions

def setup_logging(log_dir="./logs", log_level=logging.INFO):
    """
    Setup logging configuration
    
    Args:
        log_dir: Directory to save log files
        log_level: Logging level
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Create timestamp for log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"crbl_pytorch_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging setup complete. Log file: {log_file}")
    
    return logger

def save_config(config, save_path):
    """
    Save configuration to JSON file
    
    Args:
        config: Configuration dictionary
        save_path: Path to save configuration
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Configuration saved to {save_path}")

def load_config(config_path):
    """
    Load configuration from JSON file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        config: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config

def get_device():
    """
    Get the best available device (CUDA, ROCm, MPS, or CPU)
    
    Returns:
        device: PyTorch device
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_name = torch.cuda.get_device_name(0)
        
        # Check if it's AMD GPU with ROCm
        try:
            if 'rocm' in torch.version.cuda.lower() or 'hip' in torch.version.cuda.lower():
                print(f"Using AMD GPU with ROCm: {device_name}")
            else:
                print(f"Using NVIDIA GPU with CUDA: {device_name}")
        except:
            print(f"Using GPU device: {device_name}")
            
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS device (Apple Silicon)")
    else:
        device = torch.device('cpu')
        print("Using CPU device")
    
    return device

def check_amd_gpu_support():
    """
    Check if AMD GPU with ROCm is properly supported
    
    Returns:
        dict: Support information
    """
    support_info = {
        'rocm_available': False,
        'rocm_version': None,
        'amd_gpu_detected': False,
        'pytorch_rocm_support': False
    }
    
    # Check PyTorch ROCm support
    try:
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            if 'rocm' in cuda_version.lower() or 'hip' in cuda_version.lower():
                support_info['rocm_available'] = True
                support_info['rocm_version'] = cuda_version
                support_info['pytorch_rocm_support'] = True
                
                # Try to create a tensor to verify functionality
                test_tensor = torch.tensor([1.0]).cuda()
                support_info['amd_gpu_detected'] = True
    except Exception as e:
        support_info['error'] = str(e)
    
    return support_info

def count_parameters(model):
    """
    Count trainable and total parameters in a model
    
    Args:
        model: PyTorch model
        
    Returns:
        trainable_params: Number of trainable parameters
        total_params: Total number of parameters
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    return trainable_params, total_params

def print_model_summary(model, input_size=(3, 128, 128)):
    """
    Print model summary including parameter counts
    
    Args:
        model: PyTorch model
        input_size: Input tensor size for summary
    """
    trainable_params, total_params = count_parameters(model)
    
    print("=" * 80)
    print("MODEL SUMMARY")
    print("=" * 80)
    print(f"Model: {model.__class__.__name__}")
    print(f"Input size: {input_size}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print("=" * 80)

def create_directories(base_dir="./crbl_pytorch_outputs"):
    """
    Create necessary directories for outputs
    
    Args:
        base_dir: Base directory for all outputs
        
    Returns:
        dirs: Dictionary of created directories
    """
    dirs = {
        'base': base_dir,
        'weights': os.path.join(base_dir, 'weights'),
        'logs': os.path.join(base_dir, 'logs'),
        'results': os.path.join(base_dir, 'results'),
        'plots': os.path.join(base_dir, 'plots'),
        'configs': os.path.join(base_dir, 'configs')
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs

def save_model_info(model, save_path, additional_info=None):
    """
    Save model information to JSON file
    
    Args:
        model: PyTorch model
        save_path: Path to save model info
        additional_info: Additional information to save
    """
    trainable_params, total_params = count_parameters(model)
    
    model_info = {
        'model_name': model.__class__.__name__,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params,
        'model_state_dict_keys': list(model.state_dict().keys()),
        'additional_info': additional_info or {}
    }
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(model_info, f, indent=4)
    
    print(f"Model info saved to {save_path}")

def compare_models(model1, model2, tolerance=1e-6):
    """
    Compare two models to check if they are identical
    
    Args:
        model1: First PyTorch model
        model2: Second PyTorch model
        tolerance: Tolerance for parameter comparison
        
    Returns:
        is_identical: Boolean indicating if models are identical
        differences: List of differences found
    """
    differences = []
    
    # Check if models have same structure
    if len(model1.state_dict()) != len(model2.state_dict()):
        differences.append("Different number of parameters")
        return False, differences
    
    # Compare parameters
    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        if name1 != name2:
            differences.append(f"Different parameter names: {name1} vs {name2}")
            continue
        
        if param1.shape != param2.shape:
            differences.append(f"Different shapes for {name1}: {param1.shape} vs {param2.shape}")
            continue
        
        if not torch.allclose(param1, param2, atol=tolerance):
            max_diff = torch.max(torch.abs(param1 - param2)).item()
            differences.append(f"Different values for {name1}, max difference: {max_diff}")
    
    is_identical = len(differences) == 0
    return is_identical, differences

def benchmark_inference(model, input_tensor, num_runs=100, warmup_runs=10):
    """
    Benchmark model inference speed
    
    Args:
        model: PyTorch model
        input_tensor: Input tensor for inference
        num_runs: Number of inference runs for timing
        warmup_runs: Number of warmup runs
        
    Returns:
        avg_time: Average inference time in seconds
        std_time: Standard deviation of inference time
    """
    import time
    
    model.eval()
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_tensor)
    
    # Timing runs
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == 'cuda':
                # Use CUDA events for GPU timing (works for both NVIDIA and AMD)
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                _ = model(input_tensor)
                end_time.record()
                torch.cuda.synchronize()
                times.append(start_time.elapsed_time(end_time) / 1000.0)  # Convert to seconds
            else:
                # Use CPU timing
                start = time.time()
                _ = model(input_tensor)
                end = time.time()
                times.append(end - start)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    return avg_time, std_time

def get_gpu_memory_info():
    """
    Get GPU memory information (works for both NVIDIA and AMD)
    
    Returns:
        dict: Memory information
    """
    memory_info = {
        'total_memory': 0,
        'allocated_memory': 0,
        'cached_memory': 0,
        'free_memory': 0
    }
    
    if torch.cuda.is_available():
        try:
            memory_info['total_memory'] = torch.cuda.get_device_properties(0).total_memory
            memory_info['allocated_memory'] = torch.cuda.memory_allocated(0)
            memory_info['cached_memory'] = torch.cuda.memory_reserved(0)
            memory_info['free_memory'] = memory_info['total_memory'] - memory_info['allocated_memory']
        except Exception as e:
            memory_info['error'] = str(e)
    
    return memory_info

def print_gpu_memory_info():
    """Print GPU memory information"""
    memory_info = get_gpu_memory_info()
    
    if 'error' in memory_info:
        print(f"Error getting GPU memory info: {memory_info['error']}")
        return
    
    if memory_info['total_memory'] > 0:
        print("GPU Memory Information:")
        print(f"  Total Memory: {memory_info['total_memory'] / 1024**3:.2f} GB")
        print(f"  Allocated: {memory_info['allocated_memory'] / 1024**3:.2f} GB")
        print(f"  Cached: {memory_info['cached_memory'] / 1024**3:.2f} GB")
        print(f"  Free: {memory_info['free_memory'] / 1024**3:.2f} GB")
    else:
        print("No GPU memory information available")

def validate_environment():
    """
    Validate the environment and dependencies
    
    Returns:
        validation_results: Dictionary of validation results
    """
    results = {}
    
    # Check PyTorch version
    results['pytorch_version'] = torch.__version__
    
    # Check CUDA availability (includes ROCm)
    results['cuda_available'] = torch.cuda.is_available()
    if results['cuda_available']:
        results['cuda_version'] = torch.version.cuda
        results['cuda_device_count'] = torch.cuda.device_count()
        results['cuda_device_name'] = torch.cuda.get_device_name(0)
        
        # Check if it's ROCm (AMD GPU)
        try:
            # ROCm typically shows up as CUDA but with different version info
            if 'rocm' in torch.version.cuda.lower() or 'hip' in torch.version.cuda.lower():
                results['is_rocm'] = True
                results['gpu_type'] = 'AMD ROCm'
            else:
                results['is_rocm'] = False
                results['gpu_type'] = 'NVIDIA CUDA'
        except:
            results['is_rocm'] = False
            results['gpu_type'] = 'Unknown GPU'
    else:
        results['is_rocm'] = False
        results['gpu_type'] = 'No GPU'
    
    # Check MPS availability (Apple Silicon)
    results['mps_available'] = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    
    # Check required directories
    required_dirs = ['./data', './weights', './data/images', './data/csv']
    results['required_directories'] = {}
    for dir_path in required_dirs:
        results['required_directories'][dir_path] = os.path.exists(dir_path)
    
    return results

def print_environment_info():
    """Print environment information"""
    print("=" * 80)
    print("ENVIRONMENT INFORMATION")
    print("=" * 80)
    
    validation_results = validate_environment()
    amd_support = check_amd_gpu_support()
    
    print(f"PyTorch version: {validation_results['pytorch_version']}")
    print(f"GPU available: {validation_results['cuda_available']}")
    print(f"GPU type: {validation_results['gpu_type']}")
    
    if validation_results['cuda_available']:
        print(f"CUDA/ROCm version: {validation_results['cuda_version']}")
        print(f"GPU devices: {validation_results['cuda_device_count']}")
        print(f"GPU device name: {validation_results['cuda_device_name']}")
        
        if validation_results.get('is_rocm', False):
            print("✓ AMD GPU with ROCm detected")
            print(f"  ROCm version: {amd_support.get('rocm_version', 'Unknown')}")
            print(f"  PyTorch ROCm support: {'✓' if amd_support.get('pytorch_rocm_support', False) else '✗'}")
        else:
            print("✓ NVIDIA GPU with CUDA detected")
    
    print(f"MPS available: {validation_results['mps_available']}")
    
    # AMD GPU specific information
    if amd_support.get('rocm_available', False):
        print("\nAMD GPU Support Status:")
        print(f"  ROCm available: {'✓' if amd_support['rocm_available'] else '✗'}")
        print(f"  AMD GPU detected: {'✓' if amd_support['amd_gpu_detected'] else '✗'}")
        print(f"  PyTorch ROCm support: {'✓' if amd_support['pytorch_rocm_support'] else '✗'}")
        if 'error' in amd_support:
            print(f"  Error: {amd_support['error']}")
    
    print("\nRequired directories:")
    for dir_path, exists in validation_results['required_directories'].items():
        status = "✓" if exists else "✗"
        print(f"  {status} {dir_path}")
    
    print("=" * 80)

if __name__ == "__main__":
    # Test utility functions
    print_environment_info()
    
    # Test seed setting
    set_seed(42)
    print("Seed set to 42")
    
    # Test directory creation
    dirs = create_directories()
    print(f"Created directories: {dirs}")
    
    print("Utility functions test completed!")
