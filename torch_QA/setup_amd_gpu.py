"""
AMD GPU Setup Script for PyTorch with ROCm
This script helps set up PyTorch with ROCm support for AMD GPUs
"""

import subprocess
import sys
import platform
import os

def check_amd_gpu():
    """Check if AMD GPU is available"""
    try:
        # Try to detect AMD GPU using lspci (Linux) or wmic (Windows)
        if platform.system() == "Linux":
            result = subprocess.run(['lspci'], capture_output=True, text=True)
            if 'AMD' in result.stdout or 'Radeon' in result.stdout:
                print("✓ AMD GPU detected")
                return True
        elif platform.system() == "Windows":
            result = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 'name'], 
                                  capture_output=True, text=True)
            if 'AMD' in result.stdout or 'Radeon' in result.stdout:
                print("✓ AMD GPU detected")
                return True
    except:
        pass
    
    print("⚠ AMD GPU not detected or detection failed")
    return False

def check_rocm_installation():
    """Check if ROCm is installed"""
    try:
        # Check for ROCm installation
        rocm_paths = [
            '/opt/rocm',
            '/usr/local/rocm',
            'C:\\Program Files\\AMD\\ROCm'
        ]
        
        for path in rocm_paths:
            if os.path.exists(path):
                print(f"✓ ROCm found at: {path}")
                return True
        
        print("⚠ ROCm not found")
        return False
    except:
        print("⚠ ROCm check failed")
        return False

def install_pytorch_rocm():
    """Install PyTorch with ROCm support"""
    print("Installing PyTorch with ROCm support...")
    
    # PyTorch installation command for ROCm
    install_cmd = [
        sys.executable, "-m", "pip", "install", 
        "torch", "torchvision", "torchaudio", 
        "--index-url", "https://download.pytorch.org/whl/rocm5.4.2"
    ]
    
    try:
        subprocess.run(install_cmd, check=True)
        print("✓ PyTorch with ROCm installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install PyTorch with ROCm: {e}")
        return False

def test_pytorch_rocm():
    """Test PyTorch ROCm installation"""
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.is_available()}")
            print(f"✓ CUDA version: {torch.version.cuda}")
            print(f"✓ Number of GPUs: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("⚠ CUDA not available")
            
        # Test ROCm specifically
        if hasattr(torch.version, 'hip'):
            print(f"✓ ROCm version: {torch.version.hip}")
        else:
            print("⚠ ROCm version not detected")
            
        return True
    except ImportError:
        print("✗ PyTorch not installed")
        return False

def setup_environment():
    """Set up environment variables for ROCm"""
    print("Setting up environment variables...")
    
    # ROCm environment variables
    env_vars = {
        'ROCM_PATH': '/opt/rocm',
        'HIP_PLATFORM': 'amd',
        'HSA_OVERRIDE_GFX_VERSION': '10.3.0',  # Adjust based on your GPU
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"  {key}={value}")

def main():
    """Main setup function"""
    print("AMD GPU Setup for PyTorch with ROCm")
    print("=" * 50)
    
    # Check system
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"Python version: {sys.version}")
    
    # Check AMD GPU
    if not check_amd_gpu():
        print("Please ensure you have an AMD GPU installed")
        return
    
    # Check ROCm
    if not check_rocm_installation():
        print("Please install ROCm first:")
        print("  Ubuntu: https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html")
        print("  Windows: https://rocm.docs.amd.com/en/latest/deploy/windows/quick_start.html")
        return
    
    # Set up environment
    setup_environment()
    
    # Install PyTorch with ROCm
    if install_pytorch_rocm():
        # Test installation
        test_pytorch_rocm()
        print("\n✓ Setup completed successfully!")
        print("You can now run the PyTorch anomaly detection model on AMD GPU")
    else:
        print("\n✗ Setup failed")
        print("Please check the error messages above and try again")

if __name__ == "__main__":
    main()
