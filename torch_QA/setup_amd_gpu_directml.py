"""
AMD GPU Setup Script for Windows with DirectML
Windows용 AMD GPU 설정 스크립트 (DirectML 사용)
"""

import subprocess
import sys
import platform
import os

def check_amd_gpu():
    """Check if AMD GPU is available"""
    try:
        result = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 'name'], 
                              capture_output=True, text=True)
        if 'AMD' in result.stdout or 'Radeon' in result.stdout:
            print("✓ AMD GPU detected")
            return True
    except:
        pass
    
    print("⚠ AMD GPU not detected or detection failed")
    return False

def install_torch_directml():
    """Install PyTorch with DirectML support"""
    print("Installing PyTorch with DirectML support...")
    
    # PyTorch with DirectML installation
    install_cmd = [
        sys.executable, "-m", "pip", "install", 
        "torch", "torchvision", "torchaudio", 
        "torch-directml"
    ]
    
    try:
        subprocess.run(install_cmd, check=True)
        print("✓ PyTorch with DirectML installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install PyTorch with DirectML: {e}")
        return False

def test_torch_directml():
    """Test PyTorch DirectML installation"""
    try:
        import torch
        import torch_directml
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ torch-directml version: {torch_directml.__version__}")
        
        # Check DirectML device
        if torch_directml.is_available():
            device = torch_directml.device()
            print(f"✓ DirectML device available: {device}")
            
            # Test tensor creation
            test_tensor = torch.randn(2, 3).to(device)
            print(f"✓ Test tensor created on DirectML device: {test_tensor.device}")
            
            return True
        else:
            print("⚠ DirectML not available")
            return False
            
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def main():
    """Main setup function"""
    print("AMD GPU Setup for Windows with DirectML")
    print("=" * 50)
    
    # Check system
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"Python version: {sys.version}")
    
    # Check AMD GPU
    if not check_amd_gpu():
        print("Please ensure you have an AMD GPU installed")
        return
    
    # Install PyTorch with DirectML
    if install_torch_directml():
        # Test installation
        if test_torch_directml():
            print("\n✓ Setup completed successfully!")
            print("You can now run the PyTorch anomaly detection model on AMD GPU using DirectML")
        else:
            print("\n⚠ Setup completed but DirectML test failed")
            print("Please check your AMD GPU drivers")
    else:
        print("\n✗ Setup failed")
        print("Please check the error messages above and try again")

if __name__ == "__main__":
    main()
