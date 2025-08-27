#!/usr/bin/env python3
"""
GEPA+ART+RULER Setup Script
============================

Unified setup for the complete system.
Handles both development and production environments.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(cmd, check=True, shell=False):
    """Run a command and return the result"""
    print(f"Running: {cmd}")
    try:
        if isinstance(cmd, str) and not shell:
            cmd = cmd.split()
        result = subprocess.run(cmd, check=check, capture_output=True, text=True, shell=shell)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return e

def check_python_version():
    """Ensure Python 3.10+ is being used"""
    if sys.version_info < (3, 10):
        print("Error: Python 3.10 or higher is required")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    print(f"✅ Python version: {sys.version.split()[0]}")

def setup_virtual_environment():
    """Create and activate virtual environment"""
    venv_path = Path("venv-310")
    
    if not venv_path.exists():
        print("🔧 Creating virtual environment...")
        run_command([sys.executable, "-m", "venv", "venv-310"])
    else:
        print("✅ Virtual environment already exists")
    
    # Get the python executable path
    if platform.system() == "Windows":
        python_exe = venv_path / "Scripts" / "python.exe"
        pip_exe = venv_path / "Scripts" / "pip.exe"
    else:
        python_exe = venv_path / "bin" / "python"
        pip_exe = venv_path / "bin" / "pip"
    
    return python_exe, pip_exe

def install_base_requirements(pip_exe):
    """Install base Python requirements"""
    print("📦 Installing base requirements...")
    
    # Upgrade pip first
    run_command([str(pip_exe), "install", "--upgrade", "pip"])
    
    # Install requirements
    run_command([str(pip_exe), "install", "-r", "requirements.txt"])

def install_local_packages(pip_exe):
    """Install local packages from all_dependencies/"""
    print("🔧 Installing local packages...")
    
    packages = [
        ("all_dependencies/gepa", "GEPA prompt optimization"),
        ("all_dependencies/OJBench", "OJBench evaluation")
    ]
    
    for package_dir, description in packages:
        if Path(package_dir).exists():
            print(f"Installing {description}...")
            try:
                result = run_command([str(pip_exe), "install", "-e", package_dir], check=False)
                if result.returncode == 0:
                    print(f"✅ {description} installed")
                else:
                    print(f"⚠️ {description} failed to install: {result.stderr}")
            except Exception as e:
                print(f"⚠️ {description} installation error: {e}")
        else:
            print(f"⚠️ {package_dir} not found - skipping {description}")
    
    # Special handling for judge-server (Linux only)
    judge_server_dir = Path("all_dependencies/judge-server")
    if judge_server_dir.exists():
        if platform.system() == "Linux":
            print("Installing DMOJ judge server (Linux)...")
            try:
                result = run_command([str(pip_exe), "install", "-e", str(judge_server_dir)], check=False)
                if result.returncode == 0:
                    print("✅ DMOJ judge server installed")
                else:
                    print(f"⚠️ DMOJ judge server failed: {result.stderr}")
            except Exception as e:
                print(f"⚠️ DMOJ judge server error: {e}")
        else:
            print("⚠️ DMOJ judge server skipped (requires Linux)")
    else:
        print("⚠️ all_dependencies/judge-server not found")

def setup_gpu_environment(pip_exe):
    """Setup GPU environment if CUDA is available"""
    print("🔍 Checking for GPU environment...")
    
    try:
        # Check if CUDA is available
        result = run_command(["nvidia-smi"], check=False)
        if result.returncode == 0:
            print("✅ NVIDIA GPU detected")
            print("🚀 Installing PyTorch with CUDA support...")
            
            # Install PyTorch with CUDA
            cuda_cmd = [
                str(pip_exe), "install", 
                "torch", "torchvision", "torchaudio",
                "--index-url", "https://download.pytorch.org/whl/cu121"
            ]
            result = run_command(cuda_cmd, check=False)
            
            if result.returncode == 0:
                print("✅ PyTorch with CUDA installed")
                return True
            else:
                print(f"⚠️ CUDA PyTorch installation failed: {result.stderr}")
        else:
            print("ℹ️ No NVIDIA GPU detected, using CPU-only PyTorch")
    
    except Exception as e:
        print(f"ℹ️ GPU check failed: {e}")
    
    return False

def test_installation(python_exe):
    """Test the installation"""
    print("🧪 Testing installation...")
    
    test_script = """
import sys
sys.path.append('src')

print('Testing core imports...')
success = True

try:
    import numpy as np
    print('✅ NumPy imported')
except ImportError as e:
    print(f'❌ NumPy failed: {e}')
    success = False

try:
    import gepa
    print('✅ GEPA imported')
except ImportError as e:
    print(f'⚠️ GEPA not available: {e}')

try:
    import ojbench
    print('✅ OJBench imported')
except ImportError as e:
    print(f'⚠️ OJBench not available: {e}')

try:
    from utils.code_parser import CodeParser
    print('✅ Code parser imported')
except ImportError as e:
    print(f'❌ Code parser failed: {e}')
    success = False

try:
    from ruler.ruler_analyzer import RULERAnalyzer
    print('✅ RULER imported')
except ImportError as e:
    print(f'❌ RULER failed: {e}')
    success = False

if success:
    print('🎉 Core installation successful!')
else:
    print('⚠️ Some components missing but basic system should work')
"""
    
    try:
        result = run_command([str(python_exe), "-c", test_script])
        if result.returncode == 0:
            print("✅ Installation test passed")
            return True
        else:
            print("⚠️ Installation test had issues")
            return False
    except Exception as e:
        print(f"❌ Installation test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🌟 GEPA+ART+RULER Setup")
    print("=" * 50)
    
    # Check Python version
    check_python_version()
    
    # Setup virtual environment
    python_exe, pip_exe = setup_virtual_environment()
    
    # Install base requirements
    install_base_requirements(pip_exe)
    
    # Install local packages
    install_local_packages(pip_exe)
    
    # Setup GPU environment if available
    gpu_available = setup_gpu_environment(pip_exe)
    
    # Test installation
    test_passed = test_installation(python_exe)
    
    # Final status
    print("\n" + "=" * 50)
    print("🏁 Setup Complete!")
    print(f"✅ Environment: {python_exe}")
    print(f"✅ GPU Support: {'Yes' if gpu_available else 'No (CPU only)'}")
    print(f"✅ Tests: {'Passed' if test_passed else 'Had issues'}")
    
    print("\n📋 Next Steps:")
    print("1. Run the minimal skeleton:")
    print(f"   {python_exe} minimal_working_skeleton.py")
    print("\n2. Run the unified pipeline:")
    print(f"   {python_exe} unified_pipeline.py")
    
    if not gpu_available:
        print("\n💡 For GPU training:")
        print("   - Ensure NVIDIA GPU with CUDA drivers installed")
        print("   - Re-run setup.py for CUDA PyTorch installation")
    
    print("\n📖 See docs/COMPLETE_IMPLEMENTATION_GUIDE.md for full instructions")

if __name__ == "__main__":
    main()