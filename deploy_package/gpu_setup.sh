#!/bin/bash
set -e

echo "🎯 GEPA+ART+RULER GPU Setup Starting..."

# Check GPU
echo "🔍 Checking GPU availability..."
nvidia-smi || { echo "❌ No GPU found!"; exit 1; }

# Create Python environment
echo "🐍 Setting up Python environment..."
python3.10 -m venv venv-gpu
source venv-gpu/bin/activate

# Install CUDA PyTorch
echo "🔥 Installing PyTorch with CUDA..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install ML dependencies
echo "🤖 Installing ML dependencies..."
pip install transformers>=4.36.0 accelerate datasets

# Install requirements
echo "📦 Installing requirements..."
pip install -r requirements.txt

# Install DMOJ judge system
echo "⚖️  Setting up DMOJ judge..."
if [ ! -d "judge-server" ]; then
    git clone https://github.com/DMOJ/judge-server.git
    cd judge-server
    git checkout f098cd3a49a60186d1fadde5132329ec5f4f2213
    pip install -e .
    cd ..
fi

# Install local packages
echo "🏗️  Installing local packages..."
pip install -e OJBench
cd gepa && pip install -e . && cd ..

# Download test data if needed
echo "📚 Getting test data..."
if [ ! -d "OJBench_testdata" ]; then
    git lfs install
    git clone https://huggingface.co/datasets/He-Ren/OJBench_testdata
fi

# Test setup
echo "🧪 Testing setup..."
python test_setup.py

echo "✅ Setup complete! Ready to run:"
echo "   python main.py                    # Quick test"
echo "   python main.py --full-gepa       # Full system"
echo "   source venv-gpu/bin/activate     # Activate environment"

