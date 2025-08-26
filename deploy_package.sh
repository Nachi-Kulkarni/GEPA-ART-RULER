#!/bin/bash
# GEPA+ART+RULER Deployment Package Creator
# Creates a portable package for GPU deployment

echo "🚀 Creating GEPA+ART+RULER deployment package..."

# Create deployment directory
mkdir -p deploy_package
cd deploy_package

# Copy essential files (excluding heavy/unnecessary items)
echo "📦 Copying core files..."

# Core system files
cp -r ../src .
cp -r ../gepa .
cp -r ../OJBench .
cp -r ../docs .

# Configuration files
cp ../main.py .
cp ../test_setup.py .
cp ../requirements.txt .
cp ../README.md .
cp ../CLAUDE.md .

# Data files (minimal datasets only)
mkdir -p data
cp -r ../data/minimal_datasets data/ 2>/dev/null || echo "No minimal datasets found"

echo "📋 Creating setup script..."

# Create GPU setup script
cat > gpu_setup.sh << 'EOF'
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

EOF

chmod +x gpu_setup.sh

echo "📄 Creating quick start guide..."

# Create quick start guide
cat > GPU_QUICKSTART.md << 'EOF'
# GPU Quick Start Guide

## 1. Transfer this package to your GPU instance
```bash
# Option A: If connection works later
scp -P 33563 -r deploy_package root@1.208.108.242:~/GEPA_ART_RULER/

# Option B: Upload via web interface/other method
# Upload deploy_package folder to GPU instance

# Option C: Create archive and transfer
tar -czf gepa_deploy.tar.gz deploy_package
# Then upload gepa_deploy.tar.gz and extract on GPU
```

## 2. On GPU Instance - Run Setup
```bash
cd GEPA_ART_RULER  # or wherever you placed the files
chmod +x gpu_setup.sh
./gpu_setup.sh
```

## 3. Run the System
```bash
source venv-gpu/bin/activate

# Quick test (3 problems)
python main.py

# Full system with GEPA optimization
python main.py --full-gepa --problems-limit 20 --difficulty medium

# Monitor GPU usage
watch nvidia-smi
```

## Expected Performance
- Model loads in 2-5 minutes
- GEPA optimization: 10-30 minutes  
- Per problem: 30-60 seconds
- Target: 25-50% success rate vs 17.9% baseline
EOF

# Create archive
cd ..
echo "📦 Creating deployment archive..."
tar -czf gepa_art_ruler_deploy.tar.gz deploy_package

echo "✅ Deployment package created!"
echo ""
echo "📁 Files created:"
echo "   - deploy_package/           # Ready-to-deploy folder"
echo "   - gepa_art_ruler_deploy.tar.gz  # Compressed archive"
echo ""
echo "🚀 Next steps:"
echo "   1. Upload deploy_package/ or gepa_art_ruler_deploy.tar.gz to your GPU instance"
echo "   2. Run ./gpu_setup.sh on the GPU instance"
echo "   3. Start with: python main.py"