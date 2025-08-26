# GPU Deployment Guide

## Quick Transfer & Setup Commands

### 1. Transfer Files to GPU Instance

```bash
# From your local machine, sync all files to GPU instance
rsync -avz --progress \
  --exclude='venv-310/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='.git/' \
  --exclude='data/cache/' \
  -e "ssh -p 33563" \
  . root@1.208.108.242:~/GEPA_ART_RULER/
```

### 2. Connect to GPU Instance

```bash
ssh -p 33563 root@1.208.108.242 -L 8080:localhost:8080
```

### 3. GPU Environment Setup

```bash
cd ~/GEPA_ART_RULER

# Check GPU status
nvidia-smi

# Create fresh Python environment
python3.10 -m venv venv-gpu
source venv-gpu/bin/activate

# Install CUDA PyTorch (adjust CUDA version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core ML dependencies
pip install transformers>=4.36.0 accelerate datasets

# Install system dependencies
pip install -r requirements.txt

# Install DMOJ judge system
git clone https://github.com/DMOJ/judge-server.git
cd judge-server
git checkout f098cd3a49a60186d1fadde5132329ec5f4f2213
pip install -e .
cd ..

# Install local packages
pip install -e OJBench
cd gepa && pip install -e . && cd ..

# Download test data (if not already present)
git lfs install
git clone https://huggingface.co/datasets/He-Ren/OJBench_testdata
```

### 4. Verify GPU Setup

```bash
# Test GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# Test model interface
python -c "from src.models.qwen_interface import Qwen3Interface; model = Qwen3Interface(); print('Model loaded successfully')"

# Run setup test
python test_setup.py
```

### 5. Production Runs

```bash
# Quick test (3 problems)
python main.py

# Full GEPA optimization with medium problems
python main.py --full-gepa --problems-limit 20 --difficulty medium

# Production run with full evaluation
python main.py --full-gepa --problems-limit 50 --output-dir results/gpu_run_$(date +%Y%m%d_%H%M%S)
```

## Transfer Options

### Option A: rsync (Recommended - Incremental)
```bash
# Initial full sync
rsync -avz --progress \
  --exclude='venv-310/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='.git/' \
  -e "ssh -p 33563" \
  . root@1.208.108.242:~/GEPA_ART_RULER/

# Later incremental updates
rsync -avz --progress \
  --exclude='venv-310/' \
  --exclude='data/cache/' \
  -e "ssh -p 33563" \
  . root@1.208.108.242:~/GEPA_ART_RULER/
```

### Option B: tar + scp (Full Archive)
```bash
# Create archive (excluding unnecessary files)
tar -czf gepa_art_ruler.tar.gz \
  --exclude='venv-310' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.git' \
  --exclude='data/cache' \
  .

# Transfer
scp -P 33563 gepa_art_ruler.tar.gz root@1.208.108.242:~/

# Extract on GPU instance
ssh -p 33563 root@1.208.108.242
cd ~
tar -xzf gepa_art_ruler.tar.gz
mv * GEPA_ART_RULER/ 2>/dev/null || true
```

### Option C: Git Clone (If repo is public/accessible)
```bash
ssh -p 33563 root@1.208.108.242
git clone <your-repo-url> GEPA_ART_RULER
cd GEPA_ART_RULER
```

## GPU-Specific Optimizations

### Memory Management
```bash
# Monitor GPU memory during runs
watch -n 1 nvidia-smi

# If memory issues, reduce batch sizes in model config
export CUDA_VISIBLE_DEVICES=0  # Use specific GPU
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # Reduce fragmentation
```

### Model Configuration Updates
```python
# In src/models/qwen_interface.py, ensure GPU optimization:
device_map = "auto"  # Let transformers handle GPU placement
torch_dtype = torch.float16  # Use FP16 for memory efficiency
```

## Troubleshooting

### CUDA Issues
```bash
# Check CUDA version
nvcc --version

# Install correct PyTorch version
# For CUDA 11.8: --index-url https://download.pytorch.org/whl/cu118
# For CUDA 12.1: --index-url https://download.pytorch.org/whl/cu121
```

### DMOJ Judge Issues (Linux-specific)
```bash
# Install system dependencies
apt-get update
apt-get install -y build-essential python3-dev libseccomp-dev

# Verify judge works
cd judge-server
python -m dmoj.judger --help
```

### Model Loading Issues
```bash
# If model too large for GPU memory
export TRANSFORMERS_CACHE=/tmp/  # Use faster temp storage
export HF_HOME=/tmp/huggingface/  # Cache location
```

## Performance Monitoring

### During Runs
```bash
# Terminal 1: Run system
python main.py --full-gepa --problems-limit 20

# Terminal 2: Monitor resources  
watch -n 2 'nvidia-smi && echo "=== CPU/MEM ===" && htop -n 1'
```

### Expected Performance
- **Model Loading**: 2-5 minutes first time
- **GEPA Optimization**: 10-30 minutes (4 generations)
- **Per Problem**: 30-60 seconds average
- **GPU Memory**: 8-12GB for Qwen3-4B
- **Target Success Rate**: 25-50% vs 17.9% baseline

## Quick Commands Summary

```bash
# 1. Transfer files
rsync -avz --exclude='venv-310/' -e "ssh -p 33563" . root@1.208.108.242:~/GEPA_ART_RULER/

# 2. SSH to GPU
ssh -p 33563 root@1.208.108.242 -L 8080:localhost:8080

# 3. Setup environment
cd ~/GEPA_ART_RULER && python3.10 -m venv venv-gpu && source venv-gpu/bin/activate

# 4. Install GPU dependencies
pip install torch transformers accelerate -f https://download.pytorch.org/whl/cu121/torch_stable.html

# 5. Test system
python test_setup.py && python main.py

# 6. Production run
python main.py --full-gepa --problems-limit 20 --output-dir results/production_run
```