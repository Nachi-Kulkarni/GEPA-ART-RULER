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
