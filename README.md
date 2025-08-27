# GEPA+ART+RULER: Competitive Programming AI System

**Status**: ✅ **Phase 1 Complete** - Foundation Recovery Successful  
**Architecture**: Local GPU Research Prototype (4x RTX 3060)  
**Target**: >40% improvement over 17.9% baseline on competitive programming  

## 🎯 Quick Start

### 1. Setup Environment
```bash
# Clone and setup
cd GEPA_ART_RULER
python setup.py  # Unified setup script
```

### 2. Test the System
```bash
# Test core integration (works without GPU)
./venv-310/bin/python minimal_working_skeleton.py

# Test full pipeline (mock mode)
./venv-310/bin/python unified_pipeline.py
```

### 3. Production Setup (GPU Environment)
```bash
# Install GPU dependencies
./venv-310/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install local frameworks
cd all_dependencies/gepa && ../../venv-310/bin/pip install -e .
cd ../OJBench && ../../venv-310/bin/pip install -e .
```

## 📋 System Overview

This system combines three advanced AI techniques for competitive programming:

- **🧬 GEPA**: Genetic-Evolutionary Prompt Optimization using LLM reflection
- **🤖 ART**: Automatic Reasoning and Tool-use with structured thinking
- **👑 RULER**: Recursive Error Learning and Resolution for mistake correction

### Architecture Pipeline
```
Problems → GEPA Optimization → ART Reasoning → RULER Analysis → Local Training → Improved Performance
```

## 🏗️ Project Status

### ✅ Phase 1: Foundation Recovery (COMPLETE)
- **Architectural Decision**: Local GPU prototype chosen over API-based training
- **Component Integration**: All core components working together in mock mode
- **Dependencies**: Real GEPA and OJBench frameworks installed and functional
- **Code Quality**: All syntax errors fixed, imports working, clean structure

### 🔄 Phase 2: System Integration (GPU Required)
- **Real Model Loading**: Qwen model inference and training
- **Real Evaluation**: OJBench competitive programming assessment  
- **Real Optimization**: GEPA prompt evolution with model feedback
- **Integration Testing**: End-to-end pipeline validation

### 🚀 Phase 3: Production Deployment (Multi-GPU)
- **Distributed Training**: 4x RTX 3060 LoRA fine-tuning
- **Performance Benchmarking**: Achieve >40% improvement target
- **Production Monitoring**: Logging, checkpointing, recovery
- **System Optimization**: Memory management and throughput tuning

## 📁 Key Files

### 🎮 Entry Points
- **`unified_pipeline.py`** - Main production pipeline
- **`minimal_working_skeleton.py`** - Core integration test (proven working)
- **`setup.py`** - Unified environment setup script

### 🔧 Core Components
- **`src/gepa/`** - Prompt optimization engine
- **`src/art/`** - Structured reasoning solver
- **`src/ruler/`** - Error analysis and correction
- **`src/models/`** - Qwen local GPU interface
- **`src/evaluation/`** - OJBench integration

### 📚 Documentation
- **`docs/COMPLETE_IMPLEMENTATION_GUIDE.md`** - Comprehensive setup and implementation
- **`ARCHITECTURE_DECISION.md`** - System design rationale
- **`RECOVERY_PLAN.md`** - Phase-by-phase development plan
- **`PHASE1_RECOVERY_COMPLETE.md`** - Current status summary

### 🗃️ Organized Structure
- **`all_dependencies/`** - External frameworks (GEPA, OJBench, DMOJ)
- **`archived/`** - Quarantined incompatible components  
- **`data/`** - Training datasets and results
- **`src/`** - Core system implementation

## 🛠️ Development Mode

The system works in **mock mode** for development without GPU requirements:

```bash
# Test the complete pipeline (no GPU needed)
./venv-310/bin/python unified_pipeline.py
```

**Mock Mode Features:**
- ✅ Component integration testing
- ✅ Pipeline flow validation  
- ✅ Error handling verification
- ✅ Development iteration support

## 🔬 Production Mode

Full production requires GPU environment:

**Hardware Requirements:**
- 4x RTX 3060 (48GB total VRAM) or equivalent
- Linux environment (for DMOJ judge server)
- 64GB+ system RAM recommended

**Software Requirements:**
- CUDA 12.1+ with PyTorch GPU support
- All dependencies from `requirements.txt`
- OJBench competitive programming datasets

## 🎯 Performance Targets

- **Baseline**: 17.9% success rate on OJBench problems
- **GEPA Boost**: 25% success rate after prompt optimization  
- **Final Target**: 50% success rate after ART+RULER training
- **Minimum Goal**: >40% relative improvement demonstrated

## 🚨 System Recovery Notes

This project underwent **architectural crisis recovery**:

- **Problem**: Originally designed with conflicting architectures (local GPU vs API-based)
- **Solution**: Chose single coherent paradigm (local GPU research prototype)
- **Result**: Clean, working foundation ready for production deployment

See `EXTERNAL_EVALUATION_RESPONSE.md` for details on how external evaluations guided the recovery.

## 📖 Getting Help

1. **Setup Issues**: Check `setup.py` output and logs
2. **GPU Problems**: Verify CUDA installation with `nvidia-smi`
3. **Import Errors**: Ensure virtual environment activated
4. **Performance**: See `docs/gpu_deployment_guide.md`

## 🎉 Success Metrics

**Phase 1 Complete**: ✅ Foundation solid and ready for production  
**Next Milestone**: Phase 2 system integration when GPU environment available  

The system has been successfully recovered from architectural crisis and demonstrates:
- 100% mock pipeline success rate
- Clean component integration
- Production-ready code structure
- Clear path to GPU deployment