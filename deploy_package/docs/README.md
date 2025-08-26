# GEPA + LangGraph ART + RULER Integration

## 🎯 Project Overview

Advanced competitive programming AI system combining:
- **GEPA**: Genetic prompt optimization
- **LangGraph ART**: Reinforcement learning agents with tool use
- **RULER**: Error analysis and reward shaping

## 📊 Training Data

### External Training Sources (1500+ problems)
- **Codeforces**: 500 problems (GEPA optimization)
- **AtCoder**: 500 problems (RL training)  
- **USACO**: 500 problems (RL training)

### Evaluation Data
- **OJBench**: 232 problems (NOI + ICPC)
- **Never used in training** - ensures valid evaluation

## 🚀 Quick Start

### 1. Install Dependencies
```bash
# Core dependencies
pip install -r requirements.txt

# LangGraph + OpenPipe
pip install langgraph langchain langchain-community openpipe

# PyTorch (for production)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate datasets
```

### 2. Load Training Data
```python
from training_data import load_external_training_data

# Load complete training dataset
dataset = load_external_training_data(gepa_size=500, rl_size=1000)
```

### 3. Run Training Pipeline
```python
from langgraph_art import LangGraphARTAgent, create_rl_trainer
from langgraph_art.ruler_reward_shaper import create_ruler_reward_shaper

# Initialize components
agent = LangGraphARTAgent(enable_openpipe=True)
trainer = create_rl_trainer(agent)
ruler_shaper = create_ruler_reward_shaper()

# Run training iteration
await trainer.train_iteration(
    train_problems=dataset['rl_train'],
    val_problems=dataset['rl_val'],
    optimized_prompt=gepa_optimized_prompt
)
```

## 📂 Project Structure

```
GEPA_ART_RULER/
├── src/
│   ├── langgraph_art/          # LangGraph RL agents
│   ├── training_data/          # External data loaders
│   ├── gepa/                   # GEPA optimization
│   ├── ruler/                  # RULER error analysis
│   ├── art/                    # Original ART solver
│   ├── models/                 # Model interfaces
│   └── evaluation/             # OJBench integration
├── data/
│   ├── final_datasets/         # Training splits
│   ├── cache/                  # Cached results
│   └── results/               # Evaluation results
├── OJBench/                   # Evaluation benchmark
├── gepa/                      # GEPA framework
└── docs/                      # Documentation
```

## 🎯 System Workflow

1. **GEPA Phase**: Optimize prompts on Codeforces problems
2. **RL Training**: Train agents on AtCoder/USACO problems
3. **RULER Enhancement**: Shape rewards with error analysis
4. **Evaluation**: Test on unseen OJBench problems

## 📈 Expected Performance

- **Baseline**: 17.9% (Qwen3-4B on OJBench)
- **Target**: 35-60% (with GEPA+RL+RULER)
- **Continuous Improvement**: Performance grows with training

## 🔧 Production Deployment

1. Enable GPU training with real models
2. Scale to full datasets (1500+ training problems)
3. Enable OpenPipe cloud RL training
4. Run comprehensive evaluation on OJBench

## 🧪 Key Features

- **Data Separation**: Proper train/test splits
- **Multi-Modal Learning**: Genetic + RL optimization
- **Error-Guided Learning**: RULER reward shaping
- **Tool Integration**: OJBench evaluation tools
- **Scalable Architecture**: Production-ready components

## 📋 Status

✅ **Core Implementation**: Complete
✅ **Training Data**: 1500+ problems collected
✅ **LangGraph Integration**: Functional
✅ **RULER Enhancement**: Implemented
🔄 **Production Training**: Ready to deploy
