# GEPA + LangGraph ART + RULER Integration

## 🎯 What This Is

Advanced AI system for competitive programming that combines:
- **GEPA**: Genetic prompt optimization  
- **LangGraph ART**: Reinforcement learning agents with OpenPipe integration
- **RULER**: Error analysis and reward shaping

## 🚀 Quick Start

### 1. Setup
```bash
# Install dependencies
pip install -r requirements.txt
pip install langgraph langchain openpipe

# For production (with GPU)
pip install torch transformers accelerate
```

### 2. Train System
```bash
python train_minimal.py
```

### 3. Evaluate
```bash
python main.py  # Evaluate on OJBench
```

## 📂 Key Components

```
GEPA_ART_RULER/
├── src/langgraph_art/          # LangGraph RL agents
├── src/training_data/          # External data loaders  
├── data/minimal_datasets/      # 100 training problems
├── train_minimal.py           # Main training script
└── main.py                    # Evaluation script
```

## 🎯 Training Data

- **Training**: 100 external problems (Codeforces + AtCoder)
- **Evaluation**: 232 OJBench problems (never seen in training)  
- **Proper separation**: No data leakage

## 📈 Expected Results

- **Baseline**: 17.9% (Qwen3-4B on OJBench)
- **Target**: 35%+ with GEPA+RL+RULER enhancement

## 🔧 System Features

✅ LangGraph agents with tool integration
✅ OpenPipe RL training support  
✅ RULER error analysis and reward shaping
✅ Proper train/test data separation
✅ Minimal dataset for manageable training
✅ Production-ready architecture

## 🧪 Research Innovation

First system to combine genetic prompt optimization with reinforcement learning for competitive programming, enhanced by error-guided reward shaping.
