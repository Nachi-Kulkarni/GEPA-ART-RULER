# Training Guide

## 🔄 Complete Training Pipeline

### Step 1: Data Loading
```python
# Load external training data (NOT OJBench)
dataset = load_external_training_data(gepa_size=500, rl_size=1000)
```

### Step 2: GEPA Optimization
```python
# Run GEPA on Codeforces problems
from gepa import run_gepa_optimization
optimized_prompt = run_gepa_optimization(dataset['gepa_train'])
```

### Step 3: RL Agent Training
```python
# Train LangGraph agent with RL
agent = LangGraphARTAgent(enable_openpipe=True)
trainer = create_rl_trainer(agent)
results = await trainer.train_iteration(
    train_problems=dataset['rl_train'],
    val_problems=dataset['rl_val'],
    optimized_prompt=optimized_prompt
)
```

### Step 4: RULER Enhancement
```python
# Enhance rewards with error analysis
ruler_shaper = create_ruler_reward_shaper()
enhanced_trajectories = ruler_shaper.batch_enhance_trajectories(trajectories)
```

### Step 5: Final Evaluation
```python
# Evaluate on OJBench (never seen in training)
from evaluation.ojbench_interface import OJBenchEvaluator
evaluator = OJBenchEvaluator()
final_results = evaluator.evaluate_on_full_benchmark()
```

## 📊 Expected Timeline

- **Data Loading**: 30 minutes
- **GEPA Optimization**: 2-4 hours  
- **RL Training**: 8-12 hours
- **Final Evaluation**: 1-2 hours
- **Total**: ~16 hours for complete pipeline
