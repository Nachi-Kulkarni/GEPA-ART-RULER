from typing import List, Dict
import json
from pathlib import Path
from .evolution_engine import GEPAEvolutionEngine
try:
    from ..evaluation.ojbench_interface import OJBenchEvaluator
except ImportError:
    from ..evaluation.mock_ojbench import MockOJBenchEvaluator as OJBenchEvaluator
from ..prompts.base_prompt import BASE_PROMPT

def select_training_problems(evaluator: OJBenchEvaluator, count: int = 12) -> List[Dict]:
    """Select diverse problems for GEPA training"""
    
    # For mock evaluator, create sample problems
    if hasattr(evaluator, 'is_mock'):
        print(f"🎭 Using mock training problems for GEPA")
        return [
            {"id": "gepa_train_1", "prompt": "Given n, output n*2", "difficulty": "easy", "dataset": "mock"},
            {"id": "gepa_train_2", "prompt": "Find sum of array", "difficulty": "easy", "dataset": "mock"},
            {"id": "gepa_train_3", "prompt": "Count vowels in string", "difficulty": "medium", "dataset": "mock"},
            {"id": "gepa_train_4", "prompt": "Binary search implementation", "difficulty": "medium", "dataset": "mock"},
            {"id": "gepa_train_5", "prompt": "Longest common subsequence", "difficulty": "hard", "dataset": "mock"},
        ]
    
    # Get balanced selection across difficulties and datasets
    selected = []
    
    # 4 easy, 5 medium, 3 hard - balanced across NOI and ICPC
    for difficulty, target_count in [("easy", 4), ("medium", 5), ("hard", 3)]:
        # Get problems of this difficulty
        candidates = evaluator.get_problems_subset(
            difficulty=difficulty, 
            language="cpp"
        )
        
        # Balance between NOI and ICPC
        noi_problems = [p for p in candidates if p['dataset'] == 'NOI']
        icpc_problems = [p for p in candidates if p['dataset'] == 'ICPC']
        
        # Take roughly half from each dataset
        half = target_count // 2
        selected.extend(noi_problems[:half])
        selected.extend(icpc_problems[:target_count - half])
    
    print(f"Selected {len(selected)} training problems:")
    for p in selected:
        print(f"  - {p['id']}: {p['difficulty']} ({p['dataset']})")
    
    return selected

def run_gepa_optimization(model_interface, output_dir: str = "data/gepa_results") -> str:
    """Run the complete GEPA optimization process"""
    
    # Setup
    evaluator = OJBenchEvaluator()
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Select training problems
    training_problems = select_training_problems(evaluator)
    
    # Initialize GEPA engine
    gepa = GEPAEvolutionEngine(
        model_interface=model_interface,
        evaluator=evaluator,
        population_size=6  # Small population due to cost constraints
    )
    
    # Run evolution
    best_candidate = gepa.run_evolution(
        base_prompt=BASE_PROMPT,
        test_problems=training_problems,
        max_generations=4
    )
    
    # Save results
    results = {
        "best_prompt": best_candidate.text,
        "best_score": best_candidate.fitness_score,
        "evolution_log": gepa.evolution_log,
        "training_problems": [p["id"] for p in training_problems]
    }
    
    with open(output_path / "evolution_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    with open(output_path / "best_prompt.txt", "w") as f:
        f.write(best_candidate.text)
    
    print(f"💾 Results saved to {output_path}")
    print(f"📈 Improvement: {((best_candidate.fitness_score - 0.179) / 0.179 * 100):+.1f}%")
    
    return best_candidate.text