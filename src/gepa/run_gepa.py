from typing import List, Dict
import json
from pathlib import Path
from .evolution_engine import GEPAEvolutionEngine
from evaluation.real_ojbench import OJBenchEvaluator
from prompts.base_prompt import BASE_PROMPT

def select_training_problems(evaluator: OJBenchEvaluator, count: int = 12) -> List[Dict]:
    """Select diverse problems for GEPA training from real OJBench problems"""
    
    # Get balanced selection from real OJBench problems
    selected = []
    
    # Initialize evaluator if needed
    if not hasattr(evaluator, 'problems') or not evaluator.problems:
        from pathlib import Path
        problem_dirs = [
            Path("OJBench_testdata/NOI"),
            Path("OJBench_testdata/ICPC")
        ]
        evaluator.initialize_problems(problem_dirs)
    
    # 4 easy, 5 medium, 3 hard - balanced across NOI and ICPC  
    for difficulty, target_count in [("easy", 4), ("medium", 5), ("hard", 3)]:
        # Get problems of this difficulty
        candidates = evaluator.get_problems_subset(
            difficulty=difficulty,
            language="cpp",
            limit=target_count * 2  # Get more candidates to choose from
        )
        
        if not candidates:
            print(f"⚠️  No {difficulty} problems found, using available problems")
            candidates = evaluator.get_problems_subset(limit=target_count)
        
        # Balance between NOI and ICPC if both datasets available
        noi_problems = [p for p in candidates if p.get('dataset') == 'NOI']
        icpc_problems = [p for p in candidates if p.get('dataset') == 'ICPC']
        
        if noi_problems and icpc_problems:
            # Take roughly half from each dataset
            half = target_count // 2
            selected.extend(noi_problems[:half])
            selected.extend(icpc_problems[:target_count - half])
        else:
            # Use whatever problems are available
            selected.extend(candidates[:target_count])
    
    print(f"✅ Selected {len(selected)} real OJBench training problems:")
    for p in selected:
        dataset = p.get('dataset', 'unknown')
        difficulty = p.get('difficulty', 'unknown')
        print(f"  - {p['id']}: {difficulty} ({dataset})")
    
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
    
    print(f"💾 GEPA evolution results saved to {output_path}")
    if best_candidate.fitness_score > 0:
        print(f"📈 Performance improvement: {((best_candidate.fitness_score - 0.179) / 0.179 * 100):+.1f}%")
    else:
        print(f"📊 Best fitness score: {best_candidate.fitness_score:.3f}")
    print(f"🧬 Evolved prompt ready for ART+RULER training")
    
    return best_candidate.text