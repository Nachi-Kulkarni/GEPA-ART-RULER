"""
GEPA Runner with OpenRouter API Integration
Main orchestration script for competitive programming optimization
"""
import sys
import os
from pathlib import Path
from typing import List, Dict, Any

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import GEPA and OpenRouter components
from gepa.api import optimize
from models.openrouter_interface import create_litellm_function
from config.openrouter_config import OpenRouterConfig, setup_openrouter_env
from evaluation.real_ojbench import OJBenchEvaluator
from utils.code_parser import extract_python_code

class OpenRouterGEPARunner:
    """Main class for running GEPA optimization with OpenRouter models"""
    
    def __init__(self):
        # Verify environment setup
        if not setup_openrouter_env():
            raise Exception("OpenRouter environment not properly configured")
            
        # Initialize components
        self.config = OpenRouterConfig()
        self.evaluator = OJBenchEvaluator()
        
        # Create model functions for GEPA
        self.task_lm_func = create_litellm_function(self.config.TASK_MODEL)
        self.reflection_lm_func = create_litellm_function(self.config.REFLECTION_MODEL)
        
        print(f"✅ OpenRouter GEPA Runner initialized")
        print(f"   Task Model: {self.config.TASK_MODEL}")
        print(f"   Reflection Model: {self.config.REFLECTION_MODEL}")
    
    def load_ojbench_problems(self, difficulty: str = "easy", limit: int = 20) -> List[Dict[str, Any]]:
        """Load OJBench problems for optimization"""
        from ojbench_testdata.prompts import full
        
        # Load problems from JSONL
        problems_path = Path(__file__).parent.parent.parent / "OJBench_testdata" / "prompts" / "full.jsonl"
        problems = []
        
        with open(problems_path, 'r') as f:
            for line in f:
                import json
                problem = json.loads(line.strip())
                if problem.get("difficulty") == difficulty and len(problems) < limit:
                    problems.append({
                        "problem_id": problem["id"],
                        "prompt": problem["prompt"],
                        "dataset": problem["dataset"],
                        "difficulty": problem["difficulty"],
                        "language": "python"
                    })
        
        print(f"✅ Loaded {len(problems)} {difficulty} problems from OJBench")
        return problems
    
    def create_evaluation_metric(self):
        """Create evaluation metric for GEPA based on OJBench success rate"""
        
        def ojbench_metric(example, prediction, trace=None):
            """
            Evaluation metric for competitive programming
            Returns 1.0 if solution is accepted (AC), 0.0 otherwise
            """
            try:
                # Extract code from model response
                if hasattr(prediction, 'solution_code'):
                    code = prediction.solution_code
                elif hasattr(prediction, 'response'):
                    code = extract_python_code(prediction.response)
                else:
                    code = extract_python_code(str(prediction))
                
                if not code:
                    return 0.0
                
                # Evaluate using OJBench
                result = self.evaluator.evaluate_solution(
                    example["problem_id"], 
                    code, 
                    "python"
                )
                
                # Return 1.0 for AC (Accepted), 0.0 for any other verdict
                return 1.0 if result.get("verdict") == "AC" else 0.0
                
            except Exception as e:
                print(f"⚠️  Evaluation error: {e}")
                return 0.0
        
        return ojbench_metric
    
    def create_seed_prompt(self) -> Dict[str, str]:
        """Create initial seed prompt for competitive programming"""
        return {
            "system_prompt": """You are an expert competitive programmer. Your task is to solve competitive programming problems accurately and efficiently.

Instructions:
1. Read the problem statement carefully
2. Identify the problem type (algorithms, data structures, math, etc.)
3. Design an efficient solution considering time/space constraints
4. Implement the solution in clean, correct Python code
5. Ensure proper input/output handling as specified

Code Requirements:
- Read from stdin using input()
- Write to stdout using print()
- Handle edge cases (n=1, empty inputs, etc.)
- Use efficient algorithms for large constraints
- Follow the exact output format specified

Return your solution in the specified code block format."""
        }
    
    def run_optimization(self, 
                        problems_limit: int = 20,
                        max_metric_calls: int = 100,
                        difficulty: str = "easy") -> Any:
        """Run GEPA optimization on competitive programming problems"""
        
        print(f"\n🚀 Starting GEPA optimization...")
        print(f"   Problems: {problems_limit} {difficulty} problems")
        print(f"   Budget: {max_metric_calls} evaluations")
        
        # Load problems
        all_problems = self.load_ojbench_problems(difficulty, problems_limit * 2)
        
        # Split into train/val sets
        train_size = int(problems_limit * 0.7)  # 70% for training
        trainset = all_problems[:train_size]
        valset = all_problems[train_size:train_size + (problems_limit - train_size)]
        
        print(f"   Training set: {len(trainset)} problems")
        print(f"   Validation set: {len(valset)} problems")
        
        # Create initial prompt
        seed_prompt = self.create_seed_prompt()
        
        # Create evaluation metric
        metric_func = self.create_evaluation_metric()
        
        # Run GEPA optimization
        try:
            result = optimize(
                seed_candidate=seed_prompt,
                trainset=trainset,
                valset=valset,
                
                # Use OpenRouter models
                task_lm=self.config.TASK_MODEL,
                reflection_lm=self.reflection_lm_func,
                
                # Budget control
                max_metric_calls=max_metric_calls,
                
                # GEPA configuration
                reflection_minibatch_size=3,
                skip_perfect_score=True,
                perfect_score=1.0,
                
                # Logging
                logger=None,  # Use default stdout logger
                
                # Reproducibility
                seed=42
            )
            
            print("\n🎉 GEPA optimization completed!")
            print(f"Best prompt performance: {result.best_score:.2%}")
            
            return result
            
        except Exception as e:
            print(f"❌ Optimization failed: {e}")
            raise
    
    def validate_optimized_prompt(self, optimized_prompt: str, test_problems: List[Dict]) -> Dict[str, float]:
        """Validate the optimized prompt on held-out test problems"""
        
        print(f"\n🧪 Validating optimized prompt on {len(test_problems)} test problems...")
        
        successes = 0
        total = len(test_problems)
        
        for i, problem in enumerate(test_problems):
            try:
                # Generate solution using optimized prompt
                full_prompt = f"{optimized_prompt}\n\n{problem['prompt']}"
                response = self.task_lm_func(full_prompt)
                
                # Extract and evaluate code
                code = extract_python_code(response)
                result = self.evaluator.evaluate_solution(
                    problem["problem_id"], 
                    code, 
                    "python"
                )
                
                is_correct = result.get("verdict") == "AC"
                successes += is_correct
                
                print(f"   Problem {i+1}/{total}: {'✅ AC' if is_correct else '❌ ' + result.get('verdict', 'ERROR')}")
                
            except Exception as e:
                print(f"   Problem {i+1}/{total}: ❌ ERROR - {e}")
        
        accuracy = successes / total
        print(f"\n📊 Validation Results:")
        print(f"   Accuracy: {accuracy:.2%} ({successes}/{total})")
        
        return {
            "accuracy": accuracy,
            "successes": successes,
            "total": total
        }

def main():
    """Main execution function"""
    print("🔧 Setting up OpenRouter GEPA optimization for competitive programming...")
    
    try:
        # Initialize runner
        runner = OpenRouterGEPARunner()
        
        # Run optimization
        result = runner.run_optimization(
            problems_limit=15,  # Start small for testing
            max_metric_calls=50,  # Budget control
            difficulty="easy"
        )
        
        # Display results
        print("\n📈 Optimization Results:")
        print(f"Best candidate: {result.best_candidate}")
        print(f"Final score: {result.best_score:.2%}")
        
        # Validate on test set
        test_problems = runner.load_ojbench_problems("easy", 10)[15:]  # Different problems
        validation_results = runner.validate_optimized_prompt(
            result.best_candidate["system_prompt"], 
            test_problems
        )
        
        print("\n🎯 Success! GEPA optimization completed with OpenRouter API.")
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())