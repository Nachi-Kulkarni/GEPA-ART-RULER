"""
Mock OJBench interface for testing without DMOJ judge system
"""
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Any

class MockOJBenchEvaluator:
    """Mock evaluator for testing GEPA optimization without DMOJ"""
    
    def __init__(self):
        self.initialized = True
        self.is_mock = True
        print("✅ Mock OJBench initialized (no actual judging - for GEPA testing only)")
    
    def get_problems_subset(self, difficulty: str = None, language: str = "cpp", limit: int = 10):
        """Mock method to return sample problems for GEPA training"""
        problems = [
            {"id": f"mock_{difficulty}_1", "prompt": f"Sample {difficulty} problem 1", 
             "difficulty": difficulty, "dataset": "mock", "language": language},
            {"id": f"mock_{difficulty}_2", "prompt": f"Sample {difficulty} problem 2", 
             "difficulty": difficulty, "dataset": "mock", "language": language},
        ]
        return problems[:limit]
    
    def evaluate_solution(self, problem_id: int, code: str, language: str = "python") -> Dict[str, Any]:
        """
        Mock evaluation that simulates code judging
        Returns random results for testing GEPA optimization
        """
        # Simulate processing time
        time.sleep(0.1)
        
        # Simple heuristics to make some problems "easier" than others
        # This allows GEPA to learn patterns even without real judging
        
        # Basic code quality checks
        has_main_function = "def main(" in code
        has_input_handling = "input()" in code
        has_print_output = "print(" in code
        has_error_handling = "try:" in code or "except:" in code
        lines_of_code = len([line for line in code.split('\n') if line.strip()])
        
        # Score based on code quality indicators
        quality_score = 0
        if has_main_function:
            quality_score += 0.3
        if has_input_handling:
            quality_score += 0.3
        if has_print_output:
            quality_score += 0.2
        if has_error_handling:
            quality_score += 0.1
        if 5 <= lines_of_code <= 50:  # Reasonable length
            quality_score += 0.1
            
        # Add some randomness to simulate real problem difficulty
        random_factor = random.random() * 0.4
        final_score = quality_score + random_factor
        
        # Determine verdict based on score
        if final_score > 0.7:
            verdict = "AC"
            success = True
        elif final_score > 0.5:
            verdict = random.choice(["WA", "AC"])
            success = (verdict == "AC")
        elif final_score > 0.3:
            verdict = random.choice(["WA", "TLE", "AC"])
            success = (verdict == "AC")
        else:
            verdict = random.choice(["WA", "TLE", "RE", "CE"])
            success = False
        
        return {
            "success": success,
            "verdict": verdict,
            "time": random.uniform(0.1, 2.0),
            "memory": random.randint(1000, 10000),
            "message": f"Mock evaluation for problem {problem_id}",
            "problem_id": problem_id,
            "mock_score": final_score  # For debugging
        }
    
    def get_success_rate(self, results: List[Dict[str, Any]]) -> float:
        """Calculate success rate from evaluation results"""
        if not results:
            return 0.0
        
        successes = sum(1 for r in results if r.get("success", False))
        return successes / len(results)

def init(problem_dirs):
    """Mock init function"""
    print("🧪 Mock OJBench init - using simulated judging for GEPA testing")
    return True

def judge_jsonl(input_path, output_path):
    """Mock judge function"""
    print(f"🧪 Mock judging: {input_path} -> {output_path}")
    return True

# Create mock module
import types
mock_ojbench = types.ModuleType('ojbench')
mock_ojbench.init = init
mock_ojbench.judge_jsonl = judge_jsonl
mock_ojbench.MockOJBenchEvaluator = MockOJBenchEvaluator

# Add to sys.modules so it can be imported
import sys
sys.modules['ojbench'] = mock_ojbench
