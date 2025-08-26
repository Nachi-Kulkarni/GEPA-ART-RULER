"""
OJBench evaluation interface for GEPA optimization
Handles competitive programming problem evaluation
"""
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional
import ojbench

class OJBenchEvaluator:
    """Interface for evaluating competitive programming solutions with OJBench"""
    
    def __init__(self):
        self.initialized = False
        self._initialize_ojbench()
    
    def _initialize_ojbench(self):
        """Initialize OJBench with problem datasets"""
        try:
            # Get paths to test data
            base_path = Path(__file__).parent.parent.parent
            noi_path = base_path / "OJBench_testdata" / "NOI"
            icpc_path = base_path / "OJBench_testdata" / "ICPC"
            
            # Initialize OJBench
            problem_dirs = []
            if noi_path.exists():
                problem_dirs.append(noi_path)
            if icpc_path.exists():
                problem_dirs.append(icpc_path)
            
            if not problem_dirs:
                raise Exception("No OJBench test data found. Please ensure OJBench_testdata exists.")
            
            ojbench.init(problem_dirs)
            self.initialized = True
            print(f"✅ OJBench initialized with {len(problem_dirs)} problem directories")
            
        except Exception as e:
            print(f"❌ Failed to initialize OJBench: {e}")
            self.initialized = False
    
    def evaluate_solution(self, problem_id: int, code: str, language: str = "python") -> Dict[str, Any]:
        """
        Evaluate a solution against OJBench test cases
        
        Args:
            problem_id: Problem ID from OJBench dataset
            code: Solution code to evaluate
            language: Programming language ("python" or "cpp")
            
        Returns:
            Dict with evaluation results including verdict and details
        """
        if not self.initialized:
            return {
                "success": False,
                "verdict": "ERROR",
                "message": "OJBench not initialized"
            }
        
        try:
            # Create temporary file for solution
            with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{language}', delete=False) as f:
                f.write(code)
                solution_path = Path(f.name)
            
            # Create input JSONL for OJBench
            input_data = {
                "id": problem_id,
                "code": code,
                "language": language
            }
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
                json.dump(input_data, f)
                f.write('\n')
                input_jsonl_path = Path(f.name)
            
            # Create output path for results
            output_jsonl_path = input_jsonl_path.with_suffix('.out.jsonl')
            
            try:
                # Run OJBench evaluation
                ojbench.judge_jsonl(str(input_jsonl_path), str(output_jsonl_path))
                
                # Read results
                if output_jsonl_path.exists():
                    with open(output_jsonl_path, 'r') as f:
                        result_line = f.readline().strip()
                        if result_line:
                            result = json.loads(result_line)
                            
                            return {
                                "success": result.get("is_passed", False),
                                "verdict": result.get("verdict", "ERROR"),
                                "time": result.get("time", 0),
                                "memory": result.get("memory", 0),
                                "message": result.get("message", ""),
                                "problem_id": problem_id
                            }
                
                # If no results found
                return {
                    "success": False,
                    "verdict": "ERROR",
                    "message": "No evaluation results returned",
                    "problem_id": problem_id
                }
                
            finally:
                # Cleanup temporary files
                try:
                    solution_path.unlink(missing_ok=True)
                    input_jsonl_path.unlink(missing_ok=True) 
                    output_jsonl_path.unlink(missing_ok=True)
                except:
                    pass
                    
        except Exception as e:
            return {
                "success": False,
                "verdict": "ERROR",
                "message": str(e),
                "problem_id": problem_id
            }
    
    def get_success_rate(self, results: List[Dict[str, Any]]) -> float:
        """Calculate success rate from evaluation results"""
        if not results:
            return 0.0
        
        successes = sum(1 for r in results if r.get("success", False))
        return successes / len(results)

def load_ojbench_problems(difficulty: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load problems from OJBench dataset"""
    problems_path = Path(__file__).parent.parent.parent / "OJBench_testdata" / "prompts" / "full.jsonl"
    
    if not problems_path.exists():
        raise FileNotFoundError(f"OJBench problems file not found: {problems_path}")
    
    problems = []
    
    with open(problems_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
                
            try:
                problem = json.loads(line.strip())
                
                # Filter by difficulty if specified
                if difficulty and problem.get("difficulty") != difficulty:
                    continue
                
                problems.append(problem)
                
                # Stop if we've reached the limit
                if limit and len(problems) >= limit:
                    break
                    
            except json.JSONDecodeError as e:
                print(f"⚠️  Skipping malformed line: {e}")
                continue
    
    return problems