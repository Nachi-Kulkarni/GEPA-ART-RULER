"""
Real OJBench integration for competitive programming evaluation.
No mocks - production DMOJ judge server integration.
"""
import subprocess
import json
import tempfile
import os
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class OJBenchEvaluator:
    """Production OJBench evaluator using DMOJ judge server."""
    
    def __init__(self):
        self.initialized = False
        self.problem_dirs = []
        self.temp_dir = Path(tempfile.mkdtemp(prefix="ojbench_"))
        
        # Verify DMOJ judge is available
        if not self._check_dmoj_availability():
            raise RuntimeError("DMOJ judge server not available. Run setup_dmoj.sh first.")
        
        print("✅ Real OJBench initialized with DMOJ judge server")
        self.initialized = True
    
    def _check_dmoj_availability(self) -> bool:
        """Check if DMOJ judge server is properly installed."""
        try:
            result = subprocess.run(
                ["dmoj-cli", "--help"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            logger.error("DMOJ judge server not found. Install with:")
            logger.error("cd judge-server && pip install -e .")
            return False
    
    def initialize_problems(self, problem_dirs: List[Path]) -> bool:
        """Initialize OJBench with problem directories."""
        try:
            import ojbench
            
            # Initialize with problem directories
            success = ojbench.init(problem_dirs)
            if success:
                self.problem_dirs = problem_dirs
                logger.info(f"OJBench initialized with {len(problem_dirs)} problem directories")
                return True
            else:
                logger.error("OJBench initialization failed")
                return False
                
        except ImportError:
            logger.error("OJBench module not found. Install with: pip install -e OJBench")
            return False
        except Exception as e:
            logger.error(f"OJBench initialization error: {e}")
            return False
    
    def get_problems_subset(self, difficulty: str = None, language: str = "cpp", limit: int = 100) -> List[Dict[str, Any]]:
        """Get subset of problems for training/evaluation."""
        try:
            import ojbench
            
            # Get all available problems
            problems = ojbench.get_problems()
            
            # Filter by difficulty if specified
            if difficulty:
                problems = [p for p in problems if p.get('difficulty', '').lower() == difficulty.lower()]
            
            # Filter by language support
            problems = [p for p in problems if language in p.get('languages', ['cpp', 'python'])]
            
            # Limit results
            problems = problems[:limit]
            
            logger.info(f"Retrieved {len(problems)} problems (difficulty={difficulty}, language={language})")
            return problems
            
        except Exception as e:
            logger.error(f"Failed to get problems: {e}")
            return []
    
    def get_problem_info(self, problem_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific problem."""
        try:
            import ojbench
            
            problem = ojbench.get_problem(problem_id)
            if problem:
                return {
                    "id": problem_id,
                    "title": problem.get('title', f'Problem {problem_id}'),
                    "statement": problem.get('statement', ''),
                    "constraints": problem.get('constraints', []),
                    "sample_input": problem.get('sample_input', ''),
                    "sample_output": problem.get('sample_output', ''),
                    "difficulty": problem.get('difficulty', 'unknown'),
                    "dataset": problem.get('dataset', 'unknown'),
                    "time_limit": problem.get('time_limit', 2000),
                    "memory_limit": problem.get('memory_limit', 262144),
                    "languages": problem.get('languages', ['cpp', 'python'])
                }
            return None
            
        except Exception as e:
            logger.error(f"Failed to get problem {problem_id}: {e}")
            return None
    
    def evaluate_solution(self, problem_id: str, code: str, language: str = "cpp") -> Dict[str, Any]:
        """Evaluate a code solution using DMOJ judge."""
        try:
            # Create temporary files for evaluation
            solution_file = self.temp_dir / f"solution_{problem_id}.{self._get_extension(language)}"
            input_file = self.temp_dir / f"input_{problem_id}.jsonl"
            output_file = self.temp_dir / f"output_{problem_id}.jsonl"
            
            # Write solution to file
            with open(solution_file, 'w', encoding='utf-8') as f:
                f.write(code)
            
            # Create evaluation input
            eval_input = {
                "id": problem_id,
                "code": code,
                "language": language,
                "problem_id": problem_id
            }
            
            with open(input_file, 'w', encoding='utf-8') as f:
                json.dump(eval_input, f)
                f.write('\n')
            
            # Run OJBench evaluation
            import ojbench
            result = ojbench.judge_jsonl(str(input_file), str(output_file))
            
            if result and output_file.exists():
                # Read evaluation results
                with open(output_file, 'r', encoding='utf-8') as f:
                    eval_result = json.loads(f.readline().strip())
                
                return self._format_result(eval_result)
            else:
                return self._error_result(problem_id, "Evaluation failed")
                
        except Exception as e:
            logger.error(f"Evaluation error for {problem_id}: {e}")
            return self._error_result(problem_id, str(e))
    
    def _get_extension(self, language: str) -> str:
        """Get file extension for language."""
        extensions = {
            "cpp": "cpp",
            "c++": "cpp", 
            "python": "py",
            "py": "py",
            "java": "java",
            "c": "c"
        }
        return extensions.get(language.lower(), "txt")
    
    def _format_result(self, eval_result: Dict[str, Any]) -> Dict[str, Any]:
        """Format evaluation result to standard format."""
        verdict = eval_result.get('verdict', 'unknown')
        
        return {
            "success": verdict == "AC",  # Accepted
            "verdict": verdict,
            "execution_time": eval_result.get('time', 0.0),
            "memory_usage": eval_result.get('memory', 0),
            "test_cases_passed": eval_result.get('cases_passed', 0),
            "total_test_cases": eval_result.get('total_cases', 0),
            "score": eval_result.get('score', 0.0),
            "error_message": eval_result.get('error', ''),
            "feedback": eval_result.get('feedback', ''),
            "problem_id": eval_result.get('problem_id', ''),
            "language": eval_result.get('language', ''),
            "detailed_results": eval_result.get('case_results', []),
            "timestamp": eval_result.get('timestamp'),
            "error_analysis": self._analyze_error(verdict, eval_result)
        }
    
    def _error_result(self, problem_id: str, error_msg: str) -> Dict[str, Any]:
        """Create error result format."""
        return {
            "success": False,
            "verdict": "IE",  # Internal Error
            "execution_time": 0.0,
            "memory_usage": 0,
            "test_cases_passed": 0,
            "total_test_cases": 0,
            "score": 0.0,
            "error_message": error_msg,
            "feedback": f"System error during evaluation: {error_msg}",
            "problem_id": problem_id,
            "language": "unknown",
            "detailed_results": [],
            "timestamp": None,
            "error_analysis": {"type": "system_error", "message": error_msg}
        }
    
    def _analyze_error(self, verdict: str, eval_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze error type and provide suggestions."""
        analysis = {"verdict": verdict}
        
        if verdict == "CE":  # Compilation Error
            analysis.update({
                "type": "compilation_error",
                "category": "syntax",
                "suggestions": [
                    "Check syntax and includes",
                    "Verify language-specific requirements",
                    "Ensure proper main function signature"
                ]
            })
        elif verdict == "WA":  # Wrong Answer
            analysis.update({
                "type": "wrong_answer", 
                "category": "logic",
                "suggestions": [
                    "Verify algorithm logic",
                    "Check edge cases handling",
                    "Validate input/output format"
                ]
            })
        elif verdict == "TLE":  # Time Limit Exceeded
            analysis.update({
                "type": "time_limit",
                "category": "efficiency", 
                "suggestions": [
                    "Optimize algorithm complexity",
                    "Use more efficient data structures",
                    "Avoid unnecessary computations"
                ]
            })
        elif verdict == "MLE":  # Memory Limit Exceeded
            analysis.update({
                "type": "memory_limit",
                "category": "efficiency",
                "suggestions": [
                    "Reduce memory usage",
                    "Use more memory-efficient data structures",
                    "Avoid storing unnecessary data"
                ]
            })
        elif verdict == "RE":  # Runtime Error
            analysis.update({
                "type": "runtime_error",
                "category": "implementation",
                "suggestions": [
                    "Check array bounds",
                    "Handle division by zero",
                    "Validate pointer/reference usage"
                ]
            })
        
        return analysis
    
    def get_success_rate(self, results: List[Dict[str, Any]]) -> float:
        """Calculate success rate from evaluation results."""
        if not results:
            return 0.0
        
        successes = sum(1 for r in results if r.get("success", False))
        return successes / len(results)
    
    def cleanup(self):
        """Clean up temporary files."""
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
            logger.info("OJBench cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
    
    def __del__(self):
        """Cleanup on destruction."""
        self.cleanup()