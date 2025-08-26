from typing import Dict, List, Tuple, Optional
import time

try:
    from ..models.qwen_interface import Qwen3Interface
except ImportError:
    from ..models.mock_model_interface import MockModelInterface as Qwen3Interface
    
from ..utils.code_parser import CodeParser
try:
    from ..evaluation.ojbench_interface import OJBenchEvaluator
except ImportError:
    from ..evaluation.mock_ojbench import MockOJBenchEvaluator as OJBenchEvaluator

class ARTSolver:
    def __init__(self, model_interface: Optional[Qwen3Interface] = None):
        self.model = model_interface or Qwen3Interface()
        self.parser = CodeParser()
        self.evaluator = OJBenchEvaluator()
        
    def solve_problem(self, problem: Dict, optimized_prompt: str, 
                     max_attempts: int = 3) -> Dict:
        """
        Solve a single problem using ART methodology
        
        ART Process:
        1. Give problem + optimized prompt to model
        2. Model thinks step-by-step ($ blocks)
        3. Model generates code
        4. **PAUSE** model generation  
        5. Run code through tools (OJBench)
        6. **RESUME** model with tool results
        7. Model analyzes results and corrects if needed
        """
        
        solution_log = {
            "problem_id": problem["id"],
            "attempts": [],
            "final_result": None,
            "success": False,
            "total_time": 0
        }
        
        start_time = time.time()
        
        # Build initial conversation
        conversation = f"{optimized_prompt}\n\n{problem['prompt']}"
        
        for attempt in range(max_attempts):
            print(f"🤖 Attempt {attempt + 1}/{max_attempts}")
            
            attempt_log = {
                "attempt_number": attempt + 1,
                "generation_time": 0,
                "evaluation_time": 0,
                "think_blocks": [],
                "code_generated": None,
                "language": None,
                "evaluation_result": None
            }
            
            # Step 1: Generate solution
            gen_start = time.time()
            response = self.model.generate(conversation, max_tokens=32000)
            attempt_log["generation_time"] = time.time() - gen_start
            
            # Step 2: Parse the response
            think_blocks = self.parser.extract_think_blocks(response)
            attempt_log["think_blocks"] = think_blocks
            
            try:
                language, code = self.parser.get_main_solution(response)
                attempt_log["code_generated"] = code
                attempt_log["language"] = language
                
                print(f"  Generated {language} solution ({len(code)} chars)")
                
            except ValueError as e:
                print(f"  ❌ No code found: {e}")
                attempt_log["error"] = str(e)
                solution_log["attempts"].append(attempt_log)
                continue
            
            # Step 3: **PAUSE** and use tools (evaluate with OJBench)
            print("  ⚙️  Running through OJBench...")
            eval_start = time.time()
            
            evaluation_result = self.evaluator.evaluate_solution(
                problem["id"], code, language
            )
            
            attempt_log["evaluation_time"] = time.time() - eval_start
            attempt_log["evaluation_result"] = evaluation_result
            
            # Step 4: Check if successful
            if evaluation_result["success"]:
                print("  ✅ Solution successful!")
                solution_log["success"] = True
                solution_log["final_result"] = "success"
                solution_log["attempts"].append(attempt_log)
                break
            
            print(f"  ❌ Solution failed: {evaluation_result['verdict']}")
            
            # Step 5: **RESUME** model with feedback for correction
            if attempt < max_attempts - 1:  # Not the last attempt
                feedback_prompt = self._create_feedback_prompt(
                    evaluation_result, think_blocks, code
                )
                conversation += f"\n\n{response}\n\n{feedback_prompt}"
                print("  🔄 Generating correction...")
            
            solution_log["attempts"].append(attempt_log)
        
        # Finalize results
        if not solution_log["success"]:
            solution_log["final_result"] = "max_attempts_exceeded"
        
        solution_log["total_time"] = time.time() - start_time
        
        return solution_log
    
    def _create_feedback_prompt(self, evaluation_result: Dict, 
                               think_blocks: List[str], code: str) -> str:
        """Create feedback prompt based on evaluation results"""
        
        verdict = evaluation_result["verdict"]
        
        # Analyze what went wrong
        error_analysis = self._analyze_error(verdict, evaluation_result)
        
        feedback = f"""
EVALUATION RESULT: Your solution failed with verdict "{verdict}"

ERROR ANALYSIS:
{error_analysis}

DEBUGGING GUIDANCE:
Please analyze your previous reasoning and code:

Your thinking process was:
{chr(10).join(f"- {block[:100]}..." for block in think_blocks)}

Your code attempt:
```{evaluation_result.get('language', 'cpp')}
{code[:300]}...
```

CORRECTION TASK:
1. Identify what went wrong in your approach
2. Revise your algorithm if needed  
3. Fix the implementation
4. Provide a corrected solution

Let me analyze what went wrong...
"""
        return feedback    
    def _analyze_error(self, verdict: str, evaluation_result: Dict) -> str:
        """Provide specific guidance based on the error type"""
        
        error_guidance = {
            "WA": "Wrong Answer - Your logic or implementation has an error. Check edge cases, off-by-one errors, and algorithm correctness.",
            "TLE": "Time Limit Exceeded - Your algorithm is too slow. Consider more efficient algorithms or data structures.",
            "MLE": "Memory Limit Exceeded - Your solution uses too much memory. Optimize data structures or algorithm approach.",
            "RE": "Runtime Error - Your code crashed during execution. Check array bounds, null pointers, and input handling.",
            "CE": "Compilation Error - Your code has syntax errors. Check C++ syntax, missing headers, or typos.",
            "OLE": "Output Limit Exceeded - Your code produces too much output. Check for infinite loops in output generation."
        }
        
        base_analysis = error_guidance.get(verdict, "Unknown error type")
        
        # Add more specific details if available
        detailed_results = evaluation_result.get("detailed_results", [])
        if detailed_results:
            first_failure = next((r for r in detailed_results 
                                if r.get("readable_main_code") != "AC"), None)
            if first_failure:
                feedback = first_failure.get("feedback", "")
                if feedback:
                    base_analysis += f"\n\nSpecific error details: {feedback}"
        
        return base_analysis

