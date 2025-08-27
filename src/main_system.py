import time
import json
from pathlib import Path
from typing import Dict, List, Optional

from .models.qwen_local_interface import Qwen3LocalInterface
from .gepa.run_gepa import run_gepa_optimization
from .art.art_solver import ARTSolver  
from .ruler.ruler_analyzer import RULERAnalyzer
from .evaluation.real_ojbench import OJBenchEvaluator

class GEPAARTRULERSystem:
    def __init__(self, cache_dir: str = "data/cache"):
        """Initialize the complete system"""
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize components
        print("🤖 Initializing Qwen3-4B-Thinking-2507 model...")
        # Model config would be passed from the main configuration
        model_config = {
            "name": "Qwen/Qwen3-4B-Thinking-2507",
            "context_length": 4096,
            "max_generation_tokens": 4096,
            "trust_remote_code": True,
            "torch_dtype": "float16",
            "device_map": "auto"
        }
        self.model = Qwen3LocalInterface(model_config)
        
        print("⚙️  Initializing ART solver...")
        self.art_solver = ARTSolver(self.model)
        
        print("👑 Initializing RULER analyzer...")
        self.ruler_analyzer = RULERAnalyzer()
        
        print("📊 Initializing OJBench evaluator...")
        self.evaluator = OJBenchEvaluator()
        
        # Load or prepare optimized prompt
        self.optimized_prompt = self._get_optimized_prompt()
        
        print("✅ System initialization complete!")
    
    def _get_optimized_prompt(self) -> str:
        """Get optimized prompt (from GEPA cache or run GEPA)"""
        
        cached_prompt_file = self.cache_dir / "optimized_prompt.txt"
        
        if cached_prompt_file.exists():
            print("📄 Loading cached optimized prompt...")
            return cached_prompt_file.read_text()
        else:
            print("🧬 No cached prompt found. Running GEPA optimization...")
            return self.run_gepa_phase()
    
    def run_gepa_phase(self) -> str:
        """Run the GEPA prompt optimization phase"""
        
        print("🧬 PHASE 1: GEPA Prompt Evolution")
        print("=" * 50)
        
        optimized_prompt = run_gepa_optimization(
            model_interface=self.model,
            output_dir=str(self.cache_dir / "gepa_results")
        )
        
        # Cache the result
        cached_prompt_file = self.cache_dir / "optimized_prompt.txt"
        cached_prompt_file.write_text(optimized_prompt)
        
        print("💾 Optimized prompt cached for future use")
        
        return optimized_prompt
    
    def solve_problem_with_art_ruler(self, problem: Dict) -> Dict:
        """
        Solve a single problem using ART+RULER integration
        
        This is where ART and RULER work together:
        1. ART handles the reasoning and tool-use flow
        2. RULER analyzes failures and provides correction guidance
        3. They iterate together until success or max attempts
        """
        
        print(f"🎯 Solving problem {problem['id']} ({problem['difficulty']})")
        
        solution_log = {
            "problem_id": problem["id"],
            "problem_info": {
                "difficulty": problem.get("difficulty"),
                "dataset": problem.get("dataset"),
                "language": problem.get("language")
            },
            "attempts": [],
            "ruler_analyses": [],
            "final_result": None,
            "success": False,
            "total_time": 0
        }
        
        start_time = time.time()
        conversation_history = f"{self.optimized_prompt}\n\n{problem['prompt']}"
        max_attempts = 3
        
        for attempt in range(max_attempts):
            print(f"  🔄 Attempt {attempt + 1}/{max_attempts}")
            
            # ART: Generate solution with structured reasoning
            attempt_start = time.time()
            response = self.model.generate(conversation_history, max_tokens=3200)
            
            # Parse the response
            from .utils.code_parser import CodeParser
            parser = CodeParser()
            
            think_blocks = parser.extract_think_blocks(response)
            
            try:
                language, code = parser.get_main_solution(response)
                print(f"    💻 Generated {language} solution")
                
                # ART: Evaluate with tools (OJBench)
                evaluation_result = self.evaluator.evaluate_solution(
                    problem["id"], code, language
                )
                
                attempt_log = {
                    "attempt_number": attempt + 1,
                    "think_blocks": think_blocks,
                    "generated_code": code,
                    "language": language,
                    "evaluation_result": evaluation_result,
                    "time_taken": time.time() - attempt_start
                }
                
                # Check for success
                if evaluation_result["success"]:
                    print("    ✅ Solution successful!")
                    solution_log["success"] = True
                    solution_log["final_result"] = "success"
                    solution_log["attempts"].append(attempt_log)
                    break
                
                print(f"    ❌ Failed: {evaluation_result['verdict']}")
                solution_log["attempts"].append(attempt_log)
                
                # RULER: Analyze failure and provide guidance
                if attempt < max_attempts - 1:  # Not the last attempt
                    
                    # RULER analysis of internal reasoning
                    think_analysis = self.ruler_analyzer.analyze_think_blocks(think_blocks)
                    
                    # RULER analysis of execution error  
                    execution_diagnosis = self.ruler_analyzer.analyze_execution_error(evaluation_result)
                    
                    # RULER creates correction guidance
                    correction_guidance = self.ruler_analyzer.create_correction_guidance(
                        think_analysis, execution_diagnosis, code
                    )
                    
                    ruler_analysis = {
                        "attempt_number": attempt + 1,
                        "think_analysis": think_analysis,
                        "execution_diagnosis": execution_diagnosis.__dict__,
                        "correction_guidance": correction_guidance
                    }
                    solution_log["ruler_analyses"].append(ruler_analysis)
                    
                    # Update conversation with RULER's guidance
                    conversation_history += f"\n\n{response}\n\n{correction_guidance}"
                    
                    print("    🔧 RULER analysis complete, generating correction...")
                
            except ValueError as e:
                print(f"    ❌ No code generated: {e}")
                attempt_log = {
                    "attempt_number": attempt + 1,
                    "error": str(e),
                    "think_blocks": think_blocks,
                    "time_taken": time.time() - attempt_start
                }
                solution_log["attempts"].append(attempt_log)
        
        # Finalize results
        if not solution_log["success"]:
            solution_log["final_result"] = "max_attempts_exceeded"
        
        solution_log["total_time"] = time.time() - start_time
        
        return solution_log
    
    def run_evaluation(self, problems: List[Dict], results_dir: str = "data/results") -> Dict:
        """Run complete evaluation on a set of problems"""
        
        print("🚀 PHASE 2: ART+RULER Problem Solving")
        print("=" * 50)
        
        results_path = Path(results_dir)
        results_path.mkdir(exist_ok=True, parents=True)
        
        evaluation_results = {
            "timestamp": time.time(),
            "total_problems": len(problems),
            "results": [],
            "summary": {}
        }
        
        successful_solutions = 0
        total_time = 0
        
        for i, problem in enumerate(problems):
            print(f"\n--- Problem {i+1}/{len(problems)} ---")
            
            solution_log = self.solve_problem_with_art_ruler(problem)
            evaluation_results["results"].append(solution_log)
            
            if solution_log["success"]:
                successful_solutions += 1
                print(f"✅ Problem {problem['id']}: SUCCESS")
            else:
                print(f"❌ Problem {problem['id']}: FAILED")
            
            total_time += solution_log["total_time"]
        
        # Calculate summary statistics
        pass_rate = successful_solutions / len(problems)
        
        evaluation_results["summary"] = {
            "pass_rate": pass_rate,
            "successful_solutions": successful_solutions,
            "failed_solutions": len(problems) - successful_solutions,
            "total_evaluation_time": total_time,
            "average_time_per_problem": total_time / len(problems)
        }
        
        # Save results
        with open(results_path / "evaluation_results.json", "w") as f:
            json.dump(evaluation_results, f, indent=2)
        
        # Generate summary report
        self._generate_summary_report(evaluation_results, results_path)
        
        print("\n" + "=" * 50)
        print("🎉 EVALUATION COMPLETE!")
        print(f"Pass Rate: {pass_rate:.1%} ({successful_solutions}/{len(problems)})")
        print(f"Baseline: 17.9% → Our System: {pass_rate:.1%}")
        print(f"Improvement: {((pass_rate - 0.179) / 0.179 * 100):+.1f}%")
        
        return evaluation_results
    
    def _generate_summary_report(self, results: Dict, output_path: Path):
        """Generate human-readable summary report"""
        
        report = f'''# GEPA+ART+RULER System Evaluation Report

## System Configuration
- Model: Qwen3-4B-Thinking-2507
- Framework: GEPA + ART + RULER
- Max Attempts per Problem: 3

## Performance Results
- **Pass Rate**: {results['summary']['pass_rate']:.1%}
- **Successful Solutions**: {results['summary']['successful_solutions']}
- **Failed Solutions**: {results['summary']['failed_solutions']}
- **Total Problems**: {results['total_problems']}

## Performance Comparison
- **Baseline (Qwen3-4B)**: 17.9%
- **Our System**: {results['summary']['pass_rate']:.1%}
- **Improvement**: {((results['summary']['pass_rate'] - 0.179) / 0.179 * 100):+.1f}%

## Timing Analysis
- **Total Evaluation Time**: {results['summary']['total_evaluation_time']:.1f} seconds
- **Average per Problem**: {results['summary']['average_time_per_problem']:.1f} seconds

## Detailed Results

| Problem ID | Result | Attempts | Time (s) |
|------------|--------|----------|----------|
'''
        
        for result in results['results']:
            status = "✅" if result['success'] else "❌"
            attempts_used = len(result['attempts'])
            problem_id = result['problem_id']
            time_taken = result['total_time']
            
            report += f"| {problem_id} | {status} | {attempts_used} | {time_taken:.1f} |\n"
        
        report += f'''

## System Analysis

### Success Patterns
- Problems solved in 1 attempt: {sum(1 for r in results['results'] if r['success'] and len(r['attempts']) == 1)}
- Problems solved in 2 attempts: {sum(1 for r in results['results'] if r['success'] and len(r['attempts']) == 2)}
- Problems solved in 3 attempts: {sum(1 for r in results['results'] if r['success'] and len(r['attempts']) == 3)}

### RULER Effectiveness
- Total RULER analyses performed: {sum(len(r.get('ruler_analyses', [])) for r in results['results'])}
- Problems corrected by RULER: {sum(1 for r in results['results'] if r['success'] and len(r['attempts']) > 1)}

### Key Insights
1. **GEPA Impact**: Optimized prompt provides better initial reasoning
2. **ART Structure**: Systematic thinking and tool integration improves success rate
3. **RULER Correction**: Error analysis and correction guidance enables recovery from failures
4. **System Synergy**: The three components work together to achieve {((results['summary']['pass_rate'] - 0.179) / 0.179 * 100):+.1f}% improvement

## Next Steps
1. Analyze failure patterns for further optimization
2. Extend GEPA evolution for domain-specific prompts
3. Enhance RULER with more sophisticated error patterns
4. Scale evaluation to full OJBench dataset
'''
        
        with open(output_path / "summary_report.md", "w") as f:
            f.write(report)
        
        print(f"📄 Detailed report saved to {output_path / 'summary_report.md'}")