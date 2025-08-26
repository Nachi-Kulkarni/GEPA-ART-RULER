#!/usr/bin/env python3
"""
GEPA+ART+RULER System - Main Entry Point

A sophisticated AI code generation system combining:
- GEPA: Genetic-Pareto prompt optimization using LLM-based reflection
- ART: Automatic Reasoning and Tool-use for competitive programming  
- RULER: Recursive Error Resolution for error analysis and correction

Targets OJBench competitive programming problems to improve
Qwen3-4B-Thinking-2507 from 17.9% to 50%+ success rate.
"""

import argparse
import json
import time
from pathlib import Path
from src.main_system import GEPAARTRULERSystem
from src.evaluation.mock_ojbench import MockOJBenchEvaluator

def get_test_problems(difficulty: str = "easy", limit: int = 5) -> list:
    """Generate test problems for demonstration"""
    
    problems = []
    
    if difficulty == "easy":
        problems = [
            {
                "id": "sample_easy_1",
                "prompt": "Given an integer n, output n * 2.",
                "difficulty": "easy",
                "dataset": "demo",
                "language": "cpp"
            },
            {
                "id": "sample_easy_2", 
                "prompt": "Given a string s, output its length.",
                "difficulty": "easy",
                "dataset": "demo",
                "language": "cpp"
            },
            {
                "id": "sample_easy_3",
                "prompt": "Given two integers a and b, output their sum.",
                "difficulty": "easy", 
                "dataset": "demo",
                "language": "cpp"
            }
        ]
    elif difficulty == "medium":
        problems = [
            {
                "id": "sample_medium_1",
                "prompt": "Given an array of n integers, find the maximum sum of any contiguous subarray.",
                "difficulty": "medium",
                "dataset": "demo", 
                "language": "cpp"
            },
            {
                "id": "sample_medium_2",
                "prompt": "Given a binary tree, return the inorder traversal of its nodes' values.",
                "difficulty": "medium",
                "dataset": "demo",
                "language": "cpp"
            }
        ]
    else:  # hard
        problems = [
            {
                "id": "sample_hard_1", 
                "prompt": "Find the longest common subsequence between two strings.",
                "difficulty": "hard",
                "dataset": "demo",
                "language": "cpp"
            }
        ]
    
    return problems[:limit]

def main():
    parser = argparse.ArgumentParser(
        description="GEPA+ART+RULER System for Competitive Programming",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # Quick demo with 3 easy problems
  python main.py --full-gepa              # Run full GEPA optimization + evaluation
  python main.py --problems-limit 10      # Test with 10 problems
  python main.py --difficulty medium      # Use medium difficulty problems
  python main.py --output-dir results/    # Save results to custom directory
        """
    )
    
    parser.add_argument(
        "--full-gepa", 
        action="store_true",
        help="Run complete GEPA optimization (slow, uses model heavily)"
    )
    parser.add_argument(
        "--problems-limit",
        type=int, 
        default=3,
        help="Number of problems to evaluate (default: 3)"
    )
    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard"],
        default="easy", 
        help="Problem difficulty level (default: easy)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/results",
        help="Output directory for results (default: data/results)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 GEPA+ART+RULER System")
    print("   Advanced AI Code Generation for Competitive Programming")
    print("=" * 60)
    print(f"📊 Configuration:")
    print(f"   • Problems: {args.problems_limit} ({args.difficulty} difficulty)")
    print(f"   • GEPA Optimization: {'Yes' if args.full_gepa else 'No (using cached/base prompt)'}")
    print(f"   • Output Directory: {args.output_dir}")
    print()
    
    try:
        # Initialize system
        print("🔧 Initializing system components...")
        system = GEPAARTRULERSystem(cache_dir="data/cache")
        
        # Get test problems
        print(f"📚 Loading {args.problems_limit} test problems...")
        problems = get_test_problems(args.difficulty, args.problems_limit)
        
        print(f"Selected problems:")
        for p in problems:
            print(f"  • {p['id']}: {p['prompt'][:60]}...")
        
        # Run GEPA phase if requested
        if args.full_gepa:
            print("\n" + "="*50)
            print("🧬 PHASE 1: GEPA PROMPT OPTIMIZATION")
            print("="*50)
            optimized_prompt = system.run_gepa_phase()
            print(f"✅ GEPA optimization complete!")
        else:
            print(f"\n⚡ Skipping GEPA optimization (using cached/base prompt)")
            
        # Run evaluation phase
        print("\n" + "="*50)
        print("🎯 PHASE 2: ART+RULER EVALUATION")  
        print("="*50)
        
        start_time = time.time()
        results = system.run_evaluation(problems, args.output_dir)
        total_time = time.time() - start_time
        
        # Display results
        print("\n" + "="*50)
        print("📈 FINAL RESULTS")
        print("="*50)
        
        pass_rate = results["summary"]["pass_rate"] 
        successful = results["summary"]["successful_solutions"]
        total = results["total_problems"]
        
        print(f"🎯 Success Rate: {pass_rate:.1%} ({successful}/{total})")
        print(f"📊 Baseline Comparison:")
        print(f"   • Qwen3-4B Baseline: 17.9%")
        print(f"   • Our System: {pass_rate:.1%}")
        improvement = ((pass_rate - 0.179) / 0.179 * 100) if pass_rate > 0 else -100
        print(f"   • Improvement: {improvement:+.1f}%")
        
        print(f"\n⏱️  Performance:")
        print(f"   • Total Time: {total_time:.1f}s")
        print(f"   • Average per Problem: {total_time/total:.1f}s")
        
        print(f"\n📁 Results saved to: {Path(args.output_dir).absolute()}")
        
        # Show individual results
        print(f"\n📋 Individual Results:")
        for result in results["results"]:
            status = "✅" if result["success"] else "❌"
            attempts = len(result["attempts"])
            problem_id = result["problem_id"]
            time_taken = result["total_time"]
            
            print(f"   {status} {problem_id}: {attempts} attempts, {time_taken:.1f}s")
        
        # Show RULER effectiveness
        ruler_analyses = sum(len(r.get("ruler_analyses", [])) for r in results["results"])
        ruler_recoveries = sum(1 for r in results["results"] if r["success"] and len(r["attempts"]) > 1)
        
        print(f"\n🔧 RULER Effectiveness:")
        print(f"   • Total Error Analyses: {ruler_analyses}")
        print(f"   • Successful Recoveries: {ruler_recoveries}")
        
        if pass_rate >= 0.25:  # 25% target
            print(f"\n🎉 SUCCESS! Target performance achieved!")
        elif pass_rate > 0.179:
            print(f"\n📈 IMPROVEMENT! Better than baseline!")
        else:
            print(f"\n⚠️  Below baseline - system needs tuning")
            
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ System error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print(f"\n🏁 GEPA+ART+RULER evaluation complete!")
    return 0

if __name__ == "__main__":
    exit(main())