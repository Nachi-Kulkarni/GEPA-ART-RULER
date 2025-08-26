"""
External competitive programming problem loader for training.
Loads problems from Codeforces, AtCoder, and other sources for RL training.
OJBench is reserved purely for final evaluation.
"""
import json
import requests
from typing import List, Dict, Any, Optional
from pathlib import Path
import time
import random


class ExternalProblemLoader:
    """
    Loads competitive programming problems from external sources for training.
    Never touches OJBench data to ensure proper evaluation.
    """
    
    def __init__(self, cache_dir: str = "data/training_problems"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Rate limiting for API calls
        self.last_request_time = 0
        self.min_request_interval = 1.0  # seconds between requests
    
    def load_codeforces_problems(self, 
                               count: int = 200,
                               min_solved: int = 100,
                               difficulty_range: tuple = (800, 1600)) -> List[Dict[str, Any]]:
        """
        Load Codeforces problems for GEPA training.
        
        Args:
            count: Number of problems to load
            min_solved: Minimum number of people who solved the problem
            difficulty_range: (min_rating, max_rating) for problem difficulty
        """
        
        cache_file = self.cache_dir / f"codeforces_{count}_{difficulty_range[0]}_{difficulty_range[1]}.json"
        
        # Check cache first
        if cache_file.exists():
            print(f"Loading cached Codeforces problems from {cache_file}")
            with open(cache_file, 'r') as f:
                return json.load(f)
        
        print(f"Fetching {count} Codeforces problems (difficulty {difficulty_range[0]}-{difficulty_range[1]})")
        
        try:
            # Rate limiting
            self._rate_limit()
            
            # Fetch problem list from Codeforces API
            response = requests.get("https://codeforces.com/api/problemset.problems")
            response.raise_for_status()
            
            data = response.json()
            if data["status"] != "OK":
                raise Exception(f"Codeforces API error: {data}")
            
            problems = data["result"]["problems"]
            problem_stats = {
                f"{p['contestId']}{p['index']}": stat 
                for p, stat in zip(problems, data["result"]["problemStatistics"])
            }
            
            # Filter problems by criteria
            filtered_problems = []
            for problem in problems:
                # Check if problem has required fields
                if not all(key in problem for key in ['contestId', 'index', 'name', 'rating']):
                    continue
                
                # Check difficulty range
                if not (difficulty_range[0] <= problem.get('rating', 0) <= difficulty_range[1]):
                    continue
                
                # Check solve count
                problem_key = f"{problem['contestId']}{problem['index']}"
                if problem_key not in problem_stats:
                    continue
                
                solve_count = problem_stats[problem_key].get('solvedCount', 0)
                if solve_count < min_solved:
                    continue
                
                # Convert to our format
                training_problem = {
                    "id": f"cf_{problem['contestId']}_{problem['index']}",
                    "source": "codeforces",
                    "contest_id": problem['contestId'],
                    "index": problem['index'],
                    "name": problem['name'],
                    "difficulty": problem.get('rating', 1000),
                    "solved_count": solve_count,
                    "tags": problem.get('tags', []),
                    "url": f"https://codeforces.com/problemset/problem/{problem['contestId']}/{problem['index']}",
                    "prompt": f"Problem: {problem['name']}\n\nDifficulty: {problem.get('rating', 'Unknown')}\nTags: {', '.join(problem.get('tags', []))}\n\nSolve this Codeforces problem.",
                    "language": "cpp",
                    "dataset": "training"
                }
                
                filtered_problems.append(training_problem)
                
                if len(filtered_problems) >= count:
                    break
            
            # Cache the results
            with open(cache_file, 'w') as f:
                json.dump(filtered_problems, f, indent=2)
            
            print(f"✅ Loaded {len(filtered_problems)} Codeforces problems")
            return filtered_problems
            
        except Exception as e:
            print(f"❌ Failed to load Codeforces problems: {e}")
            return self._get_fallback_codeforces_problems(count)
    
    def load_atcoder_problems(self, count: int = 300) -> List[Dict[str, Any]]:
        """
        Load AtCoder problems for RL training.
        """
        
        cache_file = self.cache_dir / f"atcoder_{count}.json"
        
        if cache_file.exists():
            print(f"Loading cached AtCoder problems from {cache_file}")
            with open(cache_file, 'r') as f:
                return json.load(f)
        
        print(f"Fetching {count} AtCoder problems")
        
        try:
            # Rate limiting
            self._rate_limit()
            
            # Fetch from AtCoder Problems API (unofficial)
            response = requests.get("https://kenkoooo.com/atcoder/resources/problems.json")
            response.raise_for_status()
            
            problems = response.json()
            
            # Convert to our format
            training_problems = []
            for i, problem in enumerate(problems):
                if i >= count:
                    break
                
                # Extract difficulty estimate (simplified)
                difficulty = "medium"  # Default
                if problem.get('title', '').lower().startswith(('a ', 'a.')):
                    difficulty = "easy"
                elif problem.get('title', '').lower().startswith(('d ', 'd.', 'e ', 'e.', 'f ', 'f.')):
                    difficulty = "hard"
                
                training_problem = {
                    "id": f"atc_{problem.get('id', i)}",
                    "source": "atcoder",
                    "name": problem.get('title', f'AtCoder Problem {i}'),
                    "contest_id": problem.get('contest_id', 'unknown'),
                    "difficulty": difficulty,
                    "url": f"https://atcoder.jp/contests/{problem.get('contest_id', 'unknown')}/tasks/{problem.get('id', '')}",
                    "prompt": f"Problem: {problem.get('title', 'AtCoder Problem')}\n\nContest: {problem.get('contest_id', 'Unknown')}\n\nSolve this AtCoder competitive programming problem.",
                    "language": "cpp",
                    "dataset": "training"
                }
                
                training_problems.append(training_problem)
            
            # Cache the results
            with open(cache_file, 'w') as f:
                json.dump(training_problems, f, indent=2)
            
            print(f"✅ Loaded {len(training_problems)} AtCoder problems")
            return training_problems
            
        except Exception as e:
            print(f"❌ Failed to load AtCoder problems: {e}")
            return self._get_fallback_atcoder_problems(count)
    
    def load_usaco_problems(self, count: int = 150) -> List[Dict[str, Any]]:
        """
        Load USACO training problems.
        """
        
        # USACO problems are typically manually curated
        # For now, return mock problems representing USACO-style problems
        
        cache_file = self.cache_dir / f"usaco_{count}.json"
        
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                return json.load(f)
        
        print(f"Generating {count} USACO-style problems")
        
        # Generate representative USACO problems
        usaco_templates = [
            {
                "name": "Cow Migration",
                "difficulty": "easy",
                "type": "simulation",
                "description": "Track cow movements in a field"
            },
            {
                "name": "Fence Planning", 
                "difficulty": "medium",
                "type": "graph",
                "description": "Find optimal fence layout"
            },
            {
                "name": "Milk Scheduling",
                "difficulty": "hard", 
                "type": "dynamic_programming",
                "description": "Optimize milk production schedule"
            }
        ]
        
        training_problems = []
        for i in range(count):
            template = usaco_templates[i % len(usaco_templates)]
            
            training_problem = {
                "id": f"usaco_{i:03d}",
                "source": "usaco",
                "name": f"{template['name']} {i+1}",
                "difficulty": template["difficulty"],
                "problem_type": template["type"],
                "prompt": f"Problem: {template['name']} {i+1}\n\nType: {template['type']}\nDifficulty: {template['difficulty']}\n\n{template['description']}. This is a USACO-style competitive programming problem.",
                "language": "cpp",
                "dataset": "training"
            }
            
            training_problems.append(training_problem)
        
        # Cache the results
        with open(cache_file, 'w') as f:
            json.dump(training_problems, f, indent=2)
        
        print(f"✅ Generated {len(training_problems)} USACO-style problems")
        return training_problems
    
    def get_training_dataset(self, 
                           gepa_size: int = 200,
                           rl_size: int = 500,
                           validation_split: float = 0.2) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get complete training dataset with proper splits.
        
        Returns:
            dict: {
                'gepa_train': problems for GEPA optimization,
                'rl_train': problems for RL training,
                'rl_val': problems for RL validation
            }
        """
        
        print("🔄 Loading external training dataset (NOT OJBench)")
        print("=" * 60)
        
        # Load problems from different sources
        codeforces_problems = self.load_codeforces_problems(count=gepa_size + 50)  # Extra for validation
        atcoder_problems = self.load_atcoder_problems(count=rl_size // 2)
        usaco_problems = self.load_usaco_problems(count=rl_size // 2)
        
        # Combine RL training data
        all_rl_problems = atcoder_problems + usaco_problems
        random.shuffle(all_rl_problems)
        
        # Create splits
        val_size = int(len(all_rl_problems) * validation_split)
        
        dataset = {
            'gepa_train': codeforces_problems[:gepa_size],
            'gepa_val': codeforces_problems[gepa_size:gepa_size + 30],  # Small validation set for GEPA
            'rl_train': all_rl_problems[:-val_size] if val_size > 0 else all_rl_problems,
            'rl_val': all_rl_problems[-val_size:] if val_size > 0 else all_rl_problems[:50],
        }
        
        print(f"📊 Training Dataset Summary:")
        print(f"   GEPA training: {len(dataset['gepa_train'])} problems (Codeforces)")
        print(f"   GEPA validation: {len(dataset['gepa_val'])} problems (Codeforces)")
        print(f"   RL training: {len(dataset['rl_train'])} problems (AtCoder + USACO)")
        print(f"   RL validation: {len(dataset['rl_val'])} problems (AtCoder + USACO)")
        print(f"   🚫 OJBench: Reserved for final evaluation only (232 problems)")
        
        return dataset
    
    def _rate_limit(self):
        """Rate limiting for API calls."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _get_fallback_codeforces_problems(self, count: int) -> List[Dict[str, Any]]:
        """Fallback Codeforces problems when API fails."""
        
        fallback_problems = []
        problem_types = ["implementation", "math", "greedy", "dp", "graph", "string"]
        
        for i in range(count):
            problem_type = problem_types[i % len(problem_types)]
            difficulty = 800 + (i % 8) * 100  # 800-1500 range
            
            fallback_problem = {
                "id": f"cf_fallback_{i:03d}",
                "source": "codeforces_fallback",
                "name": f"CF Problem {i+1}",
                "difficulty": difficulty,
                "tags": [problem_type],
                "prompt": f"Problem: Codeforces-style {problem_type} problem\n\nDifficulty: {difficulty}\n\nSolve this competitive programming problem using {problem_type} techniques.",
                "language": "cpp",
                "dataset": "training"
            }
            
            fallback_problems.append(fallback_problem)
        
        print(f"⚠️ Using {len(fallback_problems)} fallback Codeforces problems")
        return fallback_problems
    
    def _get_fallback_atcoder_problems(self, count: int) -> List[Dict[str, Any]]:
        """Fallback AtCoder problems when API fails."""
        
        fallback_problems = []
        contest_types = ["ABC", "ARC", "AGC"]
        
        for i in range(count):
            contest_type = contest_types[i % len(contest_types)]
            contest_num = 100 + i // 6
            problem_letter = chr(ord('A') + (i % 6))
            
            fallback_problem = {
                "id": f"atc_fallback_{contest_type}_{contest_num}_{problem_letter}",
                "source": "atcoder_fallback",
                "name": f"{contest_type} {contest_num} {problem_letter}",
                "contest_id": f"{contest_type.lower()}{contest_num}",
                "difficulty": "easy" if problem_letter in "AB" else "medium" if problem_letter in "CD" else "hard",
                "prompt": f"Problem: {contest_type} {contest_num} Problem {problem_letter}\n\nThis is an AtCoder competitive programming problem. Solve it step by step.",
                "language": "cpp",
                "dataset": "training"
            }
            
            fallback_problems.append(fallback_problem)
        
        print(f"⚠️ Using {len(fallback_problems)} fallback AtCoder problems")
        return fallback_problems


# Convenience function
def load_external_training_data(gepa_size: int = 200, rl_size: int = 500) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load external competitive programming problems for training.
    OJBench is reserved for evaluation only.
    """
    loader = ExternalProblemLoader()
    return loader.get_training_dataset(gepa_size=gepa_size, rl_size=rl_size)