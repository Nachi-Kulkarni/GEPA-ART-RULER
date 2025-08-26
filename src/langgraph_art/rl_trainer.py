"""
Reinforcement Learning trainer for the LangGraph ART agent.
Integrates with OpenPipe for trajectory-based learning.
"""
from typing import List, Dict, Any, Optional, Callable
import asyncio
import json
import numpy as np
from datetime import datetime
from pathlib import Path

from .competitive_programming_trajectory import CompetitiveProgrammingTrajectory
from .langgraph_art_agent import LangGraphARTAgent

# OpenPipe RL integration (if available)
try:
    from openpipe import log_trajectory, start_training
    OPENPIPE_RL_AVAILABLE = True
except ImportError:
    OPENPIPE_RL_AVAILABLE = False
    print("OpenPipe RL features not available, using local training simulation")


class CompetitiveProgrammingRLTrainer:
    """
    Reinforcement Learning trainer for competitive programming agents.
    """
    
    def __init__(self, 
                 agent: LangGraphARTAgent,
                 reward_weights: Optional[Dict[str, float]] = None,
                 save_trajectories: bool = True,
                 trajectory_save_path: str = "data/trajectories"):
        
        self.agent = agent
        self.save_trajectories = save_trajectories
        self.trajectory_save_path = Path(trajectory_save_path)
        self.trajectory_save_path.mkdir(parents=True, exist_ok=True)
        
        # Default reward weights for competitive programming
        self.reward_weights = reward_weights or {
            "success_reward": 4.0,      # Primary goal: solve the problem
            "time_efficiency": 1.0,     # Solve quickly
            "attempt_efficiency": 1.0,  # Solve in fewer attempts
            "reasoning_quality": 1.5,   # Good reasoning process
            "internal_analysis": 1.0,   # Quality internal analysis
            "partial_progress": 0.5,    # Credit for partial solutions
            "tool_efficiency": 0.5      # Efficient tool usage
        }
        
        self.training_history = []
        self.best_performance = 0.0
        
    async def collect_trajectories(self, 
                                 problems: List[Dict], 
                                 optimized_prompt: str = "",
                                 num_episodes: int = 1) -> List[CompetitiveProgrammingTrajectory]:
        """
        Collect trajectories by running the agent on problems.
        """
        
        all_trajectories = []
        
        for episode in range(num_episodes):
            print(f"\n🎯 Episode {episode + 1}/{num_episodes}")
            
            for i, problem in enumerate(problems):
                print(f"  Problem {i+1}/{len(problems)}: {problem['id']}")
                
                # Clear previous trajectories
                self.agent.clear_trajectories()
                
                # Solve problem
                result = await self.agent.solve_problem(problem, optimized_prompt)
                
                # Get trajectories from this run
                problem_trajectories = self.agent.get_trajectories()
                all_trajectories.extend(problem_trajectories)
                
                print(f"    Result: {'✅' if result['success'] else '❌'} "
                      f"({result.get('final_result', 'unknown')})")
        
        print(f"\n📊 Collected {len(all_trajectories)} trajectories")
        
        # Save trajectories locally
        if self.save_trajectories:
            self._save_trajectories_to_disk(all_trajectories)
        
        # Log to OpenPipe (if available)
        if OPENPIPE_RL_AVAILABLE:
            await self._log_trajectories_to_openpipe(all_trajectories)
        
        return all_trajectories
    
    def evaluate_trajectories(self, 
                            trajectories: List[CompetitiveProgrammingTrajectory]) -> Dict[str, Any]:
        """
        Evaluate a collection of trajectories.
        """
        
        if not trajectories:
            return {"error": "No trajectories to evaluate"}
        
        # Basic statistics
        total_trajectories = len(trajectories)
        successful_trajectories = sum(1 for t in trajectories if t.success)
        success_rate = successful_trajectories / total_trajectories
        
        # Performance by problem type
        noi_problems = [t for t in trajectories if t.problem_dataset == "NOI"]
        icpc_problems = [t for t in trajectories if t.problem_dataset == "ICPC"]
        
        noi_success_rate = (sum(1 for t in noi_problems if t.success) / len(noi_problems)) if noi_problems else 0.0
        icpc_success_rate = (sum(1 for t in icpc_problems if t.success) / len(icpc_problems)) if icpc_problems else 0.0
        
        # Performance by difficulty
        difficulty_stats = {}
        for difficulty in ["easy", "medium", "hard"]:
            diff_trajectories = [t for t in trajectories if t.problem_difficulty == difficulty]
            if diff_trajectories:
                diff_success_rate = sum(1 for t in diff_trajectories if t.success) / len(diff_trajectories)
                difficulty_stats[difficulty] = {
                    "count": len(diff_trajectories),
                    "success_rate": diff_success_rate
                }
        
        # Reward analysis
        rewards = [t.calculate_total_reward(self.reward_weights) for t in trajectories]
        avg_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        
        # Time analysis (successful solutions only)
        successful_times = [t.total_solve_time for t in trajectories if t.success and t.total_solve_time > 0]
        avg_solve_time = np.mean(successful_times) if successful_times else 0.0
        
        # Attempt analysis
        avg_attempts = np.mean([t.attempt_number for t in trajectories])
        
        # Language preference
        cpp_count = sum(1 for t in trajectories if t.language == "cpp")
        python_count = sum(1 for t in trajectories if t.language == "python")
        
        evaluation_result = {
            "total_trajectories": total_trajectories,
            "success_rate": success_rate,
            "successful_solutions": successful_trajectories,
            "performance_by_dataset": {
                "NOI": {"count": len(noi_problems), "success_rate": noi_success_rate},
                "ICPC": {"count": len(icpc_problems), "success_rate": icpc_success_rate}
            },
            "performance_by_difficulty": difficulty_stats,
            "reward_statistics": {
                "average_reward": avg_reward,
                "std_reward": std_reward,
                "max_reward": max(rewards) if rewards else 0.0,
                "min_reward": min(rewards) if rewards else 0.0
            },
            "timing_statistics": {
                "average_solve_time": avg_solve_time,
                "successful_solutions_count": len(successful_times)
            },
            "attempt_statistics": {
                "average_attempts": avg_attempts
            },
            "language_usage": {
                "cpp": cpp_count,
                "python": python_count
            },
            "evaluation_timestamp": datetime.now().isoformat()
        }
        
        return evaluation_result
    
    async def train_iteration(self, 
                            train_problems: List[Dict],
                            val_problems: List[Dict],
                            optimized_prompt: str = "",
                            num_episodes: int = 1) -> Dict[str, Any]:
        """
        Run one training iteration.
        """
        
        print(f"\n🚀 Starting training iteration")
        print(f"   Training problems: {len(train_problems)}")
        print(f"   Validation problems: {len(val_problems)}")
        print(f"   Episodes per problem: {num_episodes}")
        
        # Collect training trajectories
        print("\n📚 Collecting training trajectories...")
        train_trajectories = await self.collect_trajectories(
            train_problems, optimized_prompt, num_episodes
        )
        
        # Evaluate training performance
        train_evaluation = self.evaluate_trajectories(train_trajectories)
        train_success_rate = train_evaluation.get("success_rate", 0.0)
        
        print(f"   Training success rate: {train_success_rate:.1%}")
        
        # Collect validation trajectories
        print("\n🔍 Collecting validation trajectories...")
        val_trajectories = await self.collect_trajectories(
            val_problems, optimized_prompt, 1  # Single episode for validation
        )
        
        # Evaluate validation performance
        val_evaluation = self.evaluate_trajectories(val_trajectories)
        val_success_rate = val_evaluation.get("success_rate", 0.0)
        
        print(f"   Validation success rate: {val_success_rate:.1%}")
        
        # Update best performance
        if val_success_rate > self.best_performance:
            self.best_performance = val_success_rate
            print(f"   🎉 New best validation performance: {self.best_performance:.1%}")
        
        # Training iteration results
        iteration_result = {
            "train_evaluation": train_evaluation,
            "val_evaluation": val_evaluation,
            "train_success_rate": train_success_rate,
            "val_success_rate": val_success_rate,
            "best_performance": self.best_performance,
            "train_trajectories_count": len(train_trajectories),
            "val_trajectories_count": len(val_trajectories),
            "iteration_timestamp": datetime.now().isoformat()
        }
        
        # Save iteration results
        self.training_history.append(iteration_result)
        
        # Trigger OpenPipe training (if available)
        if OPENPIPE_RL_AVAILABLE and len(train_trajectories) > 0:
            await self._trigger_openpipe_training(train_trajectories)
        
        return iteration_result
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training progress."""
        
        if not self.training_history:
            return {"message": "No training iterations completed yet"}
        
        iterations = len(self.training_history)
        latest_train_success = self.training_history[-1]["train_success_rate"]
        latest_val_success = self.training_history[-1]["val_success_rate"]
        
        # Track performance over time
        train_success_rates = [h["train_success_rate"] for h in self.training_history]
        val_success_rates = [h["val_success_rate"] for h in self.training_history]
        
        summary = {
            "training_iterations": iterations,
            "best_validation_performance": self.best_performance,
            "latest_train_success_rate": latest_train_success,
            "latest_val_success_rate": latest_val_success,
            "train_success_trend": train_success_rates,
            "val_success_trend": val_success_rates,
            "performance_improvement": val_success_rates[-1] - val_success_rates[0] if len(val_success_rates) > 1 else 0.0,
            "total_trajectories": sum(h["train_trajectories_count"] + h["val_trajectories_count"] for h in self.training_history),
            "training_start": self.training_history[0]["iteration_timestamp"],
            "last_update": self.training_history[-1]["iteration_timestamp"]
        }
        
        return summary
    
    def _save_trajectories_to_disk(self, trajectories: List[CompetitiveProgrammingTrajectory]):
        """Save trajectories to local disk."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.trajectory_save_path / f"trajectories_{timestamp}.json"
        
        # Convert trajectories to serializable format
        trajectory_data = [t.to_dict() for t in trajectories]
        
        with open(filename, 'w') as f:
            json.dump(trajectory_data, f, indent=2)
        
        print(f"   💾 Saved {len(trajectories)} trajectories to {filename}")
    
    async def _log_trajectories_to_openpipe(self, 
                                          trajectories: List[CompetitiveProgrammingTrajectory]):
        """Log trajectories to OpenPipe for RL training."""
        
        try:
            for trajectory in trajectories:
                # Convert trajectory to OpenPipe format
                openpipe_trajectory = {
                    "problem_id": trajectory.problem_id,
                    "inputs": {
                        "problem_statement": trajectory.problem_statement,
                        "difficulty": trajectory.problem_difficulty,
                        "dataset": trajectory.problem_dataset
                    },
                    "outputs": {
                        "generated_code": trajectory.generated_code,
                        "language": trajectory.language,
                        "think_blocks": trajectory.think_blocks
                    },
                    "reward": trajectory.calculate_total_reward(self.reward_weights),
                    "metadata": {
                        "verdict": trajectory.verdict,
                        "success": trajectory.success,
                        "attempt_number": trajectory.attempt_number,
                        "session_id": trajectory.session_id,
                        "timestamp": trajectory.timestamp.isoformat()
                    }
                }
                
                # Log to OpenPipe
                await log_trajectory(openpipe_trajectory)
            
            print(f"   🔗 Logged {len(trajectories)} trajectories to OpenPipe")
            
        except Exception as e:
            print(f"   ⚠️ Failed to log trajectories to OpenPipe: {str(e)}")
    
    async def _trigger_openpipe_training(self, trajectories: List[CompetitiveProgrammingTrajectory]):
        """Trigger OpenPipe RL training with collected trajectories."""
        
        try:
            # Start training with the collected data
            training_config = {
                "model": "qwen3-4b-thinking",
                "reward_type": "competitive_programming",
                "min_trajectories": len(trajectories),
                "training_params": {
                    "learning_rate": 1e-4,
                    "batch_size": 8,
                    "max_epochs": 5
                }
            }
            
            training_job = await start_training(training_config)
            print(f"   🔥 Started OpenPipe RL training: {training_job}")
            
        except Exception as e:
            print(f"   ⚠️ Failed to start OpenPipe training: {str(e)}")
    
    def save_training_state(self, filepath: str):
        """Save the current training state."""
        
        state = {
            "reward_weights": self.reward_weights,
            "best_performance": self.best_performance,
            "training_history": self.training_history,
            "save_timestamp": datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        print(f"💾 Training state saved to {filepath}")
    
    def load_training_state(self, filepath: str):
        """Load training state from file."""
        
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.reward_weights = state.get("reward_weights", self.reward_weights)
            self.best_performance = state.get("best_performance", 0.0)
            self.training_history = state.get("training_history", [])
            
            print(f"📂 Training state loaded from {filepath}")
            print(f"   Best performance: {self.best_performance:.1%}")
            print(f"   Training iterations: {len(self.training_history)}")
            
        except Exception as e:
            print(f"⚠️ Failed to load training state: {str(e)}")


# Convenience function for easy training setup
def create_rl_trainer(agent: LangGraphARTAgent, **kwargs) -> CompetitiveProgrammingRLTrainer:
    """Create an RL trainer for the given agent."""
    return CompetitiveProgrammingRLTrainer(agent, **kwargs)