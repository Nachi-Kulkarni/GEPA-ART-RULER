"""
RULER error analysis integration for reward shaping in RL training.
Enhances trajectory rewards with detailed error analysis and correction guidance.
"""
from typing import Dict, List, Any, Optional, Tuple
import re
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from ruler.ruler_analyzer import RULERAnalyzer
except ImportError:
    print("RULER analyzer not available, using fallback")
    RULERAnalyzer = None

from .competitive_programming_trajectory import CompetitiveProgrammingTrajectory


class RULERRewardShaper:
    """
    Integrates RULER error analysis to enhance reward shaping for RL training.
    Provides detailed feedback on reasoning quality and error patterns.
    """
    
    def __init__(self):
        self.ruler_analyzer = RULERAnalyzer() if RULERAnalyzer else None
        
        # Error type to reward impact mapping
        self.error_penalties = {
            "CE": 0.9,   # Compilation error - moderate penalty
            "WA": 0.7,   # Wrong answer - higher penalty  
            "TLE": 0.6,  # Time limit - algorithm issue penalty
            "MLE": 0.5,  # Memory limit - algorithm issue penalty
            "RE": 0.4,   # Runtime error - severe penalty
            "OLE": 0.8   # Output limit - moderate penalty
        }
        
        # Reasoning quality indicators
        self.reasoning_patterns = {
            "algorithm_analysis": [
                "algorithm", "complexity", "time complexity", "space complexity",
                "O(", "approach", "strategy", "method"
            ],
            "edge_case_consideration": [
                "edge case", "corner case", "boundary", "special case",
                "minimum", "maximum", "empty", "zero", "overflow"
            ],
            "problem_understanding": [
                "understand", "given", "need to", "problem asks", "requirement",
                "constraint", "input format", "output format"
            ],
            "implementation_planning": [
                "implement", "code", "variable", "loop", "condition",
                "function", "data structure", "array", "vector"
            ]
        }
    
    def enhance_trajectory_reward(self, trajectory: CompetitiveProgrammingTrajectory) -> Dict[str, Any]:
        """
        Enhance trajectory with RULER-based reward components.
        """
        
        enhanced_components = trajectory.get_reward_components().copy()
        
        # Add RULER-specific reward components
        ruler_components = self._analyze_with_ruler(trajectory)
        enhanced_components.update(ruler_components)
        
        # Calculate enhanced total reward
        enhanced_weights = {
            # Original components
            "success_reward": 4.0,
            "time_efficiency": 1.0,
            "attempt_efficiency": 1.0,
            "reasoning_quality": 1.5,
            "internal_analysis": 1.0,
            "partial_progress": 0.5,
            "tool_efficiency": 0.5,
            
            # RULER-enhanced components
            "ruler_error_analysis": 1.5,    # Error pattern recognition
            "ruler_correction_quality": 1.0, # Quality of error corrections
            "ruler_reasoning_depth": 1.0,    # Depth of reasoning analysis
            "ruler_learning_progress": 0.8   # Learning from mistakes
        }
        
        enhanced_reward = sum(
            enhanced_components.get(key, 0.0) * weight 
            for key, weight in enhanced_weights.items()
        )
        
        return {
            "enhanced_components": enhanced_components,
            "enhanced_reward": enhanced_reward,
            "ruler_analysis": ruler_components,
            "improvement_suggestions": self._generate_improvement_suggestions(trajectory, ruler_components)
        }
    
    def _analyze_with_ruler(self, trajectory: CompetitiveProgrammingTrajectory) -> Dict[str, float]:
        """
        Perform RULER analysis on trajectory and return reward components.
        """
        
        ruler_components = {}
        
        # 1. Error Pattern Analysis
        error_analysis_score = self._analyze_error_patterns(trajectory)
        ruler_components["ruler_error_analysis"] = error_analysis_score
        
        # 2. Reasoning Quality Analysis
        reasoning_depth_score = self._analyze_reasoning_depth(trajectory.think_blocks)
        ruler_components["ruler_reasoning_depth"] = reasoning_depth_score
        
        # 3. Correction Quality (if there were multiple attempts)
        if trajectory.attempt_number > 1:
            correction_quality_score = self._analyze_correction_quality(trajectory)
            ruler_components["ruler_correction_quality"] = correction_quality_score
        else:
            ruler_components["ruler_correction_quality"] = 0.5  # Neutral for first attempt
        
        # 4. Learning Progress Assessment
        learning_progress_score = self._assess_learning_progress(trajectory)
        ruler_components["ruler_learning_progress"] = learning_progress_score
        
        return ruler_components
    
    def _analyze_error_patterns(self, trajectory: CompetitiveProgrammingTrajectory) -> float:
        """
        Analyze error patterns and assign reward based on error type and recovery.
        """
        
        if trajectory.success:
            return 1.0  # Full reward for successful solutions
        
        # Base penalty based on error type
        base_penalty = self.error_penalties.get(trajectory.verdict, 0.3)
        
        # Adjust based on partial progress
        if trajectory.total_test_cases > 0:
            partial_success = trajectory.test_cases_passed / trajectory.total_test_cases
            # Reward partial progress, especially for algorithmic errors
            if trajectory.verdict in ["WA", "TLE", "MLE"]:
                base_penalty = min(base_penalty + partial_success * 0.3, 1.0)
        
        # Use RULER analyzer if available
        if self.ruler_analyzer and trajectory.error_analysis:
            ruler_feedback = self._get_ruler_error_feedback(trajectory)
            if ruler_feedback.get("has_correction_guidance", False):
                base_penalty += 0.1  # Bonus for receiving helpful error analysis
        
        return base_penalty
    
    def _analyze_reasoning_depth(self, think_blocks: List[str]) -> float:
        """
        Analyze the depth and quality of reasoning in think blocks.
        """
        
        if not think_blocks:
            return 0.1  # Very low score for no reasoning
        
        total_score = 0.0
        max_possible_score = len(self.reasoning_patterns)
        
        # Check for presence of different types of reasoning
        all_text = " ".join(think_blocks).lower()
        
        for pattern_type, keywords in self.reasoning_patterns.items():
            pattern_score = 0.0
            
            # Check for keyword presence
            keyword_matches = sum(1 for keyword in keywords if keyword in all_text)
            if keyword_matches > 0:
                pattern_score = min(keyword_matches / len(keywords), 1.0)
            
            total_score += pattern_score
        
        # Normalize to 0-1 range
        depth_score = total_score / max_possible_score
        
        # Bonus for longer, more detailed reasoning
        avg_length = sum(len(block) for block in think_blocks) / len(think_blocks)
        length_bonus = min(avg_length / 200, 0.2)  # Up to 0.2 bonus for detailed reasoning
        
        # Penalty for very short or generic reasoning
        if avg_length < 50:
            depth_score *= 0.5
        
        return min(depth_score + length_bonus, 1.0)
    
    def _analyze_correction_quality(self, trajectory: CompetitiveProgrammingTrajectory) -> float:
        """
        Analyze quality of error corrections in multi-attempt solutions.
        """
        
        # This would analyze how well the agent learned from previous failures
        # For now, implement basic heuristics
        
        correction_score = 0.5  # Base score
        
        # Bonus for successful correction
        if trajectory.success:
            correction_score += 0.4
        
        # Penalty for taking many attempts
        attempt_penalty = (trajectory.attempt_number - 1) * 0.1
        correction_score = max(correction_score - attempt_penalty, 0.1)
        
        # Bonus for showing understanding of the error
        if trajectory.correction_suggestions:
            correction_score += 0.1
        
        return min(correction_score, 1.0)
    
    def _assess_learning_progress(self, trajectory: CompetitiveProgrammingTrajectory) -> float:
        """
        Assess how well the agent is learning from the problem-solving process.
        """
        
        learning_score = 0.5  # Base score
        
        # Bonus for good tool usage efficiency
        learning_score += trajectory.tool_usage_efficiency * 0.2
        
        # Bonus for high reasoning coherence
        learning_score += trajectory.reasoning_coherence * 0.2
        
        # Bonus for successful solutions with good internal analysis
        if trajectory.success and trajectory.internal_reasoning_quality > 0.7:
            learning_score += 0.1
        
        return min(learning_score, 1.0)
    
    def _get_ruler_error_feedback(self, trajectory: CompetitiveProgrammingTrajectory) -> Dict[str, Any]:
        """
        Get RULER-specific feedback on errors (placeholder for full integration).
        """
        
        if not self.ruler_analyzer:
            return {"has_correction_guidance": False}
        
        try:
            # Call the production RULER analyzer for error analysis
            if hasattr(trajectory, 'generated_code') and trajectory.generated_code:
                # Analyze the generated code and error for correction guidance
                evaluation_result = {
                    "verdict": trajectory.verdict,
                    "success": trajectory.verdict == "AC",
                    "detailed_results": getattr(trajectory, 'detailed_results', [])
                }
                
                # Use RULER analyzer for comprehensive error analysis
                execution_diagnosis = self.ruler_analyzer.analyze_execution_error(evaluation_result)
                
                return {
                    "has_correction_guidance": trajectory.verdict in ["WA", "TLE", "MLE", "RE", "CE"],
                    "error_category": execution_diagnosis.error_type,
                    "correction_guidance": execution_diagnosis.guidance_text,
                    "complexity_issues": execution_diagnosis.likely_causes
                }
            else:
                return {
                    "has_correction_guidance": trajectory.verdict in ["WA", "TLE", "MLE"],
                    "error_category": self._categorize_error(trajectory.verdict),
                "correction_confidence": 0.7
            }
        except Exception:
            return {"has_correction_guidance": False}
    
    def _categorize_error(self, verdict: str) -> str:
        """Categorize error types for RULER analysis."""
        
        categories = {
            "CE": "syntax_error",
            "WA": "logic_error", 
            "TLE": "efficiency_error",
            "MLE": "memory_error",
            "RE": "runtime_error",
            "OLE": "output_error"
        }
        
        return categories.get(verdict, "unknown_error")
    
    def _generate_improvement_suggestions(self, 
                                        trajectory: CompetitiveProgrammingTrajectory,
                                        ruler_components: Dict[str, float]) -> List[str]:
        """
        Generate specific improvement suggestions based on RULER analysis.
        """
        
        suggestions = []
        
        # Reasoning depth suggestions
        if ruler_components.get("ruler_reasoning_depth", 0) < 0.5:
            suggestions.append("Improve reasoning depth by analyzing algorithm complexity and edge cases")
        
        # Error analysis suggestions
        if ruler_components.get("ruler_error_analysis", 0) < 0.6:
            if trajectory.verdict == "WA":
                suggestions.append("Focus on logic verification and test case analysis")
            elif trajectory.verdict == "TLE":
                suggestions.append("Consider more efficient algorithms or data structures")
            elif trajectory.verdict == "MLE":
                suggestions.append("Optimize memory usage or reconsider data structures")
            elif trajectory.verdict == "CE":
                suggestions.append("Review syntax and language-specific requirements")
        
        # Learning progress suggestions
        if ruler_components.get("ruler_learning_progress", 0) < 0.6:
            suggestions.append("Work on connecting problem patterns to solution approaches")
        
        # Tool usage suggestions
        if trajectory.tool_usage_efficiency < 0.7:
            suggestions.append("Improve tool usage by testing solutions incrementally")
        
        return suggestions
    
    def batch_enhance_trajectories(self, 
                                 trajectories: List[CompetitiveProgrammingTrajectory]) -> List[Dict[str, Any]]:
        """
        Enhance multiple trajectories with RULER analysis.
        """
        
        enhanced_trajectories = []
        
        for trajectory in trajectories:
            enhanced = self.enhance_trajectory_reward(trajectory)
            enhanced["original_trajectory"] = trajectory
            enhanced_trajectories.append(enhanced)
        
        return enhanced_trajectories
    
    def get_reward_shaping_stats(self, enhanced_trajectories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics on reward shaping effectiveness.
        """
        
        if not enhanced_trajectories:
            return {"error": "No trajectories provided"}
        
        # Calculate reward improvements
        original_rewards = []
        enhanced_rewards = []
        
        for enh_traj in enhanced_trajectories:
            orig_traj = enh_traj["original_trajectory"]
            orig_reward = orig_traj.calculate_total_reward()
            enhanced_reward = enh_traj["enhanced_reward"]
            
            original_rewards.append(orig_reward)
            enhanced_rewards.append(enhanced_reward)
        
        import numpy as np
        
        stats = {
            "trajectory_count": len(enhanced_trajectories),
            "original_reward": {
                "mean": np.mean(original_rewards),
                "std": np.std(original_rewards),
                "min": np.min(original_rewards),
                "max": np.max(original_rewards)
            },
            "enhanced_reward": {
                "mean": np.mean(enhanced_rewards),
                "std": np.std(enhanced_rewards),
                "min": np.min(enhanced_rewards),
                "max": np.max(enhanced_rewards)
            },
            "improvement": {
                "mean_difference": np.mean(enhanced_rewards) - np.mean(original_rewards),
                "improvement_rate": (np.mean(enhanced_rewards) - np.mean(original_rewards)) / np.mean(original_rewards) * 100
            }
        }
        
        return stats


# Convenience function
def create_ruler_reward_shaper() -> RULERRewardShaper:
    """Create a RULER reward shaper instance."""
    return RULERRewardShaper()