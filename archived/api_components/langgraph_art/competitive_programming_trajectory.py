"""
Custom trajectory class for competitive programming problems.
Based on OpenPipe ART integration documentation.
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

@dataclass
class CompetitiveProgrammingTrajectory:
    """
    Custom trajectory class for competitive programming domain.
    Stores metadata and performance metrics specific to coding competitions.
    """
    
    # Problem information
    problem_id: str
    problem_difficulty: str
    problem_dataset: str  # "NOI" or "ICPC"
    problem_statement: str
    
    # Solution attempt information
    language: str  # "cpp" or "python"
    generated_code: str
    think_blocks: List[str]  # Reasoning steps from model
    
    # Evaluation results
    verdict: str  # "AC", "WA", "TLE", "MLE", "RE", "CE", "OLE"
    success: bool
    execution_time: Optional[float]
    memory_usage: Optional[float]
    test_cases_passed: int
    total_test_cases: int
    
    # RULER analysis
    error_analysis: Optional[Dict[str, Any]]
    correction_suggestions: List[str]
    internal_reasoning_quality: float  # 0.0-1.0 score
    
    # Performance metrics
    generation_time: float
    evaluation_time: float
    total_solve_time: float
    attempt_number: int
    max_attempts: int
    
    # Metadata for RL training
    prompt_optimization_score: Optional[float]  # From GEPA
    tool_usage_efficiency: float  # How well tools were used
    reasoning_coherence: float  # Quality of thinking blocks
    
    # Session context
    session_id: str
    timestamp: datetime
    model_version: str
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def get_reward_components(self) -> Dict[str, float]:
        """
        Calculate different reward components for RL training.
        """
        components = {}
        
        # Primary reward: Success/failure
        components["success_reward"] = 1.0 if self.success else 0.0
        
        # Efficiency rewards
        if self.success:
            components["time_efficiency"] = max(0.0, 1.0 - (self.total_solve_time / 300.0))  # Normalize to 5 min
            components["attempt_efficiency"] = 1.0 / self.attempt_number  # Fewer attempts = better
        else:
            components["time_efficiency"] = 0.0
            components["attempt_efficiency"] = 0.0
        
        # Reasoning quality rewards
        components["reasoning_quality"] = self.reasoning_coherence
        components["internal_analysis"] = self.internal_reasoning_quality
        
        # Partial progress rewards (even for failed attempts)
        if not self.success and self.total_test_cases > 0:
            components["partial_progress"] = self.test_cases_passed / self.total_test_cases
        else:
            components["partial_progress"] = 0.0
        
        # Tool usage efficiency
        components["tool_efficiency"] = self.tool_usage_efficiency
        
        return components
    
    def calculate_total_reward(self, weights: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate weighted total reward for RL training.
        """
        if weights is None:
            weights = {
                "success_reward": 4.0,
                "time_efficiency": 1.0,
                "attempt_efficiency": 1.0,
                "reasoning_quality": 1.5,
                "internal_analysis": 1.0,
                "partial_progress": 0.5,
                "tool_efficiency": 0.5
            }
        
        components = self.get_reward_components()
        total_reward = sum(components.get(key, 0.0) * weight 
                          for key, weight in weights.items())
        
        return total_reward
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trajectory to dictionary for storage/analysis."""
        return {
            "problem_id": self.problem_id,
            "problem_difficulty": self.problem_difficulty,
            "problem_dataset": self.problem_dataset,
            "language": self.language,
            "verdict": self.verdict,
            "success": self.success,
            "execution_time": self.execution_time,
            "memory_usage": self.memory_usage,
            "test_cases_passed": self.test_cases_passed,
            "total_test_cases": self.total_test_cases,
            "generation_time": self.generation_time,
            "evaluation_time": self.evaluation_time,
            "total_solve_time": self.total_solve_time,
            "attempt_number": self.attempt_number,
            "max_attempts": self.max_attempts,
            "reasoning_coherence": self.reasoning_coherence,
            "internal_reasoning_quality": self.internal_reasoning_quality,
            "tool_usage_efficiency": self.tool_usage_efficiency,
            "reward_components": self.get_reward_components(),
            "total_reward": self.calculate_total_reward(),
            "timestamp": self.timestamp.isoformat(),
            "session_id": self.session_id,
            "model_version": self.model_version
        }
    
    @classmethod
    def from_art_solution_log(cls, solution_log: Dict, problem: Dict, 
                             session_id: str, model_version: str = "qwen3-4b-thinking") -> 'CompetitiveProgrammingTrajectory':
        """
        Create trajectory from existing ART solver solution log.
        """
        last_attempt = solution_log["attempts"][-1] if solution_log["attempts"] else {}
        
        # Extract evaluation result
        eval_result = last_attempt.get("evaluation_result", {})
        
        return cls(
            problem_id=solution_log["problem_id"],
            problem_difficulty=problem.get("difficulty", "unknown"),
            problem_dataset=problem.get("dataset", "unknown"),
            problem_statement=problem.get("prompt", ""),
            language=last_attempt.get("language", "cpp"),
            generated_code=last_attempt.get("code_generated", ""),
            think_blocks=last_attempt.get("think_blocks", []),
            verdict=eval_result.get("verdict", "unknown"),
            success=solution_log["success"],
            execution_time=eval_result.get("execution_time"),
            memory_usage=eval_result.get("memory_usage"),
            test_cases_passed=eval_result.get("test_cases_passed", 0),
            total_test_cases=eval_result.get("total_test_cases", 0),
            error_analysis=eval_result.get("error_analysis"),
            correction_suggestions=[],
            internal_reasoning_quality=cls._assess_reasoning_quality(last_attempt.get("think_blocks", [])),
            generation_time=last_attempt.get("generation_time", 0.0),
            evaluation_time=last_attempt.get("evaluation_time", 0.0),
            total_solve_time=solution_log["total_time"],
            attempt_number=len(solution_log["attempts"]),
            max_attempts=3,  # Default from ART solver
            prompt_optimization_score=None,
            tool_usage_efficiency=cls._assess_tool_usage(eval_result),
            reasoning_coherence=cls._assess_reasoning_coherence(last_attempt.get("think_blocks", [])),
            session_id=session_id,
            timestamp=datetime.now(),
            model_version=model_version
        )
    
    @staticmethod
    def _assess_reasoning_quality(think_blocks: List[str]) -> float:
        """Assess quality of internal reasoning (simplified heuristic)."""
        if not think_blocks:
            return 0.0
        
        quality_score = 0.0
        
        # Check for structured thinking
        has_algorithm_analysis = any("algorithm" in block.lower() for block in think_blocks)
        has_complexity_analysis = any("complexity" in block.lower() or "O(" in block for block in think_blocks)
        has_edge_case_consideration = any("edge" in block.lower() or "corner" in block.lower() for block in think_blocks)
        
        quality_score += 0.3 if has_algorithm_analysis else 0.0
        quality_score += 0.3 if has_complexity_analysis else 0.0
        quality_score += 0.2 if has_edge_case_consideration else 0.0
        
        # Length and detail bonus (more detailed reasoning is generally better)
        avg_length = sum(len(block) for block in think_blocks) / len(think_blocks)
        length_bonus = min(0.2, avg_length / 500.0)  # Normalize to reasonable length
        quality_score += length_bonus
        
        return min(1.0, quality_score)
    
    @staticmethod
    def _assess_reasoning_coherence(think_blocks: List[str]) -> float:
        """Assess coherence of reasoning flow."""
        if not think_blocks:
            return 0.0
        
        # Simple heuristics for coherence
        coherence_score = 0.5  # Base score
        
        # Check for logical flow indicators
        has_problem_understanding = any(word in think_blocks[0].lower() 
                                      for word in ["understand", "given", "need to", "problem"])
        has_solution_approach = any("approach" in block.lower() or "solution" in block.lower() 
                                  for block in think_blocks)
        has_implementation_details = any("implement" in block.lower() or "code" in block.lower() 
                                       for block in think_blocks)
        
        coherence_score += 0.2 if has_problem_understanding else 0.0
        coherence_score += 0.2 if has_solution_approach else 0.0
        coherence_score += 0.1 if has_implementation_details else 0.0
        
        return min(1.0, coherence_score)
    
    @staticmethod
    def _assess_tool_usage(eval_result: Dict) -> float:
        """Assess how efficiently tools (OJBench) were used."""
        if not eval_result:
            return 0.0
        
        # If successful, tool usage was efficient
        if eval_result.get("success", False):
            return 1.0
        
        # If failed but got partial results, moderate efficiency
        if eval_result.get("test_cases_passed", 0) > 0:
            return 0.5
        
        # If failed completely, low efficiency (but not zero - at least tried)
        return 0.2