"""
LangGraph-based ART agent with OpenPipe RL integration for competitive programming.
"""
from typing import Dict, List, Any, Optional, Annotated
import asyncio
from datetime import datetime
import uuid

# Production imports - no fallbacks
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from pydantic import BaseModel

# ART integration - production mode
try:
    import art
    from art import TrajectoryGroup
    ART_AVAILABLE = True
except ImportError:
    raise ImportError("ART framework required for production. Install with: pip install openpipe-art")

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from .ojbench_tools import create_ojbench_tools
from .competitive_programming_trajectory import CompetitiveProgrammingTrajectory

from models.qwen_local_interface import Qwen3LocalInterface

from utils.code_parser import CodeParser


class AgentState(BaseModel):
    """State for the competitive programming agent."""
    messages: Annotated[list, add_messages]
    problem_id: str
    problem: Dict[str, Any]
    current_attempt: int
    max_attempts: int
    solution_history: List[Dict]
    session_id: str
    optimized_prompt: str
    
    class Config:
        arbitrary_types_allowed = True


class LangGraphARTAgent:
    """
    Production LangGraph agent for competitive programming with ART methodology.
    No fallbacks - requires proper setup.
    """
    
    def __init__(self, 
                 model_config: Dict[str, Any],
                 max_attempts: int = 3):
        
        self.model_interface = Qwen3LocalInterface(model_config)
        self.parser = CodeParser()
        self.max_attempts = max_attempts
        
        # Initialize tools
        self.tools = create_ojbench_tools()
        
        # Create the agent graph
        self.graph = self._create_agent_graph()
        
        # Trajectory storage for RL training
        self.trajectories: List[CompetitiveProgrammingTrajectory] = []
    
    def _create_agent_graph(self) -> StateGraph:
        """Create the LangGraph agent workflow."""
        
        # Use local Qwen model for all operations
        self.chat_model = self.model_interface
        
        # Create system prompt for competitive programming
        system_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert competitive programming AI agent with access to OJBench evaluation tools.

Your task is to solve competitive programming problems using the ART (Automatic Reasoning and Tool-use) methodology:

1. **ANALYZE**: First, use get_problem_details to understand the problem thoroughly
2. **REASON**: Think step by step about the algorithm and approach using <think> blocks
3. **TOOL-USE**: Use compile_and_test_solution to validate your approach iteratively  
4. **SUBMIT**: Use submit_solution only when confident in your final solution

Key principles:
- Always think in <think>...</think> blocks to show your reasoning
- Use tools strategically - test early and often
- Consider edge cases, time/space complexity
- If a solution fails, analyze the feedback and improve
- Choose the most appropriate language (C++ preferred for performance)

Available tools: {tool_descriptions}

Current attempt: {current_attempt}/{max_attempts}
"""),
            MessagesPlaceholder("messages")
        ])
        
        # Create workflow graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("analyze_problem", self._analyze_problem_node)
        workflow.add_node("generate_solution", self._generate_solution_node) 
        workflow.add_node("test_solution", self._test_solution_node)
        workflow.add_node("refine_solution", self._refine_solution_node)
        workflow.add_node("submit_solution", self._submit_solution_node)
        
        # Add edges
        workflow.set_entry_point("analyze_problem")
        workflow.add_edge("analyze_problem", "generate_solution")
        workflow.add_edge("generate_solution", "test_solution")
        
        # Conditional edges based on test results
        workflow.add_conditional_edges(
            "test_solution",
            self._decide_next_action,
            {
                "submit": "submit_solution",
                "refine": "refine_solution", 
                "retry": "generate_solution",
                "end": END
            }
        )
        
        workflow.add_edge("refine_solution", "test_solution")
        workflow.add_edge("submit_solution", END)
        
        # Compile graph
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)
    
    async def _analyze_problem_node(self, state: AgentState) -> AgentState:
        """Analyze the problem using tools."""
        
        # Use get_problem_details tool
        problem_tool = self.tools[1]  # get_problem_details
        problem_details = problem_tool._run(state.problem_id)
        
        analysis_message = HumanMessage(
            content=f"""
Problem ID: {state.problem_id}
Problem Details: {problem_details}

Please analyze this competitive programming problem. Think about:
1. What is the core problem asking for?
2. What algorithm or data structure would be most appropriate?
3. What are the time/space complexity requirements?
4. What edge cases should be considered?

Use <think>...</think> blocks to show your reasoning.
"""
        )
        
        state.messages.append(analysis_message)
        
        # Generate analysis
        response = await self._call_model(state)
        state.messages.append(AIMessage(content=response))
        
        return state
    
    async def _generate_solution_node(self, state: AgentState) -> AgentState:
        """Generate a code solution."""
        
        generation_prompt = HumanMessage(
            content=f"""
Based on your analysis, generate a complete solution for this problem.

Requirements:
- Choose the most appropriate language (C++ preferred for performance)
- Include all necessary headers/imports
- Handle input/output correctly
- Consider time and space complexity
- Add brief comments explaining key logic

Use <think>...</think> blocks to explain your approach before coding.

This is attempt {state.current_attempt} of {state.max_attempts}.
"""
        )
        
        state.messages.append(generation_prompt)
        
        # Generate solution
        response = await self._call_model(state)
        state.messages.append(AIMessage(content=response))
        
        return state
    
    async def _test_solution_node(self, state: AgentState) -> AgentState:
        """Test the generated solution using OJBench tools."""
        
        # Extract code from the last response
        last_message = state.messages[-1].content
        
        try:
            language, code = self.parser.get_main_solution(last_message)
        except ValueError as e:
            # No code found, ask for code
            error_message = HumanMessage(
                content=f"I couldn't find a complete solution in your response. Error: {str(e)}. Please provide a complete code solution."
            )
            state.messages.append(error_message)
            return state
        
        # Test using compile_and_test_solution tool
        test_tool = self.tools[0]  # compile_and_test_solution
        test_result = test_tool._run(state.problem_id, code, language)
        
        # Record attempt in solution history
        attempt_record = {
            "attempt_number": state.current_attempt,
            "language": language,
            "code": code,
            "test_result": test_result,
            "timestamp": datetime.now().isoformat()
        }
        state.solution_history.append(attempt_record)
        
        # Add test result to conversation
        test_message = HumanMessage(
            content=f"""
Test Result for Attempt {state.current_attempt}:
{test_result}

Please analyze this result and determine next steps.
"""
        )
        state.messages.append(test_message)
        
        return state
    
    async def _refine_solution_node(self, state: AgentState) -> AgentState:
        """Refine the solution based on test feedback."""
        
        last_attempt = state.solution_history[-1]
        test_result = last_attempt["test_result"]
        
        refinement_prompt = HumanMessage(
            content=f"""
Your solution failed with the following result:
{test_result}

Please:
1. Analyze what went wrong using <think>...</think> blocks
2. Identify the specific issue (algorithm, implementation, edge cases, etc.)
3. Propose a refined approach
4. Generate an improved solution

This is attempt {state.current_attempt} of {state.max_attempts}.
"""
        )
        
        state.messages.append(refinement_prompt)
        
        # Generate refinement
        response = await self._call_model(state)
        state.messages.append(AIMessage(content=response))
        
        return state
    
    async def _submit_solution_node(self, state: AgentState) -> AgentState:
        """Submit the final solution."""
        
        # Get the latest code
        last_attempt = state.solution_history[-1]
        language = last_attempt["language"] 
        code = last_attempt["code"]
        
        # Submit using submit_solution tool
        submit_tool = self.tools[2]  # submit_solution
        submission_result = submit_tool._run(state.problem_id, code, language)
        
        # Add submission result to conversation
        final_message = HumanMessage(
            content=f"Final Submission Result: {submission_result}"
        )
        state.messages.append(final_message)
        
        # Create trajectory for RL training
        self._create_trajectory(state, submission_result)
        
        return state
    
    def _decide_next_action(self, state: AgentState) -> str:
        """Decide next action based on test results."""
        
        if not state.solution_history:
            return "retry"
        
        last_attempt = state.solution_history[-1]
        test_result = last_attempt.get("test_result", {})
        
        # Check if solution passed
        if test_result.get("result", {}).get("success", False):
            return "submit"
        
        # Check if max attempts reached
        if state.current_attempt >= state.max_attempts:
            return "submit"  # Submit anyway for final evaluation
        
        # Check if we should refine or retry
        verdict = test_result.get("result", {}).get("verdict", "unknown")
        
        # Compilation errors - need refinement
        if verdict == "CE":
            state.current_attempt += 1
            return "refine"
        
        # Runtime/logic errors - need refinement  
        if verdict in ["WA", "RE", "TLE", "MLE"]:
            state.current_attempt += 1
            return "refine"
        
        # Default: retry
        state.current_attempt += 1
        return "retry"
    
    async def _call_model(self, state: AgentState) -> str:
        """Call the underlying model with current conversation."""
        
        # Convert messages to the format expected by our model interface
        conversation = self._format_conversation(state.messages)
        
        try:
            if hasattr(self.chat_model, 'generate'):
                response = self.chat_model.generate(conversation, max_tokens=6000)
            else:
                # Fallback for different interfaces
                response = str(self.chat_model.invoke(conversation))
            
            return response
            
        except Exception as e:
            return f"Model generation failed: {str(e)}. Please try a different approach."
    
    def _format_conversation(self, messages: List) -> str:
        """Format messages for our model interface."""
        
        formatted_parts = []
        
        for msg in messages:
            if hasattr(msg, 'content'):
                content = msg.content
                if hasattr(msg, 'type'):
                    msg_type = msg.type
                    if msg_type == 'system':
                        formatted_parts.append(f"SYSTEM: {content}")
                    elif msg_type == 'human':
                        formatted_parts.append(f"USER: {content}")
                    elif msg_type == 'ai':
                        formatted_parts.append(f"ASSISTANT: {content}")
                    else:
                        formatted_parts.append(content)
                else:
                    formatted_parts.append(content)
            else:
                formatted_parts.append(str(msg))
        
        return "\n\n".join(formatted_parts)
    
    def _create_trajectory(self, state: AgentState, submission_result: Dict):
        """Create a trajectory for RL training."""
        
        if not state.solution_history:
            return
        
        last_attempt = state.solution_history[-1]
        
        # Create trajectory from the final attempt
        trajectory = CompetitiveProgrammingTrajectory(
            problem_id=state.problem_id,
            problem_difficulty=state.problem.get("difficulty", "unknown"),
            problem_dataset=state.problem.get("dataset", "unknown"), 
            problem_statement=state.problem.get("prompt", ""),
            language=last_attempt["language"],
            generated_code=last_attempt["code"],
            think_blocks=self.parser.extract_think_blocks("\n".join([str(msg.content) for msg in state.messages])),
            verdict=submission_result.get("submission_result", {}).get("verdict", "unknown"),
            success=submission_result.get("submission_result", {}).get("success", False),
            execution_time=submission_result.get("submission_result", {}).get("execution_time"),
            memory_usage=submission_result.get("submission_result", {}).get("memory_usage"),
            test_cases_passed=submission_result.get("submission_result", {}).get("test_cases_passed", 0),
            total_test_cases=submission_result.get("submission_result", {}).get("total_test_cases", 0),
            error_analysis=submission_result.get("submission_result", {}).get("error_analysis"),
            correction_suggestions=[],
            internal_reasoning_quality=0.5,  # Would be computed by RULER
            generation_time=5.0,  # Estimated, would be tracked
            evaluation_time=2.0,  # Estimated, would be tracked
            total_solve_time=60.0,  # Estimated, would be tracked
            attempt_number=state.current_attempt,
            max_attempts=state.max_attempts,
            prompt_optimization_score=None,
            tool_usage_efficiency=0.8,  # Would be computed based on tool usage pattern
            reasoning_coherence=0.7,  # Would be computed by analyzing think blocks
            session_id=state.session_id,
            timestamp=datetime.now(),
            model_version="qwen3-4b-thinking"
        )
        
        self.trajectories.append(trajectory)
    
    async def solve_problem(self, problem: Dict, optimized_prompt: str = "") -> Dict:
        """
        Solve a single problem using the LangGraph agent.
        """
        
        session_id = str(uuid.uuid4())
        
        # Create initial state
        initial_state = AgentState(
            messages=[],
            problem_id=problem["id"],
            problem=problem,
            current_attempt=1,
            max_attempts=self.max_attempts,
            solution_history=[],
            session_id=session_id,
            optimized_prompt=optimized_prompt
        )
        
        try:
            # Run the agent workflow
            config = {"configurable": {"thread_id": session_id}}
            final_state = await self.graph.ainvoke(initial_state, config)
            
            # Extract results
            success = False
            final_result = None
            
            if final_state.solution_history:
                last_submission = final_state.solution_history[-1]
                test_result = last_submission.get("test_result", {})
                success = test_result.get("result", {}).get("success", False)
                final_result = test_result.get("result", {}).get("verdict", "unknown")
            
            return {
                "problem_id": problem["id"],
                "success": success,
                "final_result": final_result,
                "attempts": len(final_state.solution_history),
                "solution_history": final_state.solution_history,
                "session_id": session_id,
                "total_time": 120.0  # Estimated, would be tracked
            }
            
        except Exception as e:
            print(f"Error solving problem {problem['id']}: {str(e)}")
            return {
                "problem_id": problem["id"],
                "success": False,
                "final_result": "error",
                "error": str(e),
                "attempts": 0,
                "solution_history": [],
                "session_id": session_id,
                "total_time": 0.0
            }
    
    def get_trajectories(self) -> List[CompetitiveProgrammingTrajectory]:
        """Get collected trajectories for RL training."""
        return self.trajectories
    
    def clear_trajectories(self):
        """Clear trajectory history."""
        self.trajectories.clear()