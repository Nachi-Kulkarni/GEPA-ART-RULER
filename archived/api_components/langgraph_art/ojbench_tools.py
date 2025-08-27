"""
LangGraph tools for OJBench evaluation.
Converts our existing OJBench interface into LangGraph-compatible tools.
"""
from typing import Dict, Any, List, Optional
try:
    from langchain_core.tools import BaseTool
except ImportError:
    # Fallback for environments without LangChain
    class BaseTool:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
        def _run(self, *args, **kwargs):
            return {"status": "error", "message": "LangChain not available"}
from pydantic import BaseModel, Field
import json

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from evaluation.real_ojbench import OJBenchEvaluator

from utils.code_parser import CodeParser


class CompileAndTestInput(BaseModel):
    """Input schema for compile_and_test_solution tool."""
    problem_id: str = Field(description="Problem ID to test against")
    code: str = Field(description="Source code to test")
    language: str = Field(description="Programming language (cpp or python)")


class GetProblemDetailsInput(BaseModel):
    """Input schema for get_problem_details tool."""
    problem_id: str = Field(description="Problem ID to get details for")


class SubmitSolutionInput(BaseModel):
    """Input schema for submit_solution tool."""
    problem_id: str = Field(description="Problem ID to submit to")
    code: str = Field(description="Final solution code")
    language: str = Field(description="Programming language (cpp or python)")


class CompileAndTestTool(BaseTool):
    """LangGraph tool for compiling and testing code solutions."""
    
    name: str = "compile_and_test_solution"
    description: str = """
    Compile and test a code solution against a specific problem.
    Returns detailed results including verdict, execution time, memory usage, and test case results.
    Use this tool to validate your solution before final submission.
    """
    args_schema: type[BaseModel] = CompileAndTestInput
    
    def __init__(self):
        super().__init__()
        # Initialize evaluator outside of Pydantic validation
        object.__setattr__(self, 'evaluator', OJBenchEvaluator())
    
    def _run(self, problem_id: str, code: str, language: str) -> Dict[str, Any]:
        """Execute the tool."""
        try:
            result = self.evaluator.evaluate_solution(problem_id, code, language)
            
            # Format result for LangGraph agent
            formatted_result = {
                "success": result.get("success", False),
                "verdict": result.get("verdict", "unknown"),
                "execution_time": result.get("execution_time"),
                "memory_usage": result.get("memory_usage"),
                "test_cases_passed": result.get("test_cases_passed", 0),
                "total_test_cases": result.get("total_test_cases", 0),
                "error_message": result.get("error_message", ""),
                "feedback": result.get("feedback", ""),
                "detailed_results": result.get("detailed_results", [])
            }
            
            return {
                "status": "success",
                "result": formatted_result,
                "message": f"Evaluation complete. Verdict: {result.get('verdict', 'unknown')}"
            }
            
        except Exception as e:
            return {
                "status": "error", 
                "error": str(e),
                "message": f"Failed to evaluate solution: {str(e)}"
            }


class GetProblemDetailsTool(BaseTool):
    """LangGraph tool for retrieving problem information."""
    
    name: str = "get_problem_details"
    description: str = """
    Get detailed information about a competitive programming problem.
    Returns problem statement, constraints, sample inputs/outputs, and metadata.
    Use this to understand the problem requirements before coding.
    """
    args_schema: type[BaseModel] = GetProblemDetailsInput
    
    def __init__(self):
        super().__init__()
        object.__setattr__(self, 'evaluator', OJBenchEvaluator())
    
    def _run(self, problem_id: str) -> Dict[str, Any]:
        """Execute the tool."""
        try:
            # Get problem details from evaluator
            problem_info = self.evaluator.get_problem_info(problem_id)
            
            if problem_info:
                return {
                    "status": "success",
                    "problem": problem_info,
                    "message": f"Retrieved details for problem {problem_id}"
                }
            else:
                return {
                    "status": "error",
                    "error": f"Problem {problem_id} not found",
                    "message": f"Could not find problem {problem_id}"
                }
                
        except Exception as e:
            return {
                "status": "error",
                "error": str(e), 
                "message": f"Failed to retrieve problem details: {str(e)}"
            }


class SubmitSolutionTool(BaseTool):
    """LangGraph tool for final solution submission."""
    
    name: str = "submit_solution"  
    description: str = """
    Submit your final solution for a competitive programming problem.
    This performs a complete evaluation and returns the final verdict.
    Only use this when you're confident in your solution.
    """
    args_schema: type[BaseModel] = SubmitSolutionInput
    
    def __init__(self):
        super().__init__()
        object.__setattr__(self, 'evaluator', OJBenchEvaluator())
        object.__setattr__(self, 'parser', CodeParser())
    
    def _run(self, problem_id: str, code: str, language: str) -> Dict[str, Any]:
        """Execute the tool."""
        try:
            # Validate and clean the code
            try:
                cleaned_code = self.parser.clean_code(code, language)
            except Exception as parse_error:
                return {
                    "status": "error",
                    "error": f"Code parsing failed: {str(parse_error)}",
                    "message": "Please check your code syntax"
                }
            
            # Perform final evaluation
            result = self.evaluator.evaluate_solution(problem_id, cleaned_code, language)
            
            # Format comprehensive result
            final_result = {
                "problem_id": problem_id,
                "language": language,
                "success": result.get("success", False),
                "verdict": result.get("verdict", "unknown"),
                "execution_time": result.get("execution_time"),
                "memory_usage": result.get("memory_usage"),
                "test_cases_passed": result.get("test_cases_passed", 0),
                "total_test_cases": result.get("total_test_cases", 0),
                "score": result.get("score", 0.0),
                "detailed_feedback": result.get("feedback", ""),
                "error_analysis": result.get("error_analysis", {}),
                "submission_timestamp": result.get("timestamp")
            }
            
            if result.get("success", False):
                message = f"✅ SUCCESS! Problem {problem_id} solved with verdict {result.get('verdict', 'AC')}"
            else:
                message = f"❌ FAILED: Problem {problem_id} failed with verdict {result.get('verdict', 'unknown')}"
            
            return {
                "status": "success",
                "submission_result": final_result,
                "message": message
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "message": f"Submission failed: {str(e)}"
            }


class OJBenchToolkit:
    """Toolkit containing all OJBench-related tools for LangGraph agents."""
    
    def __init__(self):
        self.compile_and_test = CompileAndTestTool()
        self.get_problem_details = GetProblemDetailsTool()
        self.submit_solution = SubmitSolutionTool()
    
    def get_tools(self) -> List[BaseTool]:
        """Get all tools as a list for LangGraph agent initialization."""
        return [
            self.compile_and_test,
            self.get_problem_details, 
            self.submit_solution
        ]
    
    def get_tool_names(self) -> List[str]:
        """Get list of tool names."""
        return [tool.name for tool in self.get_tools()]
    
    def describe_tools(self) -> str:
        """Get a description of all available tools."""
        descriptions = []
        for tool in self.get_tools():
            descriptions.append(f"- {tool.name}: {tool.description.strip()}")
        
        return "Available OJBench Tools:\n" + "\n".join(descriptions)


# Convenience function for easy integration
def create_ojbench_tools() -> List[BaseTool]:
    """Create and return all OJBench tools for LangGraph agent."""
    toolkit = OJBenchToolkit()
    return toolkit.get_tools()