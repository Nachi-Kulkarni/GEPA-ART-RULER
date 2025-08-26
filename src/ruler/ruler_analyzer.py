from dataclasses import dataclass
from typing import List, Dict, Optional
import re

@dataclass
class ErrorDiagnosis:
    error_type: str
    specific_issue: str
    think_analysis: str
    correction_strategy: str
    urgency: int  # 1=critical, 2=important, 3=minor

class RULERAnalyzer:
    def __init__(self):
        """Initialize RULER error analysis system"""
        self.error_patterns = self._build_error_patterns()
    
    def _build_error_patterns(self) -> Dict:
        """Define error detection patterns"""
        return {
            "complexity_issues": [
                (r"o\(n\^?2\)|nested loop|brute force", r"10\^5|10\^6|100000|1000000"),
                (r"o\(n\^?3\)|triple.*loop", r"10\^3|1000")
            ],
            "algorithm_red_flags": [
                (r"bubble sort|selection sort", r"n.*>.*1000"),
                (r"recursive.*factorial", r"n.*>.*1000")
            ],
            "implementation_risks": [
                (r"int.*array\[.*\]", r"10\^6|1000000"),  # Large int arrays
                (r"vector.*vector", r"matrix|grid"),       # 2D vectors
                (r"string.*concatenation", r"loop|for|while")  # String concat in loops
            ]
        }
    
    def analyze_think_blocks(self, think_blocks: List[str]) -> Dict:
        """Analyze the AI's internal reasoning for potential issues"""
        
        analysis = {
            "complexity_warnings": [],
            "algorithm_concerns": [],
            "missing_considerations": [],
            "logic_issues": []
        }
        
        combined_thinking = " ".join(think_blocks).lower()
        
        # Check for complexity issues
        for complexity_pattern, constraint_pattern in self.error_patterns["complexity_issues"]:
            if (re.search(complexity_pattern, combined_thinking) and 
                re.search(constraint_pattern, combined_thinking)):
                
                analysis["complexity_warnings"].append({
                    "issue": f"Potential TLE: {complexity_pattern} algorithm with large constraints",
                    "severity": "high"
                })
        
        # Check for missing edge case considerations
        edge_case_keywords = ["edge case", "corner case", "empty", "zero", "negative", 
                             "overflow", "underflow", "maximum", "minimum"]
        
        if not any(keyword in combined_thinking for keyword in edge_case_keywords):
            analysis["missing_considerations"].append({
                "issue": "No explicit edge case analysis mentioned",
                "severity": "medium"
            })
        
        # Check for algorithm appropriateness
        for algo_pattern, constraint_pattern in self.error_patterns["algorithm_red_flags"]:
            if (re.search(algo_pattern, combined_thinking) and 
                re.search(constraint_pattern, combined_thinking)):
                
                analysis["algorithm_concerns"].append({
                    "issue": f"Inefficient algorithm choice: {algo_pattern}",
                    "severity": "high"
                })
        
        return analysis
    
    def analyze_execution_error(self, evaluation_result: Dict) -> ErrorDiagnosis:
        """Analyze execution errors from OJBench"""
        
        verdict = evaluation_result.get("verdict", "UNKNOWN")
        
        # Get detailed error information
        detailed_results = evaluation_result.get("detailed_results", [])
        error_details = ""
        
        if detailed_results:
            failed_test = next((r for r in detailed_results 
                              if r.get("readable_main_code") != "AC"), None)
            if failed_test:
                error_details = failed_test.get("feedback", "")
        
        # Create diagnosis based on verdict type
        if verdict == "WA":
            return self._diagnose_wrong_answer(error_details, evaluation_result)
        elif verdict == "TLE":
            return self._diagnose_time_limit(error_details, evaluation_result)
        elif verdict == "MLE":
            return self._diagnose_memory_limit(error_details, evaluation_result)
        elif verdict == "RE":
            return self._diagnose_runtime_error(error_details, evaluation_result)
        elif verdict == "CE":
            return self._diagnose_compilation_error(error_details, evaluation_result)
        else:
            return ErrorDiagnosis(
                error_type="unknown",
                specific_issue=f"Unknown verdict: {verdict}",
                think_analysis="Unable to analyze unknown error type",
                correction_strategy="Review solution logic and implementation",
                urgency=2
            )
    
    def _diagnose_wrong_answer(self, error_details: str, result: Dict) -> ErrorDiagnosis:
        """Diagnose Wrong Answer errors"""
        
        # Common WA patterns
        if "expected" in error_details.lower() and "got" in error_details.lower():
            issue = "Output format or value mismatch"
        elif any(word in error_details.lower() for word in ["boundary", "edge", "corner"]):
            issue = "Edge case handling error"
        else:
            issue = "Logic or algorithm error"
        
        return ErrorDiagnosis(
            error_type="logic_error",
            specific_issue=f"Wrong Answer: {issue}",
            think_analysis="Review algorithmic approach and edge case handling",
            correction_strategy="Debug with small test cases and trace through logic",
            urgency=1
        )
    
    def _diagnose_time_limit(self, error_details: str, result: Dict) -> ErrorDiagnosis:
        """Diagnose Time Limit Exceeded errors"""
        
        return ErrorDiagnosis(
            error_type="performance_error", 
            specific_issue="Algorithm complexity too high",
            think_analysis="Current algorithm is too slow for the given constraints",
            correction_strategy="Find more efficient algorithm or optimize current approach",
            urgency=1
        )
    
    def _diagnose_memory_limit(self, error_details: str, result: Dict) -> ErrorDiagnosis:
        """Diagnose Memory Limit Exceeded errors"""
        
        return ErrorDiagnosis(
            error_type="memory_error",
            specific_issue="Memory usage exceeds limits", 
            think_analysis="Data structures or algorithm uses too much memory",
            correction_strategy="Optimize data structures or use memory-efficient approach",
            urgency=1
        )
    
    def _diagnose_runtime_error(self, error_details: str, result: Dict) -> ErrorDiagnosis:
        """Diagnose Runtime Error"""
        
        common_causes = {
            "segmentation fault": "Array bounds violation or null pointer access",
            "stack overflow": "Infinite recursion or too deep recursion",
            "floating point": "Division by zero or invalid math operation",
            "assertion": "Assertion failure in code"
        }
        
        specific_cause = "General runtime error"
        for pattern, cause in common_causes.items():
            if pattern in error_details.lower():
                specific_cause = cause
                break
        
        return ErrorDiagnosis(
            error_type="runtime_error",
            specific_issue=specific_cause,
            think_analysis="Code execution failed due to illegal operation",
            correction_strategy="Check array bounds, pointer usage, and input validation",
            urgency=1
        )
    
    def _diagnose_compilation_error(self, error_details: str, result: Dict) -> ErrorDiagnosis:
        """Diagnose Compilation Error"""
        
        return ErrorDiagnosis(
            error_type="syntax_error",
            specific_issue="Code compilation failed",
            think_analysis="Syntax errors or missing includes prevent compilation",
            correction_strategy="Fix syntax errors and ensure all headers are included",
            urgency=1
        )
    
    def create_correction_guidance(self, think_analysis: Dict, 
                                 execution_diagnosis: ErrorDiagnosis,
                                 previous_code: str) -> str:
        """Create comprehensive correction guidance"""
        
        guidance = f"""
COMPREHENSIVE ERROR ANALYSIS:

🧠 REASONING ANALYSIS:
"""
        
        # Add think block analysis
        if think_analysis["complexity_warnings"]:
            guidance += "\n⚠️  COMPLEXITY ISSUES DETECTED:\n"
            for warning in think_analysis["complexity_warnings"]:
                guidance += f"- {warning['issue']}\n"
        
        if think_analysis["missing_considerations"]:
            guidance += "\n🔍 MISSING CONSIDERATIONS:\n" 
            for consideration in think_analysis["missing_considerations"]:
                guidance += f"- {consideration['issue']}\n"
        
        # Add execution analysis
        guidance += f"""
🚨 EXECUTION ERROR:
- Type: {execution_diagnosis.error_type}
- Issue: {execution_diagnosis.specific_issue}
- Strategy: {execution_diagnosis.correction_strategy}

🎯 CORRECTION FOCUS:
Based on the analysis above, here's what you need to fix:

1. {execution_diagnosis.correction_strategy}
2. Address any complexity issues identified
3. Consider edge cases that might have been missed
4. Verify your algorithm logic step by step

DEBUGGING APPROACH:
- Trace through your algorithm with a small example
- Check boundary conditions (empty input, single element, maximum values)
- Verify time/space complexity meets problem constraints
- Test your logic on edge cases

Now provide your corrected solution:
"""
        
        return guidance