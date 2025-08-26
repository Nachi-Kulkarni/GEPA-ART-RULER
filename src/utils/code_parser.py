"""
Code parsing utilities for extracting programming language code from model responses
"""
import re
from typing import Optional, List, Tuple

class CodeParser:
    """Parser for extracting code and reasoning from model responses"""
    
    def extract_think_blocks(self, text: str) -> List[str]:
        """Extract thinking blocks from model response"""
        # Look for $...$ blocks or <think>...</think> blocks
        think_patterns = [
            r'\$\s*(.*?)\s*\$',
            r'<think>\s*(.*?)\s*</think>',
            r'```thinking\n(.*?)\n```',
        ]
        
        think_blocks = []
        for pattern in think_patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            think_blocks.extend(matches)
        
        # Clean up the blocks
        cleaned_blocks = []
        for block in think_blocks:
            cleaned = block.strip()
            if cleaned and len(cleaned) > 10:  # Filter out very short blocks
                cleaned_blocks.append(cleaned)
        
        return cleaned_blocks
    
    def get_main_solution(self, text: str) -> Tuple[str, str]:
        """Extract the main solution code and determine language"""
        
        # Try C++ first (preferred for competitive programming)
        cpp_code = extract_cpp_code(text)
        if cpp_code:
            return "cpp", cpp_code
        
        # Try Python as fallback
        python_code = extract_python_code(text)
        if python_code:
            return "python", python_code
        
        # Try generic code extraction
        generic_matches = re.findall(r'```\w*\n(.*?)\n```', text, re.DOTALL)
        if generic_matches:
            code = max(generic_matches, key=len).strip()
            # Try to detect language from content
            if any(indicator in code for indicator in ['#include', 'int main', 'cout', 'cin']):
                return "cpp", code
            elif any(indicator in code for indicator in ['def ', 'import ', 'print(']):
                return "python", code
            else:
                return "cpp", code  # Default to C++ for competitive programming
        
        raise ValueError("No valid code found in response")

def extract_python_code(text: str) -> Optional[str]:
    """Extract Python code from markdown code blocks or other formats"""
    
    # First try to find code blocks with python/py markers
    python_patterns = [
        r'```python\n(.*?)\n```',
        r'```py\n(.*?)\n```',
        r'```\n(.*?)\n```',  # Generic code block
    ]
    
    for pattern in python_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            # Return the longest match (most complete code)
            return max(matches, key=len).strip()
    
    # Try to find code between triple backticks without language specifier
    generic_code = re.findall(r'```(.*?)```', text, re.DOTALL)
    if generic_code:
        return max(generic_code, key=len).strip()
    
    # Look for code that starts with def main() or similar patterns
    def_main_pattern = r'(def main\(\):.*?)(?=\n\n|\n*$)'
    def_main_matches = re.findall(def_main_pattern, text, re.DOTALL)
    if def_main_matches:
        return def_main_matches[0].strip()
    
    # Look for Python-like code patterns
    python_like_patterns = [
        r'(^import .*?(?=\n\n|\Z))',  # Code starting with imports
        r'(^from .*?(?=\n\n|\Z))',   # Code starting with from imports
        r'(^def .*?(?=\n\n|\Z))',    # Code starting with function definitions
    ]
    
    for pattern in python_like_patterns:
        matches = re.findall(pattern, text, re.MULTILINE | re.DOTALL)
        if matches:
            return max(matches, key=len).strip()
    
    return None

def extract_cpp_code(text: str) -> Optional[str]:
    """Extract C++ code from markdown code blocks or other formats"""
    
    # Try to find code blocks with cpp/c++ markers
    cpp_patterns = [
        r'```cpp\n(.*?)\n```',
        r'```c\+\+\n(.*?)\n```',
        r'```c\n(.*?)\n```',
        r'```\n(.*?)\n```',  # Generic code block
    ]
    
    for pattern in cpp_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return max(matches, key=len).strip()
    
    # Look for C++-like patterns
    cpp_like_patterns = [
        r'(#include.*?(?=\n\n|\Z))',  # Code starting with includes
        r'(^int main\(\).*?(?=\n\n|\Z))',  # Code with main function
    ]
    
    for pattern in cpp_like_patterns:
        matches = re.findall(pattern, text, re.MULTILINE | re.DOTALL)
        if matches:
            return max(matches, key=len).strip()
    
    return None

def extract_code_by_language(text: str, language: str) -> Optional[str]:
    """Extract code for a specific programming language"""
    
    if language.lower() in ['python', 'py']:
        return extract_python_code(text)
    elif language.lower() in ['cpp', 'c++', 'c']:
        return extract_cpp_code(text)
    else:
        # Generic extraction
        pattern = f'```{language.lower()}\n(.*?)\n```'
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return max(matches, key=len).strip()
        
        # Fallback to generic code blocks
        generic_matches = re.findall(r'```(.*?)```', text, re.DOTALL)
        if generic_matches:
            return max(generic_matches, key=len).strip()
    
    return None

def clean_code(code: str) -> str:
    """Clean extracted code by removing extra whitespace and fixing common issues"""
    if not code:
        return ""
    
    # Remove leading/trailing whitespace
    code = code.strip()
    
    # Fix common formatting issues
    code = re.sub(r'\n\s*\n\s*\n', '\n\n', code)  # Remove excessive blank lines
    
    return code

def validate_python_code(code: str) -> bool:
    """Basic validation to check if extracted code looks like valid Python"""
    if not code:
        return False
    
    # Check for basic Python patterns
    python_indicators = [
        r'def\s+\w+\s*\(',  # Function definitions
        r'if\s+__name__\s*==\s*["\']__main__["\']',  # Main guard
        r'import\s+\w+',  # Import statements
        r'from\s+\w+\s+import',  # From imports
        r'print\s*\(',  # Print statements
    ]
    
    for pattern in python_indicators:
        if re.search(pattern, code):
            return True
    
    return False

def validate_cpp_code(code: str) -> bool:
    """Basic validation to check if extracted code looks like valid C++"""
    if not code:
        return False
    
    # Check for basic C++ patterns
    cpp_indicators = [
        r'#include\s*<.*?>',  # Include statements
        r'int\s+main\s*\(',  # Main function
        r'std::',  # Standard namespace
        r'using\s+namespace',  # Using statements
        r'cout\s*<<',  # Output statements
        r'cin\s*>>',  # Input statements
    ]
    
    for pattern in cpp_indicators:
        if re.search(pattern, code):
            return True
    
    return False

def extract_and_validate_code(text: str, language: str) -> Optional[str]:
    """Extract and validate code for a specific language"""
    
    code = extract_code_by_language(text, language)
    if not code:
        return None
    
    code = clean_code(code)
    
    # Validate based on language
    if language.lower() in ['python', 'py']:
        if validate_python_code(code):
            return code
    elif language.lower() in ['cpp', 'c++', 'c']:
        if validate_cpp_code(code):
            return code
    else:
        # For other languages, return if we found any code
        return code if code else None
    
    return None

if __name__ == "__main__":
    # Test the code extraction
    test_text = """
    Here's a Python solution:
    
    ```python
    def main():
        n = int(input())
        print(n * 2)
    
    if __name__ == "__main__":
        main()
    ```
    
    This should work correctly.
    """
    
    extracted = extract_python_code(test_text)
    print("Extracted Python code:")
    print(extracted)
    print(f"Valid: {validate_python_code(extracted)}")