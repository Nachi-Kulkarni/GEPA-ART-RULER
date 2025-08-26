"""Mock model interface for testing and development without GPU"""

from typing import Optional
import random

class MockModelInterface:
    """Mock model interface that simulates Qwen3-4B responses for testing"""
    
    def __init__(self, model_name: str = "mock-qwen3-4B"):
        self.model_name = model_name
        self.max_context = 131072
        print(f"🤖 Mock model '{model_name}' initialized")
    
    def generate(self, prompt: str, max_tokens: int = 1024, 
                temperature: float = 0.6) -> str:
        """Generate mock response simulating competitive programming solutions"""
        
        # Simulate thinking process
        thinking_templates = [
            "Let me analyze this step by step.",
            "I need to understand the problem constraints first.",
            "This looks like a dynamic programming problem.",
            "I should consider the time complexity requirements.",
            "Let me think about edge cases here.",
        ]
        
        # Code templates for different problem types
        cpp_templates = [
            """#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int main() {
    int n;
    cin >> n;
    
    // Solution logic here
    cout << n * 2 << endl;
    
    return 0;
}""",
            
            """#include <iostream>
#include <string>
using namespace std;

int main() {
    string s;
    getline(cin, s);
    
    // Process string
    cout << s.length() << endl;
    
    return 0;
}""",
            
            """#include <iostream>
#include <vector>
using namespace std;

int main() {
    int n;
    cin >> n;
    vector<int> arr(n);
    
    for(int i = 0; i < n; i++) {
        cin >> arr[i];
    }
    
    // Process array
    int sum = 0;
    for(int x : arr) {
        sum += x;
    }
    
    cout << sum << endl;
    return 0;
}"""
        ]
        
        # Simulate response with thinking and code
        thinking = random.choice(thinking_templates)
        code = random.choice(cpp_templates)
        
        # Add some variability to make it look realistic
        if "constraints" in prompt.lower():
            thinking += " The constraints suggest I need an O(n) solution."
        if "array" in prompt.lower():
            code = cpp_templates[2]  # Array processing template
        
        response = f"""
${thinking}

Looking at this problem, I need to:
1. Read the input carefully
2. Process according to requirements  
3. Output the result in correct format

Let me implement this:
$

```cpp
{code}
```

This solution should handle the given constraints efficiently.
"""
        
        return response.strip()
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)"""
        return len(text.split())