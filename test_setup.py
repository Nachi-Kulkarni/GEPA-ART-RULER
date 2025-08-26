#!/usr/bin/env python3
"""Quick setup test for GEPA+ART+RULER system"""

import sys
import traceback

def test_imports():
    """Test critical imports"""
    print("🔍 Testing imports...")
    
    try:
        # Test core utilities
        from src.utils.code_parser import CodeParser
        parser = CodeParser()
        print("✅ CodeParser: OK")
        
        # Test mock evaluator
        from src.evaluation.mock_ojbench import MockOJBenchEvaluator
        evaluator = MockOJBenchEvaluator()
        print("✅ MockOJBenchEvaluator: OK")
        
        # Test basic AI components
        from src.prompts.base_prompt import BASE_PROMPT
        print("✅ BASE_PROMPT: OK")
        
        # Test GEPA components
        from src.gepa.evolution_engine import GEPAEvolutionEngine
        print("✅ GEPAEvolutionEngine: OK")
        
        # Test ART components
        from src.art.art_solver import ARTSolver
        print("✅ ARTSolver: OK")
        
        # Test RULER components
        from src.ruler.ruler_analyzer import RULERAnalyzer
        analyzer = RULERAnalyzer()
        print("✅ RULERAnalyzer: OK")
        
        print("🎉 All core imports successful!")
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        traceback.print_exc()
        return False

def test_basic_functionality():
    """Test basic functionality"""
    print("\n🔧 Testing basic functionality...")
    
    try:
        # Test CodeParser
        from src.utils.code_parser import CodeParser
        parser = CodeParser()
        
        sample_response = """
        $I need to solve this step by step$
        
        ```cpp
        #include <iostream>
        using namespace std;
        
        int main() {
            int n;
            cin >> n;
            cout << n * 2 << endl;
            return 0;
        }
        ```
        """
        
        think_blocks = parser.extract_think_blocks(sample_response)
        language, code = parser.get_main_solution(sample_response)
        
        print(f"✅ Extracted {len(think_blocks)} thinking blocks")
        print(f"✅ Extracted {language} code ({len(code)} chars)")
        
        # Test mock evaluation
        from src.evaluation.mock_ojbench import MockOJBenchEvaluator
        evaluator = MockOJBenchEvaluator()
        
        result = evaluator.evaluate_solution("test_problem", code, language)
        print(f"✅ Mock evaluation: {result['verdict']}")
        
        print("🎉 Basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Functionality error: {e}")
        traceback.print_exc()
        return False

def test_system_integration():
    """Test basic system integration"""
    print("\n🚀 Testing system integration...")
    
    try:
        # Mock model interface
        class MockModelInterface:
            def generate(self, prompt, max_tokens=1024):
                return """
                $Let me solve this step by step.
                This looks like a simple input/output problem.
                I need to read a number and double it.$
                
                ```cpp
                #include <iostream>
                using namespace std;
                
                int main() {
                    int n;
                    cin >> n;
                    cout << n * 2 << endl;
                    return 0;
                }
                ```
                """
        
        mock_model = MockModelInterface()
        
        # Test RULER analyzer
        from src.ruler.ruler_analyzer import RULERAnalyzer
        analyzer = RULERAnalyzer()
        
        # Test think block analysis
        think_blocks = ["I need to solve this step by step", "This is a simple problem"]
        analysis = analyzer.analyze_think_blocks(think_blocks)
        print(f"✅ RULER think analysis: {len(analysis.get('complexity_warnings', []))} warnings")
        
        # Test error diagnosis
        mock_error_result = {
            "success": False,
            "verdict": "WA",
            "detailed_results": []
        }
        diagnosis = analyzer.analyze_execution_error(mock_error_result)
        print(f"✅ RULER error diagnosis: {diagnosis.error_type}")
        
        print("🎉 System integration tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Integration error: {e}")
        traceback.print_exc()
        return False

def main():
    print("=" * 50)
    print("🧪 GEPA+ART+RULER System Setup Test")
    print("=" * 50)
    
    success = True
    
    success &= test_imports()
    success &= test_basic_functionality()  
    success &= test_system_integration()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 ALL TESTS PASSED! System is ready.")
        print("✅ You can now run the full system with:")
        print("   python main.py")
    else:
        print("❌ TESTS FAILED! Check setup requirements.")
        sys.exit(1)
    print("=" * 50)

if __name__ == "__main__":
    main()