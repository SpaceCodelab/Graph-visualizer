"""
Verify that all dependencies are installed correctly.
"""

import sys

def check_imports():
    """Check if all required modules can be imported."""
    print("Checking imports...")
    
    required_modules = [
        'networkx',
        'matplotlib',
        'plotly',
        'numpy',
        'pandas',
        'streamlit',
        'pytest',
    ]
    
    optional_modules = [
        'pulp',
        'ortools',
        'scipy',
        'PIL',
        'reportlab',
    ]
    
    all_ok = True
    
    print("\nRequired modules:")
    for module in required_modules:
        try:
            __import__(module)
            print(f"  [OK] {module}")
        except ImportError:
            print(f"  [MISSING] {module} - MISSING")
            all_ok = False
    
    print("\nOptional modules:")
    for module in optional_modules:
        try:
            __import__(module)
            print(f"  [OK] {module}")
        except ImportError:
            print(f"  [WARN] {module} - Optional (some features may not work)")
    
    print("\nChecking project modules...")
    try:
        from src.graph import Graph
        from src.graph_generator import GraphGenerator
        from src.algorithms import ColoringSolver
        from src.visualizer import Visualizer
        from src.benchmarker import Benchmarker
        from src.graph_properties import GraphProperties
        print("  [OK] All project modules imported successfully")
    except ImportError as e:
        print(f"  [ERROR] Error importing project modules: {e}")
        all_ok = False
    
    return all_ok


def test_basic_functionality():
    """Test basic functionality."""
    print("\nTesting basic functionality...")
    
    try:
        from src.graph_generator import GraphGenerator
        from src.algorithms import ColoringSolver
        
        # Generate a simple graph
        generator = GraphGenerator()
        graph = generator.complete_graph(5)
        print(f"  [OK] Generated graph: {graph.num_vertices()} vertices, {graph.num_edges()} edges")
        
        # Solve coloring
        solver = ColoringSolver(graph)
        result = solver.solve('dsatur')
        print(f"  [OK] Solved coloring: {result['num_colors']} colors")
        print(f"  [OK] Coloring is valid: {result['is_valid']}")
        
        return True
    except Exception as e:
        print(f"  [ERROR] Error in basic functionality: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Graph Coloring Solver - Setup Verification")
    print("=" * 60)
    
    imports_ok = check_imports()
    functionality_ok = test_basic_functionality()
    
    print("\n" + "=" * 60)
    if imports_ok and functionality_ok:
        print("[SUCCESS] Setup verification PASSED")
        print("=" * 60)
        print("\nYou can now run:")
        print("  - streamlit run app.py (for web interface)")
        print("  - python example.py (for examples)")
        print("  - pytest (for tests)")
        sys.exit(0)
    else:
        print("[FAILED] Setup verification FAILED")
        print("=" * 60)
        print("\nPlease install missing dependencies:")
        print("  pip install -r requirements.txt")
        sys.exit(1)

