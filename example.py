"""
Example usage of Graph Coloring Solver.
"""

from src.graph import Graph
from src.graph_generator import GraphGenerator
from src.algorithms import ColoringSolver
from src.visualizer import Visualizer
from src.benchmarker import Benchmarker
from src.graph_properties import GraphProperties
from src.applications import RegisterAllocation, ExamScheduling


def example_basic_coloring():
    """Basic graph coloring example."""
    print("=" * 60)
    print("Example 1: Basic Graph Coloring")
    print("=" * 60)
    
    # Generate a complete graph
    generator = GraphGenerator()
    graph = generator.complete_graph(5)
    
    print(f"Graph: {graph.num_vertices()} vertices, {graph.num_edges()} edges")
    
    # Solve using DSATUR
    solver = ColoringSolver(graph)
    result = solver.solve('dsatur')
    
    print(f"\nAlgorithm: DSATUR")
    print(f"Number of colors: {result['num_colors']}")
    print(f"Execution time: {result['execution_time']:.4f} seconds")
    print(f"Coloring: {result['coloring']}")
    print(f"Valid: {result['is_valid']}")


def example_benchmarking():
    """Benchmarking example."""
    print("\n" + "=" * 60)
    print("Example 2: Algorithm Benchmarking")
    print("=" * 60)
    
    # Generate a random graph
    generator = GraphGenerator()
    graph = generator.random_graph(10, 0.5)
    
    print(f"Graph: {graph.num_vertices()} vertices, {graph.num_edges()} edges")
    
    # Benchmark all algorithms
    benchmarker = Benchmarker(graph)
    df = benchmarker.benchmark_all()
    
    print("\nBenchmark Results:")
    print(df.to_string(index=False))


def example_graph_properties():
    """Graph properties example."""
    print("\n" + "=" * 60)
    print("Example 3: Graph Properties")
    print("=" * 60)
    
    # Generate a cycle graph
    generator = GraphGenerator()
    graph = generator.cycle_graph(5)
    
    # Calculate properties
    properties = GraphProperties(graph)
    props = properties.get_all_properties()
    
    print(f"Graph: {graph.num_vertices()} vertices, {graph.num_edges()} edges")
    print(f"\nProperties:")
    print(f"  Max Degree: {props['max_degree']}")
    print(f"  Connected: {props['is_connected']}")
    print(f"  Bipartite: {props['is_bipartite']}")
    print(f"  Clique Number: {props['clique_number']}")
    print(f"  Chromatic Bounds: {props['chromatic_lower_bound']}-{props['chromatic_upper_bound']}")


def example_register_allocation():
    """Register allocation example."""
    print("\n" + "=" * 60)
    print("Example 4: Register Allocation")
    print("=" * 60)
    
    # Define variables and conflicts
    variables = ['a', 'b', 'c', 'd', 'e']
    conflicts = [
        ('a', 'b'),
        ('b', 'c'),
        ('c', 'd'),
        ('d', 'e'),
        ('e', 'a'),
        ('a', 'c')
    ]
    
    # Solve register allocation
    app = RegisterAllocation(variables, conflicts)
    result = app.solve()
    
    print(f"Variables: {variables}")
    print(f"Conflicts: {conflicts}")
    print(f"\nSolution: {result['num_registers']} registers needed")
    print(f"Register Assignment:")
    for var, reg in result['registers'].items():
        print(f"  {var} -> {reg}")


def example_exam_scheduling():
    """Exam scheduling example."""
    print("\n" + "=" * 60)
    print("Example 5: Exam Scheduling")
    print("=" * 60)
    
    # Define courses and students
    courses = ['Math', 'Physics', 'Chemistry', 'Biology']
    students = {
        'Alice': ['Math', 'Physics'],
        'Bob': ['Physics', 'Chemistry'],
        'Charlie': ['Math', 'Chemistry', 'Biology'],
        'Diana': ['Biology', 'Math']
    }
    
    # Solve exam scheduling
    app = ExamScheduling(courses, students)
    result = app.solve()
    
    print(f"Courses: {courses}")
    print(f"Students: {students}")
    print(f"\nSolution: {result['num_time_slots']} time slots needed")
    print(f"Schedule:")
    for course, slot in result['schedule'].items():
        print(f"  {course} -> {slot}")


def example_visualization():
    """Visualization example."""
    print("\n" + "=" * 60)
    print("Example 6: Graph Visualization")
    print("=" * 60)
    
    # Generate a Petersen graph
    generator = GraphGenerator()
    graph = generator.petersen_graph()
    
    # Solve coloring
    solver = ColoringSolver(graph)
    result = solver.solve('dsatur')
    
    print(f"Graph: Petersen graph ({graph.num_vertices()} vertices)")
    print(f"Number of colors: {result['num_colors']}")
    
    # Visualize (uncomment to show plot)
    # visualizer = Visualizer(graph)
    # fig = visualizer.visualize_coloring_plotly(result['coloring'], "Petersen Graph Coloring")
    # fig.show()
    
    print("(Visualization code commented out - uncomment to show plot)")


if __name__ == "__main__":
    example_basic_coloring()
    example_benchmarking()
    example_graph_properties()
    example_register_allocation()
    example_exam_scheduling()
    example_visualization()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)

