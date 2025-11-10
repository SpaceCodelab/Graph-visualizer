"""
Streamlit-based interactive UI for Graph Coloring Solver.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.graph import Graph
from src.graph_generator import GraphGenerator
from src.algorithms import ColoringSolver
from src.visualizer import Visualizer
from src.benchmarker import Benchmarker
from src.graph_properties import GraphProperties
from src.applications import (
    RegisterAllocation, ExamScheduling, MapColoring,
    FrequencyAssignment, SudokuSolver
)


# Page configuration
st.set_page_config(
    page_title="Graph Coloring Solver",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'graph' not in st.session_state:
    st.session_state.graph = None
if 'coloring_result' not in st.session_state:
    st.session_state.coloring_result = None
if 'benchmark_results' not in st.session_state:
    st.session_state.benchmark_results = None


def main():
    """Main application."""
    st.markdown('<h1 class="main-header">🎨 Graph Coloring Solver</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Home", "Graph Input", "Graph Generator", "Coloring Algorithms", 
         "Benchmarking", "Real-World Applications"]
    )
    
    if page == "Home":
        show_home()
    elif page == "Graph Input":
        show_graph_input()
    elif page == "Graph Generator":
        show_graph_generator()
    elif page == "Coloring Algorithms":
        show_coloring_algorithms()
    elif page == "Benchmarking":
        show_benchmarking()
    elif page == "Real-World Applications":
        show_applications()


def show_home():
    """Home page."""
    st.markdown("""
    ## Welcome to Graph Coloring Solver
    
    This comprehensive application implements and compares multiple graph coloring algorithms
    to solve the vertex coloring problem optimally or near-optimally.
    
    ### Features:
    - **Multiple Algorithms**: Greedy (largest-first, smallest-last, random), Welsh-Powell, 
      Backtracking, DSATUR
    - **Graph Input**: Support for adjacency matrix, adjacency list, edge list, and file loading
    - **Graph Generator**: Create various test graphs (complete, bipartite, cyclic, planar, random, etc.)
    - **Visualization**: Interactive graph visualization with color coding
    - **Benchmarking**: Compare algorithm performance
    - **Real-World Applications**: Register allocation, exam scheduling, map coloring, etc.
    
    ### Getting Started:
    1. Go to **Graph Input** or **Graph Generator** to create/load a graph
    2. Use **Coloring Algorithms** to solve the coloring problem
    3. Use **Benchmarking** to compare different algorithms
    4. Explore **Real-World Applications** for practical use cases
    
    ### Algorithm Complexity:
    - **Greedy**: O(V + E) - Fast but not optimal
    - **Welsh-Powell**: O(V²) - Better than greedy
    - **DSATUR**: O(V²) - Good heuristic
    - **Backtracking**: O(V * k^V) - Exact but exponential
    """)


def show_graph_input():
    """Graph input page."""
    st.markdown('<h2 class="sub-header">Graph Input</h2>', unsafe_allow_html=True)
    
    input_method = st.radio(
        "Input Method",
        ["Edge List", "Adjacency List", "Adjacency Matrix", "File Upload"]
    )
    
    graph = None
    
    if input_method == "Edge List":
        st.subheader("Edge List Input")
        edges_text = st.text_area(
            "Enter edges (one per line, format: vertex1 vertex2)",
            value="0 1\n1 2\n2 3\n3 0\n0 2"
        )
        if st.button("Load Graph"):
            try:
                edges = []
                for line in edges_text.strip().split('\n'):
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            try:
                                u = int(parts[0])
                                v = int(parts[1])
                            except ValueError:
                                u = parts[0]
                                v = parts[1]
                            edges.append((u, v))
                graph = Graph.from_edge_list(edges)
                st.session_state.graph = graph
                st.success(f"Graph loaded: {graph.num_vertices()} vertices, {graph.num_edges()} edges")
            except Exception as e:
                st.error(f"Error loading graph: {e}")
    
    elif input_method == "Adjacency List":
        st.subheader("Adjacency List Input")
        adj_text = st.text_area(
            "Enter adjacency list (JSON format)",
            value='{"0": [1, 2], "1": [0, 2], "2": [0, 1, 3], "3": [2]}'
        )
        if st.button("Load Graph"):
            try:
                import json
                adj_dict = json.loads(adj_text)
                # Convert string keys to integers if possible, and normalize neighbor values
                normalized_dict = {}
                for k, v in adj_dict.items():
                    # Convert key
                    key = int(k) if isinstance(k, str) and k.isdigit() else k
                    # Convert neighbor values
                    neighbors = []
                    for neighbor in v:
                        if isinstance(neighbor, str) and neighbor.isdigit():
                            neighbors.append(int(neighbor))
                        else:
                            neighbors.append(neighbor)
                    normalized_dict[key] = neighbors
                graph = Graph.from_adjacency_list(normalized_dict)
                st.session_state.graph = graph
                st.success(f"Graph loaded: {graph.num_vertices()} vertices, {graph.num_edges()} edges")
            except Exception as e:
                st.error(f"Error loading graph: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    elif input_method == "Adjacency Matrix":
        st.subheader("Adjacency Matrix Input")
        matrix_text = st.text_area(
            "Enter adjacency matrix (one row per line, space-separated)",
            value="0 1 1 0\n1 0 1 0\n1 1 0 1\n0 0 1 0"
        )
        if st.button("Load Graph"):
            try:
                rows = []
                for line in matrix_text.strip().split('\n'):
                    if line.strip():
                        row = [int(x) for x in line.strip().split()]
                        rows.append(row)
                
                if not rows:
                    raise ValueError("Matrix is empty")
                
                # Validate that matrix is square
                n = len(rows)
                for i, row in enumerate(rows):
                    if len(row) != n:
                        raise ValueError(f"Row {i+1} has {len(row)} elements, expected {n} (matrix must be square)")
                
                matrix = np.array(rows)
                graph = Graph.from_adjacency_matrix(matrix)
                st.session_state.graph = graph
                st.success(f"Graph loaded: {graph.num_vertices()} vertices, {graph.num_edges()} edges")
            except Exception as e:
                st.error(f"Error loading graph: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    elif input_method == "File Upload":
        st.subheader("File Upload")
        uploaded_file = st.file_uploader("Upload graph file (CSV or JSON)", type=['csv', 'json'])
        if uploaded_file is not None:
            try:
                import tempfile
                import os
                # Save to temporary file with proper path handling
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    temp_path = tmp_file.name
                
                try:
                    graph = Graph.from_file(temp_path)
                    st.session_state.graph = graph
                    st.success(f"Graph loaded: {graph.num_vertices()} vertices, {graph.num_edges()} edges")
                finally:
                    # Clean up
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
            except Exception as e:
                st.error(f"Error loading graph: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    # Display graph if loaded
    if st.session_state.graph is not None:
        graph = st.session_state.graph
        st.subheader("Graph Visualization")
        visualizer = Visualizer(graph)
        fig = visualizer.visualize_uncolored_plotly("Loaded Graph")
        st.plotly_chart(fig, use_container_width=True)
        
        # Graph properties
        st.subheader("Graph Properties")
        try:
            properties = GraphProperties(graph)
            props = properties.get_all_properties()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Vertices", props.get('num_vertices', 0))
                st.metric("Edges", props.get('num_edges', 0))
            with col2:
                st.metric("Max Degree", props.get('max_degree', 0))
                st.metric("Density", f"{props.get('density', 0):.3f}")
            with col3:
                st.metric("Connected", "Yes" if props.get('is_connected', False) else "No")
                st.metric("Bipartite", "Yes" if props.get('is_bipartite', False) else "No")
            with col4:
                st.metric("Clique Number", props.get('clique_number', 0))
                chrom_lower = props.get('chromatic_lower_bound', 0)
                chrom_upper = props.get('chromatic_upper_bound', 0)
                st.metric("Chromatic Bounds", f"{chrom_lower}-{chrom_upper}")
            
            # Additional properties in expander
            with st.expander("More Properties"):
                col5, col6 = st.columns(2)
                with col5:
                    st.write(f"**Min Degree:** {props.get('min_degree', 0)}")
                    st.write(f"**Average Degree:** {props.get('average_degree', 0):.3f}")
                    st.write(f"**Is Tree:** {'Yes' if props.get('is_tree', False) else 'No'}")
                    st.write(f"**Is Planar:** {'Yes' if props.get('is_planar', False) else 'No'}")
                with col6:
                    st.write(f"**Independence Number:** {props.get('independence_number', 0)}")
                    st.write(f"**Number of Components:** {props.get('num_components', 1)}")
                    chrom_poly = props.get('chromatic_polynomial')
                    if chrom_poly:
                        st.write(f"**Chromatic Polynomial:** {chrom_poly.get('type', 'N/A')}")
                        if 'formula' in chrom_poly:
                            st.write(f"Formula: {chrom_poly['formula']}")
        except Exception as e:
            st.error(f"Error calculating graph properties: {e}")
            import traceback
            st.code(traceback.format_exc())


def show_graph_generator():
    """Graph generator page."""
    st.markdown('<h2 class="sub-header">Graph Generator</h2>', unsafe_allow_html=True)
    
    graph_type = st.selectbox(
        "Graph Type",
        ["Complete", "Bipartite", "Cycle", "Path", "Planar", "Random", 
         "Random by Density", "Petersen", "Wheel", "Star", "Grid", "Tree", "Regular"]
    )
    
    generator = GraphGenerator()
    graph = None
    
    if graph_type == "Complete":
        n = st.slider("Number of vertices", 2, 20, 5)
        if st.button("Generate"):
            try:
                graph = generator.complete_graph(n)
            except Exception as e:
                st.error(f"Error generating complete graph: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    elif graph_type == "Bipartite":
        n1 = st.slider("Vertices in partition 1", 2, 10, 3)
        n2 = st.slider("Vertices in partition 2", 2, 10, 3)
        p = st.slider("Edge probability", 0.0, 1.0, 1.0)
        if st.button("Generate"):
            try:
                graph = generator.bipartite_graph(n1, n2, p)
            except Exception as e:
                st.error(f"Error generating bipartite graph: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    elif graph_type == "Cycle":
        n = st.slider("Number of vertices", 3, 20, 5)
        if st.button("Generate"):
            try:
                graph = generator.cycle_graph(n)
            except Exception as e:
                st.error(f"Error generating cycle graph: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    elif graph_type == "Path":
        n = st.slider("Number of vertices", 2, 20, 5)
        if st.button("Generate"):
            try:
                graph = generator.path_graph(n)
            except Exception as e:
                st.error(f"Error generating path graph: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    elif graph_type == "Planar":
        n = st.slider("Number of vertices", 3, 20, 10)
        if st.button("Generate"):
            try:
                graph = generator.planar_graph(n)
            except Exception as e:
                st.error(f"Error generating planar graph: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    elif graph_type == "Random":
        n = st.slider("Number of vertices", 3, 20, 10)
        p = st.slider("Edge probability", 0.0, 1.0, 0.5)
        if st.button("Generate"):
            try:
                graph = generator.random_graph(n, p)
            except Exception as e:
                st.error(f"Error generating random graph: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    elif graph_type == "Random by Density":
        n = st.slider("Number of vertices", 3, 20, 10)
        density = st.slider("Density", 0.0, 1.0, 0.5)
        if st.button("Generate"):
            try:
                graph = generator.random_graph_by_density(n, density)
            except Exception as e:
                st.error(f"Error generating random graph by density: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    elif graph_type == "Petersen":
        if st.button("Generate"):
            try:
                graph = generator.petersen_graph()
            except Exception as e:
                st.error(f"Error generating Petersen graph: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    elif graph_type == "Wheel":
        n = st.slider("Cycle size", 3, 20, 5)
        if st.button("Generate"):
            try:
                graph = generator.wheel_graph(n)
            except Exception as e:
                st.error(f"Error generating wheel graph: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    elif graph_type == "Star":
        n = st.slider("Number of leaves", 2, 20, 5)
        if st.button("Generate"):
            try:
                graph = generator.star_graph(n)
            except Exception as e:
                st.error(f"Error generating star graph: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    elif graph_type == "Grid":
        m = st.slider("Rows", 2, 10, 3)
        n = st.slider("Columns", 2, 10, 3)
        if st.button("Generate"):
            try:
                graph = generator.grid_graph(m, n)
            except Exception as e:
                st.error(f"Error generating grid graph: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    elif graph_type == "Tree":
        n = st.slider("Number of vertices", 3, 20, 10)
        if st.button("Generate"):
            try:
                graph = generator.tree_graph(n)
            except Exception as e:
                st.error(f"Error generating tree graph: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    elif graph_type == "Regular":
        n = st.slider("Number of vertices", 3, 20, 10)
        # Calculate max degree: must be less than n and n*d must be even
        max_degree = min(10, n - 1) if n > 1 else 2
        if max_degree < 2:
            st.warning("Cannot create regular graph with these parameters")
        else:
            d = st.slider("Degree", 2, max_degree, min(3, max_degree))
            if st.button("Generate"):
                try:
                    # Validate: n * d must be even for regular graphs
                    if (n * d) % 2 != 0:
                        st.error(f"Cannot create {d}-regular graph with {n} vertices: n*d must be even. Try adjusting the degree.")
                    elif d >= n:
                        st.error(f"Degree {d} must be less than number of vertices {n}")
                    else:
                        graph = generator.regular_graph(n, d)
                except Exception as e:
                    st.error(f"Error generating regular graph: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    
    if graph is not None:
        try:
            st.session_state.graph = graph
            st.success(f"Graph generated: {graph.num_vertices()} vertices, {graph.num_edges()} edges")
            
            # Visualize
            visualizer = Visualizer(graph)
            fig = visualizer.visualize_uncolored_plotly(f"{graph_type} Graph")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error displaying graph: {e}")
            import traceback
            st.code(traceback.format_exc())


def show_coloring_algorithms():
    """Coloring algorithms page."""
    st.markdown('<h2 class="sub-header">Coloring Algorithms</h2>', unsafe_allow_html=True)
    
    if st.session_state.graph is None:
        st.warning("Please load or generate a graph first.")
        return
    
    graph = st.session_state.graph
    
    algorithm = st.selectbox(
        "Select Algorithm",
        ["greedy_largest_first", "greedy_smallest_last", "greedy_random",
         "welsh_powell", "backtracking", "dsatur"]
    )
    
    if st.button("Solve"):
        solver = ColoringSolver(graph)
        with st.spinner("Solving..."):
            result = solver.solve(algorithm)
            st.session_state.coloring_result = result
        
        if result.get('is_valid'):
            st.success(f"Solution found using {result['num_colors']} colors in {result['execution_time']:.4f} seconds")
            
            # Visualize
            visualizer = Visualizer(graph)
            fig = visualizer.visualize_coloring_plotly(
                result['coloring'],
                f"Graph Coloring - {algorithm}"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show coloring
            st.subheader("Coloring Assignment")
            coloring_df = pd.DataFrame([
                {"Vertex": v, "Color": c}
                for v, c in result['coloring'].items()
            ])
            st.dataframe(coloring_df, use_container_width=True)
        else:
            st.error(f"Error: {result.get('error', 'Unknown error')}")


def show_benchmarking():
    """Benchmarking page."""
    st.markdown('<h2 class="sub-header">Algorithm Benchmarking</h2>', unsafe_allow_html=True)
    
    if st.session_state.graph is None:
        st.warning("Please load or generate a graph first.")
        return
    
    graph = st.session_state.graph
    
    # Show graph properties first
    st.subheader("Graph Properties")
    try:
        properties = GraphProperties(graph)
        props = properties.get_all_properties()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Vertices", props.get('num_vertices', 0))
            st.metric("Edges", props.get('num_edges', 0))
        with col2:
            st.metric("Max Degree", props.get('max_degree', 0))
            st.metric("Density", f"{props.get('density', 0):.3f}")
        with col3:
            st.metric("Connected", "Yes" if props.get('is_connected', False) else "No")
            st.metric("Bipartite", "Yes" if props.get('is_bipartite', False) else "No")
        with col4:
            st.metric("Clique Number", props.get('clique_number', 0))
            chrom_lower = props.get('chromatic_lower_bound', 0)
            chrom_upper = props.get('chromatic_upper_bound', 0)
            st.metric("Chromatic Bounds", f"{chrom_lower}-{chrom_upper}")
    except Exception as e:
        st.error(f"Error calculating graph properties: {e}")
    
    if st.button("Run Benchmark"):
        try:
            benchmarker = Benchmarker(graph)
            with st.spinner("Running benchmarks... This may take a moment."):
                df = benchmarker.benchmark_all()
                st.session_state.benchmark_results = df
                st.session_state.benchmarker = benchmarker
            
            st.subheader("Results")
            
            # Display results with better formatting
            display_df = df.copy()
            # Format execution time
            if 'execution_time' in display_df.columns:
                display_df['execution_time'] = display_df['execution_time'].apply(
                    lambda x: f"{x:.6f}" if pd.notna(x) and x is not None else "N/A"
                )
            # Format num_colors
            if 'num_colors' in display_df.columns:
                display_df['num_colors'] = display_df['num_colors'].apply(
                    lambda x: int(x) if pd.notna(x) and x is not None else "N/A"
                )
            
            st.dataframe(display_df, use_container_width=True)
            
            # Show summary
            valid_results = df[df['is_valid'] == True]
            if len(valid_results) > 0:
                best_result = valid_results.loc[valid_results['num_colors'].idxmin()]
                st.success(f"Best result: {best_result['algorithm']} with {int(best_result['num_colors'])} colors")
            
            # Plot comparison
            try:
                fig = benchmarker.plot_comparison_plotly()
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not generate comparison plot: {e}")
            
            # Export options
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Export to CSV"):
                    try:
                        import os
                        filepath = "benchmark_results.csv"
                        benchmarker.export_to_csv(filepath)
                        if os.path.exists(filepath):
                            st.success(f"Exported to {filepath}")
                            with open(filepath, 'rb') as f:
                                st.download_button(
                                    label="Download CSV",
                                    data=f.read(),
                                    file_name=filepath,
                                    mime="text/csv"
                                )
                    except Exception as e:
                        st.error(f"Error exporting to CSV: {e}")
            with col2:
                if st.button("Export to PDF"):
                    try:
                        import os
                        filepath = "benchmark_results.pdf"
                        benchmarker.export_to_pdf(filepath)
                        if os.path.exists(filepath):
                            st.success(f"Exported to {filepath}")
                            with open(filepath, 'rb') as f:
                                st.download_button(
                                    label="Download PDF",
                                    data=f.read(),
                                    file_name=filepath,
                                    mime="application/pdf"
                                )
                    except Exception as e:
                        st.error(f"Error exporting to PDF: {e}")
        except Exception as e:
            st.error(f"Error running benchmark: {e}")
            import traceback
            st.code(traceback.format_exc())


def show_applications():
    """Real-world applications page."""
    st.markdown('<h2 class="sub-header">Real-World Applications</h2>', unsafe_allow_html=True)
    
    app_type = st.selectbox(
        "Application Type",
        ["Register Allocation", "Exam Scheduling", "Map Coloring", 
         "Frequency Assignment", "Sudoku Solver"]
    )
    
    if app_type == "Register Allocation":
        st.subheader("Register Allocation")
        st.write("Simulate register allocation in compiler optimization.")
        
        variables = st.text_area("Variables (one per line)", value="a\nb\nc\nd")
        conflicts = st.text_area("Conflicts (one per line, format: var1 var2)", 
                                 value="a b\nb c\nc d\nd a")
        
        if st.button("Solve"):
            try:
                var_list = [v.strip() for v in variables.strip().split('\n') if v.strip()]
                conflict_list = []
                for line in conflicts.strip().split('\n'):
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            conflict_list.append((parts[0], parts[1]))
                
                app = RegisterAllocation(var_list, conflict_list)
                result = app.solve()
                
                st.success(f"Solution: {result['num_registers']} registers needed")
                st.json(result['registers'])
            except Exception as e:
                st.error(f"Error: {e}")
    
    elif app_type == "Exam Scheduling":
        st.subheader("Exam Scheduling")
        st.write("Schedule exams so that no student has two exams at the same time.")
        
        courses = st.text_area("Courses (one per line)", value="Math\nPhysics\nChemistry\nBiology")
        students_text = st.text_area("Students (JSON format: {\"student1\": [\"course1\", \"course2\"], ...})",
                                     value='{"Alice": ["Math", "Physics"], "Bob": ["Physics", "Chemistry"], "Charlie": ["Math", "Chemistry", "Biology"]}')
        
        if st.button("Solve"):
            try:
                import json
                course_list = [c.strip() for c in courses.strip().split('\n') if c.strip()]
                students_dict = json.loads(students_text)
                
                app = ExamScheduling(course_list, students_dict)
                result = app.solve()
                
                st.success(f"Solution: {result['num_time_slots']} time slots needed")
                st.json(result['schedule'])
            except Exception as e:
                st.error(f"Error: {e}")
    
    elif app_type == "Map Coloring":
        st.subheader("Map Coloring")
        st.write("Color a map so that adjacent regions have different colors.")
        
        regions = st.text_area("Regions (one per line)", value="A\nB\nC\nD")
        borders = st.text_area("Borders (one per line, format: region1 region2)",
                              value="A B\nB C\nC D\nD A\nA C")
        
        if st.button("Solve"):
            try:
                region_list = [r.strip() for r in regions.strip().split('\n') if r.strip()]
                border_list = []
                for line in borders.strip().split('\n'):
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            border_list.append((parts[0], parts[1]))
                
                app = MapColoring(region_list, border_list)
                result = app.solve()
                
                st.success(f"Solution: {result['num_colors']} colors needed")
                st.json(result['region_colors'])
            except Exception as e:
                st.error(f"Error: {e}")
    
    elif app_type == "Frequency Assignment":
        st.subheader("Frequency Assignment")
        st.write("Assign frequencies to wireless stations to avoid interference.")
        
        stations = st.text_area("Stations (one per line)", value="S1\nS2\nS3\nS4")
        interference = st.text_area("Interference (one per line, format: station1 station2)",
                                   value="S1 S2\nS2 S3\nS3 S4\nS4 S1")
        
        if st.button("Solve"):
            try:
                station_list = [s.strip() for s in stations.strip().split('\n') if s.strip()]
                interference_list = []
                for line in interference.strip().split('\n'):
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            interference_list.append((parts[0], parts[1]))
                
                app = FrequencyAssignment(station_list, interference_list)
                result = app.solve()
                
                st.success(f"Solution: {result['num_frequencies']} frequencies needed")
                st.json(result['frequencies'])
            except Exception as e:
                st.error(f"Error: {e}")
    
    elif app_type == "Sudoku Solver":
        st.subheader("Sudoku Solver")
        st.write("Solve Sudoku using backtracking (graph coloring concepts).")
        
        st.write("Enter initial Sudoku grid (9x9, use 0 for empty cells):")
        grid_input = st.text_area("Grid (9 rows, space-separated numbers)",
                                 value="5 3 0 0 7 0 0 0 0\n6 0 0 1 9 5 0 0 0\n0 9 8 0 0 0 0 6 0\n8 0 0 0 6 0 0 0 3\n4 0 0 8 0 3 0 0 1\n7 0 0 0 2 0 0 0 6\n0 6 0 0 0 0 2 8 0\n0 0 0 4 1 9 0 0 5\n0 0 0 0 8 0 0 7 9")
        
        if st.button("Solve"):
            try:
                rows = []
                for line in grid_input.strip().split('\n'):
                    if line.strip():
                        row = [int(x) for x in line.strip().split()]
                        rows.append(row)
                grid = np.array(rows)
                
                app = SudokuSolver(grid)
                solution = app.solve()
                
                if solution is not None:
                    st.success("Sudoku solved!")
                    st.dataframe(pd.DataFrame(solution), use_container_width=True)
                else:
                    st.error("No solution found")
            except Exception as e:
                st.error(f"Error: {e}")


if __name__ == "__main__":
    main()

