# Quick Start Guide

## Installation

1. **Create and activate virtual environment**:
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate
   
   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

### Interactive Web Interface (Recommended)

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

### Command Line Examples

Run the example script:

```bash
python example.py
```

## Basic Usage

### 1. Generate a Graph

```python
from src.graph_generator import GraphGenerator

generator = GraphGenerator()
graph = generator.complete_graph(5)  # Complete graph with 5 vertices
```

### 2. Solve Coloring Problem

```python
from src.algorithms import ColoringSolver

solver = ColoringSolver(graph)
result = solver.solve('dsatur')  # Use DSATUR algorithm

print(f"Number of colors: {result['num_colors']}")
print(f"Coloring: {result['coloring']}")
```

### 3. Visualize Results

```python
from src.visualizer import Visualizer

visualizer = Visualizer(graph)
fig = visualizer.visualize_coloring_plotly(result['coloring'])
fig.show()
```

### 4. Benchmark Algorithms

```python
from src.benchmarker import Benchmarker

benchmarker = Benchmarker(graph)
df = benchmarker.benchmark_all()
print(df)
```

## Available Algorithms

- `greedy_largest_first`: Greedy with largest-first ordering
- `greedy_smallest_last`: Greedy with smallest-last ordering
- `greedy_random`: Greedy with random ordering
- `welsh_powell`: Welsh-Powell algorithm
- `backtracking`: Backtracking (exact, for small graphs)
- `dsatur`: DSATUR algorithm (recommended)

## Graph Types

Available graph generators:
- `complete_graph(n)`: Complete graph K_n
- `cycle_graph(n)`: Cycle graph C_n
- `bipartite_graph(n1, n2, p)`: Bipartite graph
- `random_graph(n, p)`: Random graph (Erdős–Rényi)
- `petersen_graph()`: Petersen graph
- `wheel_graph(n)`: Wheel graph
- And many more...

## Real-World Applications

### Register Allocation

```python
from src.applications import RegisterAllocation

variables = ['a', 'b', 'c']
conflicts = [('a', 'b'), ('b', 'c')]
app = RegisterAllocation(variables, conflicts)
result = app.solve()
print(result['registers'])
```

### Exam Scheduling

```python
from src.applications import ExamScheduling

courses = ['Math', 'Physics']
students = {'Alice': ['Math', 'Physics']}
app = ExamScheduling(courses, students)
result = app.solve()
print(result['schedule'])
```

## Testing

Run all tests:

```bash
pytest
```

Run specific test file:

```bash
pytest tests/test_algorithms.py
```

## Getting Help

- See `README.md` for comprehensive documentation
- Check `example.py` for usage examples
- Run `python example.py` to see examples in action

