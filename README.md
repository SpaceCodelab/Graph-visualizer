# 🎨 Graph Visualizer & Coloring Toolkit

[![License](https://img.shields.io/github/license/SpaceCodelab/Graph-visualizer?style=for-the-badge)](LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/SpaceCodelab/Graph-visualizer?style=for-the-badge)](https://github.com/SpaceCodelab/Graph-visualizer)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue?style=for-the-badge)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-pytest-brightgreen?style=for-the-badge)](https://pytest.org/)
[![Open Issues](https://img.shields.io/github/issues/SpaceCodelab/Graph-visualizer?style=for-the-badge)](https://github.com/SpaceCodelab/Graph-visualizer/issues)

A modern toolkit for creating, visualizing, and coloring graphs. Includes multiple coloring algorithms (greedy variants, DSATUR, backtracking, ILP hooks), graph generators, benchmarking, and an interactive Streamlit UI.

---

<!-- NOTE: add a short GIF or screenshot to show the visualizer in action. Put the file in `assets/` and replace the path below. -->
![Demo placeholder](assets/demo-placeholder.png)

## ✨ Highlights

- Multiple coloring algorithms: Greedy (largest-first, smallest-last, random), Welsh–Powell, DSATUR, Backtracking (exact for small graphs)
- Graph generators: complete, bipartite, cycle, path, planar-ish, Erdős–Rényi, and special graphs
- Interactive visualizer (Plotly + Streamlit) and static export (Matplotlib)
- Benchmarking harness to compare runtime and quality across algorithms
- Command-line and programmatic APIs

## 🚀 Quickstart (Windows PowerShell)

Clone, create a venv, install, and run the Streamlit UI:

```powershell
git clone https://github.com/SpaceCodelab/Graph-visualizer.git "Graph Project"
cd "Graph Project"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Open the URL shown by Streamlit (usually http://localhost:8501).

Tip: on Windows Command Prompt use `.\.venv\Scripts\activate` instead of the PowerShell command.

## 🧩 Quick example (programmatic)

Use the library modules directly from Python scripts or REPL:

```python
from src.graph_generator import GraphGenerator
from src.algorithms import ColoringSolver

g = GraphGenerator().complete_graph(6)
solver = ColoringSolver(g)
res = solver.solve('dsatur')
print('colors used', res['num_colors'])
print(res['coloring'])
```

## 🛠 Development & Tests

Run the test suite (inside the activated virtualenv):

```powershell
pytest -q
```

Linting and static checks can be added via `flake8`/`ruff`/`mypy` if desired.

## 📂 Project Layout (high level)

```
Graph Project/
├── src/                 # core library code (graph, algorithms, visualizer)
├── tests/               # unit tests
├── app.py               # Streamlit UI
├── requirements.txt     # deps
└── README.md            # ← you are here
```

## 🤝 Contributing

Contributions are welcome. A good workflow:

1. Fork the repo
2. Create a feature branch
3. Add tests for new behavior
4. Open a pull request

Please follow the repository's coding style and include tests for new features.

## 📸 Screenshots / GIFs

Add compelling screenshots or a short demo GIF in `assets/` and reference them above. Example files:

- `assets/demo.gif` — animated demo of the visualizer
- `assets/screenshot-graph.png` — static screenshot for the project page

## 📌 Notes & Next Steps

- Consider adding a GitHub Action to run tests and show a build badge.
- Add an `assets/` folder with a 600×300 hero image or a short GIF for the repo landing page.
- If you publish to PyPI, add the PyPI badge and version badge.

## 📜 License

This project is open-source — see the `LICENSE` file for details.

---

Made with ❤️ by the SpaceCodelab team.
## Features

### Core Algorithms
- **Greedy Coloring** with multiple ordering strategies:
  - Largest-First: Order vertices by degree (descending)
  - Smallest-Last: Remove vertices with smallest degree iteratively
  - Random: Random vertex ordering
- **Welsh-Powell Algorithm**: Sorts vertices by degree and colors greedily
- **Backtracking Algorithm**: Finds exact chromatic number for smaller graphs
- **DSATUR (Degree of Saturation)**: Improved heuristic using saturation degrees

### Graph Input/Output
- **Multiple Input Formats**:
  - Adjacency matrix
  - Adjacency list
  - Edge list
  - File loading (CSV, JSON)
- **Graph Export**: Save graphs to CSV or JSON format

### Graph Generator
Create various types of test graphs:
- Complete graphs (K_n)
- Bipartite graphs
- Cycle and path graphs
- Planar graphs
- Random graphs (Erdős–Rényi model)
- Random graphs with specified density
- Special graphs: Petersen, Wheel, Star, Grid, Tree, Regular, Barbell, Lollipop

### Visualization
- **Interactive Visualization**: Using Plotly for interactive graph display
- **Static Visualization**: Using Matplotlib for publication-quality figures
- **Color Coding**: Distinct colors for different color classes
- **Graph Properties Display**: Shows graph statistics and properties

### Benchmarking
- **Performance Comparison**: Compare all algorithms on the same graph
- **Execution Time Analysis**: Measure and compare algorithm performance
- **Export Results**: Export benchmark results to CSV or PDF
- **Interactive Charts**: Plotly-based comparison charts

### Real-World Applications
- **Register Allocation**: Simulate register allocation in compiler optimization
- **Exam Scheduling**: Schedule exams so no student has conflicts
- **Map Coloring**: Color geographical regions with adjacent regions having different colors
- **Frequency Assignment**: Assign frequencies to wireless stations to avoid interference
- **Sudoku Solver**: Solve Sudoku puzzles using backtracking (graph coloring concepts)

### Advanced Features
- **Parallel Execution**: Run multiple algorithms in parallel
- **Weighted Coloring**: Color graphs where colors have associated costs
- **ILP Integration**: Solve using Integer Linear Programming (PuLP or OR-Tools)
- **Dynamic Recoloring**: Handle graph modifications (add/remove vertices/edges) with dynamic recoloring

### Graph Properties
- Chromatic number bounds (lower and upper)
- Clique number
- Independence number
- Chromatic polynomial approximation
- Connectivity, bipartiteness, planarity checks
- Graph density and degree statistics

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Setup

1. **Clone or download the project**:
   ```bash
   cd "Graph Project"
   ```

2. **Create a virtual environment**:
   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate
   
   # On Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Application

1. **Activate the virtual environment** (if not already activated):
   ```bash
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

2. **Run the Streamlit application**:
   ```bash
   streamlit run app.py
   ```

3. **Open your browser** to the URL shown in the terminal (usually `http://localhost:8501`)

### Using the Application

1. **Load or Generate a Graph**:
   - Go to "Graph Input" to load a graph from various formats
   - Go to "Graph Generator" to create test graphs

2. **Solve Coloring Problem**:
   - Go to "Coloring Algorithms"
   - Select an algorithm
   - Click "Solve" to get the coloring

3. **Benchmark Algorithms**:
   - Go to "Benchmarking"
   - Click "Run Benchmark" to compare all algorithms

4. **Explore Applications**:
   - Go to "Real-World Applications"
   - Select an application type
   - Enter your data and solve

### Command Line Usage

You can also use the modules programmatically:

```python
from src.graph import Graph
from src.graph_generator import GraphGenerator
from src.algorithms import ColoringSolver
from src.visualizer import Visualizer

# Generate a graph
generator = GraphGenerator()
graph = generator.complete_graph(5)

# Solve coloring
solver = ColoringSolver(graph)
result = solver.solve('dsatur')

print(f"Number of colors: {result['num_colors']}")
print(f"Coloring: {result['coloring']}")

# Visualize
visualizer = Visualizer(graph)
fig = visualizer.visualize_coloring_plotly(result['coloring'])
fig.show()
```

## Testing

Run the test suite using pytest:

```bash
# Activate virtual environment first
pytest tests/
```

Or run specific test files:

```bash
pytest tests/test_algorithms.py
pytest tests/test_graph.py
pytest tests/test_applications.py
```

## Project Structure

```
Graph Project/
├── src/
│   ├── __init__.py
│   ├── graph.py              # Graph data structure
│   ├── graph_generator.py    # Graph generators
│   ├── algorithms.py         # Coloring algorithms
│   ├── graph_properties.py   # Graph properties calculator
│   ├── visualizer.py         # Visualization tools
│   ├── benchmarker.py        # Benchmarking module
│   ├── applications.py       # Real-world applications
│   └── advanced.py           # Advanced features
├── tests/
│   ├── __init__.py
│   ├── test_graph.py
│   ├── test_algorithms.py
│   └── test_applications.py
├── app.py                    # Streamlit application
├── requirements.txt          # Python dependencies
├── setup_venv.bat           # Windows setup script
├── setup_venv.sh            # Linux/Mac setup script
├── .gitignore
└── README.md
```

## Algorithm Complexity

| Algorithm | Time Complexity | Space Complexity | Optimal |
|-----------|----------------|------------------|---------|
| Greedy (Largest-First) | O(V + E) | O(V) | No |
| Greedy (Smallest-Last) | O(V²) | O(V) | No |
| Welsh-Powell | O(V²) | O(V) | No |
| DSATUR | O(V²) | O(V) | No |
| Backtracking | O(V * k^V) | O(V) | Yes (for small graphs) |
| ILP | Exponential (worst case) | O(V²) | Yes |

Where:
- V = number of vertices
- E = number of edges
- k = number of colors

## Best and Worst Cases

### Greedy Algorithms
- **Best Case**: Trees and bipartite graphs (2 colors)
- **Worst Case**: Complete graphs (n colors for n vertices)

### DSATUR
- **Best Case**: Similar to greedy, but often uses fewer colors
- **Worst Case**: Still not optimal, but better than simple greedy

### Backtracking
- **Best Case**: Small graphs with low chromatic number
- **Worst Case**: Large graphs (exponential time)

## Academic References

1. Welsh, D. J. A., & Powell, M. B. (1967). An upper bound for the chromatic number of a graph and its application to timetabling problems. *The Computer Journal*, 10(1), 85-86.

2. Brélaz, D. (1979). New methods to color the vertices of a graph. *Communications of the ACM*, 22(4), 251-256.

3. Chaitin, G. J. (1982). Register allocation & spilling via graph coloring. *ACM SIGPLAN Notices*, 17(6), 98-105.

4. Appel, K., & Haken, W. (1977). Every planar map is four colorable. *Illinois Journal of Mathematics*, 21(3), 429-567.

5. Garey, M. R., & Johnson, D. S. (1979). *Computers and Intractability: A Guide to the Theory of NP-Completeness*. W. H. Freeman.

## Error Handling

The application includes comprehensive error handling for:
- Invalid input formats
- Disconnected graphs
- Self-loops (not allowed in simple graphs)
- Multigraphs (handled appropriately)
- Invalid algorithm parameters
- File I/O errors

All errors provide informative messages and suggestions for correction.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is open source and available for educational and research purposes.

## Acknowledgments

- NetworkX for graph data structures and algorithms
- Matplotlib and Plotly for visualization
- Streamlit for the interactive web interface
- All contributors to the open-source libraries used in this project

## Future Enhancements

Potential future improvements:
- GPU acceleration for large graphs
- More approximation algorithms with guaranteed bounds
- Support for directed graphs
- Graph isomorphism checking
- Chromatic polynomial exact calculation for small graphs
- Integration with more optimization libraries
- Mobile-friendly UI
- Cloud deployment support

## Features

### Core Algorithms
- **Greedy Coloring** with multiple ordering strategies:
  - Largest-First: Order vertices by degree (descending)
  - Smallest-Last: Remove vertices with smallest degree iteratively
  - Random: Random vertex ordering
- **Welsh-Powell Algorithm**: Sorts vertices by degree and colors greedily
- **Backtracking Algorithm**: Finds exact chromatic number for smaller graphs
- **DSATUR (Degree of Saturation)**: Improved heuristic using saturation degrees

### Graph Input/Output
- **Multiple Input Formats**:
  - Adjacency matrix
  - Adjacency list
  - Edge list
  - File loading (CSV, JSON)
- **Graph Export**: Save graphs to CSV or JSON format

### Graph Generator
Create various types of test graphs:
- Complete graphs (K_n)
- Bipartite graphs
- Cycle and path graphs
- Planar graphs
- Random graphs (Erdős–Rényi model)
- Random graphs with specified density
- Special graphs: Petersen, Wheel, Star, Grid, Tree, Regular, Barbell, Lollipop

### Visualization
- **Interactive Visualization**: Using Plotly for interactive graph display
- **Static Visualization**: Using Matplotlib for publication-quality figures
- **Color Coding**: Distinct colors for different color classes
- **Graph Properties Display**: Shows graph statistics and properties

### Benchmarking
- **Performance Comparison**: Compare all algorithms on the same graph
- **Execution Time Analysis**: Measure and compare algorithm performance
- **Export Results**: Export benchmark results to CSV or PDF
- **Interactive Charts**: Plotly-based comparison charts

### Real-World Applications
- **Register Allocation**: Simulate register allocation in compiler optimization
- **Exam Scheduling**: Schedule exams so no student has conflicts
- **Map Coloring**: Color geographical regions with adjacent regions having different colors
- **Frequency Assignment**: Assign frequencies to wireless stations to avoid interference
- **Sudoku Solver**: Solve Sudoku puzzles using backtracking (graph coloring concepts)

### Advanced Features
- **Parallel Execution**: Run multiple algorithms in parallel
- **Weighted Coloring**: Color graphs where colors have associated costs
- **ILP Integration**: Solve using Integer Linear Programming (PuLP or OR-Tools)
- **Dynamic Recoloring**: Handle graph modifications (add/remove vertices/edges) with dynamic recoloring

### Graph Properties
- Chromatic number bounds (lower and upper)
- Clique number
- Independence number
- Chromatic polynomial approximation
- Connectivity, bipartiteness, planarity checks
- Graph density and degree statistics

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Setup

1. **Clone or download the project**:
   ```bash
   cd "Graph Project"
   ```

2. **Create a virtual environment**:
   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate
   
   # On Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Application

1. **Activate the virtual environment** (if not already activated):
   ```bash
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

2. **Run the Streamlit application**:
   ```bash
   streamlit run app.py
   ```

3. **Open your browser** to the URL shown in the terminal (usually `http://localhost:8501`)

### Using the Application

1. **Load or Generate a Graph**:
   - Go to "Graph Input" to load a graph from various formats
   - Go to "Graph Generator" to create test graphs

2. **Solve Coloring Problem**:
   - Go to "Coloring Algorithms"
   - Select an algorithm
   - Click "Solve" to get the coloring

3. **Benchmark Algorithms**:
   - Go to "Benchmarking"
   - Click "Run Benchmark" to compare all algorithms

4. **Explore Applications**:
   - Go to "Real-World Applications"
   - Select an application type
   - Enter your data and solve

### Command Line Usage

You can also use the modules programmatically:

```python
from src.graph import Graph
from src.graph_generator import GraphGenerator
from src.algorithms import ColoringSolver
from src.visualizer import Visualizer

# Generate a graph
generator = GraphGenerator()
graph = generator.complete_graph(5)

# Solve coloring
solver = ColoringSolver(graph)
result = solver.solve('dsatur')

print(f"Number of colors: {result['num_colors']}")
print(f"Coloring: {result['coloring']}")

# Visualize
visualizer = Visualizer(graph)
fig = visualizer.visualize_coloring_plotly(result['coloring'])
fig.show()
```

## Testing

Run the test suite using pytest:

```bash
# Activate virtual environment first
pytest tests/
```

Or run specific test files:

```bash
pytest tests/test_algorithms.py
pytest tests/test_graph.py
pytest tests/test_applications.py
```

## Project Structure

```
Graph Project/
├── src/
│   ├── __init__.py
│   ├── graph.py              # Graph data structure
│   ├── graph_generator.py    # Graph generators
│   ├── algorithms.py         # Coloring algorithms
│   ├── graph_properties.py   # Graph properties calculator
│   ├── visualizer.py         # Visualization tools
│   ├── benchmarker.py        # Benchmarking module
│   ├── applications.py       # Real-world applications
│   └── advanced.py           # Advanced features
├── tests/
│   ├── __init__.py
│   ├── test_graph.py
│   ├── test_algorithms.py
│   └── test_applications.py
├── app.py                    # Streamlit application
├── requirements.txt          # Python dependencies
├── setup_venv.bat           # Windows setup script
├── setup_venv.sh            # Linux/Mac setup script
├── .gitignore
└── README.md
```

## Algorithm Complexity

| Algorithm | Time Complexity | Space Complexity | Optimal |
|-----------|----------------|------------------|---------|
| Greedy (Largest-First) | O(V + E) | O(V) | No |
| Greedy (Smallest-Last) | O(V²) | O(V) | No |
| Welsh-Powell | O(V²) | O(V) | No |
| DSATUR | O(V²) | O(V) | No |
| Backtracking | O(V * k^V) | O(V) | Yes (for small graphs) |
| ILP | Exponential (worst case) | O(V²) | Yes |

Where:
- V = number of vertices
- E = number of edges
- k = number of colors

## Best and Worst Cases

### Greedy Algorithms
- **Best Case**: Trees and bipartite graphs (2 colors)
- **Worst Case**: Complete graphs (n colors for n vertices)

### DSATUR
- **Best Case**: Similar to greedy, but often uses fewer colors
- **Worst Case**: Still not optimal, but better than simple greedy

### Backtracking
- **Best Case**: Small graphs with low chromatic number
- **Worst Case**: Large graphs (exponential time)

## Academic References

1. Welsh, D. J. A., & Powell, M. B. (1967). An upper bound for the chromatic number of a graph and its application to timetabling problems. *The Computer Journal*, 10(1), 85-86.

2. Brélaz, D. (1979). New methods to color the vertices of a graph. *Communications of the ACM*, 22(4), 251-256.

3. Chaitin, G. J. (1982). Register allocation & spilling via graph coloring. *ACM SIGPLAN Notices*, 17(6), 98-105.

4. Appel, K., & Haken, W. (1977). Every planar map is four colorable. *Illinois Journal of Mathematics*, 21(3), 429-567.

5. Garey, M. R., & Johnson, D. S. (1979). *Computers and Intractability: A Guide to the Theory of NP-Completeness*. W. H. Freeman.

## Error Handling

The application includes comprehensive error handling for:
- Invalid input formats
- Disconnected graphs
- Self-loops (not allowed in simple graphs)
- Multigraphs (handled appropriately)
- Invalid algorithm parameters
- File I/O errors

All errors provide informative messages and suggestions for correction.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is open source and available for educational and research purposes.

## Acknowledgments

- NetworkX for graph data structures and algorithms
- Matplotlib and Plotly for visualization
- Streamlit for the interactive web interface
- All contributors to the open-source libraries used in this project

## Future Enhancements

Potential future improvements:
- GPU acceleration for large graphs
- More approximation algorithms with guaranteed bounds
- Support for directed graphs
- Graph isomorphism checking
- Chromatic polynomial exact calculation for small graphs
- Integration with more optimization libraries
- Mobile-friendly UI
- Cloud deployment support

