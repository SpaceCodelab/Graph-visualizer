"""
Advanced features: parallel execution, weighted coloring, ILP integration.
"""

from typing import Dict, List, Optional, Tuple
import concurrent.futures
import time
from .graph import Graph
from .algorithms import ColoringSolver


class ParallelSolver:
    """Parallel execution of multiple algorithms."""
    
    def __init__(self, graph: Graph, max_workers: Optional[int] = None):
        """
        Initialize parallel solver.
        
        Args:
            graph: Graph to color
            max_workers: Maximum number of parallel workers (None = CPU count)
        """
        self.graph = graph
        self.solver = ColoringSolver(graph)
        self.max_workers = max_workers
    
    def solve_parallel(self, algorithms: Optional[List[str]] = None) -> Dict[str, Dict]:
        """
        Solve using multiple algorithms in parallel.
        
        Args:
            algorithms: List of algorithm names (None = all)
            
        Returns:
            Dictionary of results
        """
        if algorithms is None or len(algorithms) == 0:
            algorithms = list(self.solver.ALGORITHMS.keys())
        
        # Filter out invalid algorithm names
        valid_algorithms = [alg for alg in algorithms if alg in self.solver.ALGORITHMS]
        
        if not valid_algorithms:
            return {}
        
        results = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_algorithm = {
                executor.submit(self.solver.solve, algorithm): algorithm
                for algorithm in valid_algorithms
            }
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_algorithm):
                algorithm = future_to_algorithm[future]
                try:
                    result = future.result()
                    # Ensure result is a dictionary
                    if isinstance(result, dict):
                        results[algorithm] = result
                    else:
                        results[algorithm] = {
                            'error': f'Unexpected result type: {type(result)}',
                            'algorithm': algorithm
                        }
                except Exception as e:
                    import traceback
                    results[algorithm] = {
                        'error': str(e),
                        'algorithm': algorithm,
                        'traceback': traceback.format_exc()
                    }
        
        return results


class WeightedColoring:
    """Weighted graph coloring where colors have associated costs."""
    
    def __init__(self, graph: Graph, color_costs: Dict[int, float]):
        """
        Initialize weighted coloring.
        
        Args:
            graph: Graph to color
            color_costs: Dictionary mapping color index to cost
        """
        self.graph = graph
        self.color_costs = color_costs
        self.solver = ColoringSolver(graph)
    
    def solve_min_cost(self, algorithm: str = 'dsatur') -> Dict:
        """
        Solve weighted coloring to minimize total cost.
        
        Args:
            algorithm: Base coloring algorithm
            
        Returns:
            Dictionary with coloring and total cost
        """
        result = self.solver.solve(algorithm)
        coloring = result['coloring']
        
        # Calculate total cost
        total_cost = 0.0
        for vertex, color in coloring.items():
            cost = self.color_costs.get(color, 1.0)
            total_cost += cost
        
        return {
            'coloring': coloring,
            'num_colors': result['num_colors'],
            'total_cost': total_cost,
            'algorithm': algorithm
        }


class ILPColoring:
    """Graph coloring using Integer Linear Programming."""
    
    def __init__(self, graph: Graph):
        """
        Initialize ILP coloring.
        
        Args:
            graph: Graph to color
        """
        self.graph = graph
        self.nx_graph = graph.get_networkx_graph()
    
    def solve(self, max_colors: Optional[int] = None) -> Dict:
        """
        Solve using ILP (PuLP or OR-Tools).
        
        Args:
            max_colors: Maximum number of colors (None = upper bound)
            
        Returns:
            Dictionary with coloring result
        """
        try:
            import pulp
            return self._solve_pulp(max_colors)
        except ImportError:
            try:
                from ortools.linear_solver import pywraplp
                return self._solve_ortools(max_colors)
            except ImportError:
                raise ImportError("Neither PuLP nor OR-Tools is available. Install one of them.")
    
    def _solve_pulp(self, max_colors: Optional[int] = None) -> Dict:
        """Solve using PuLP."""
        import pulp
        
        vertices = list(self.nx_graph.nodes())
        n = len(vertices)
        
        if max_colors is None:
            max_colors = self.graph.get_max_degree() + 1
        
        # Create problem
        prob = pulp.LpProblem("Graph_Coloring", pulp.LpMinimize)
        
        # Decision variables
        # x[i][c] = 1 if vertex i has color c, 0 otherwise
        x = {}
        for i, vertex in enumerate(vertices):
            x[i] = {}
            for c in range(max_colors):
                x[i][c] = pulp.LpVariable(f"x_{i}_{c}", cat='Binary')
        
        # y[c] = 1 if color c is used, 0 otherwise
        y = {}
        for c in range(max_colors):
            y[c] = pulp.LpVariable(f"y_{c}", cat='Binary')
        
        # Objective: minimize number of colors used
        prob += pulp.lpSum([y[c] for c in range(max_colors)])
        
        # Constraints
        # Each vertex must have exactly one color
        for i in range(n):
            prob += pulp.lpSum([x[i][c] for c in range(max_colors)]) == 1
        
        # Adjacent vertices cannot have the same color
        for edge in self.nx_graph.edges():
            i = vertices.index(edge[0])
            j = vertices.index(edge[1])
            for c in range(max_colors):
                prob += x[i][c] + x[j][c] <= 1
        
        # Color is used if at least one vertex has it
        for c in range(max_colors):
            for i in range(n):
                prob += y[c] >= x[i][c]
        
        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        if prob.status != pulp.LpStatusOptimal:
            # Fallback to greedy
            solver = ColoringSolver(self.graph)
            result = solver.solve('dsatur')
            return result
        
        # Extract coloring
        coloring = {}
        for i, vertex in enumerate(vertices):
            for c in range(max_colors):
                if pulp.value(x[i][c]) == 1:
                    coloring[vertex] = c
                    break
        
        num_colors = sum(1 for c in range(max_colors) if pulp.value(y[c]) == 1)
        
        return {
            'coloring': coloring,
            'num_colors': num_colors,
            'algorithm': 'ILP (PuLP)'
        }
    
    def _solve_ortools(self, max_colors: Optional[int] = None) -> Dict:
        """Solve using OR-Tools."""
        from ortools.linear_solver import pywraplp
        
        vertices = list(self.nx_graph.nodes())
        n = len(vertices)
        
        if max_colors is None:
            max_colors = self.graph.get_max_degree() + 1
        
        # Create solver
        solver = pywraplp.Solver.CreateSolver('SCIP')
        if not solver:
            raise RuntimeError("Could not create OR-Tools solver")
        
        # Decision variables
        x = {}
        for i, vertex in enumerate(vertices):
            x[i] = {}
            for c in range(max_colors):
                x[i][c] = solver.IntVar(0, 1, f"x_{i}_{c}")
        
        y = {}
        for c in range(max_colors):
            y[c] = solver.IntVar(0, 1, f"y_{c}")
        
        # Objective: minimize number of colors
        solver.Minimize(solver.Sum([y[c] for c in range(max_colors)]))
        
        # Constraints
        # Each vertex has exactly one color
        for i in range(n):
            solver.Add(solver.Sum([x[i][c] for c in range(max_colors)]) == 1)
        
        # Adjacent vertices cannot have same color
        for edge in self.nx_graph.edges():
            i = vertices.index(edge[0])
            j = vertices.index(edge[1])
            for c in range(max_colors):
                solver.Add(x[i][c] + x[j][c] <= 1)
        
        # Color is used if at least one vertex has it
        for c in range(max_colors):
            for i in range(n):
                solver.Add(y[c] >= x[i][c])
        
        # Solve
        status = solver.Solve()
        
        if status != pywraplp.Solver.OPTIMAL:
            # Fallback to greedy
            solver_obj = ColoringSolver(self.graph)
            result = solver_obj.solve('dsatur')
            return result
        
        # Extract coloring
        coloring = {}
        for i, vertex in enumerate(vertices):
            for c in range(max_colors):
                if x[i][c].solution_value() == 1:
                    coloring[vertex] = c
                    break
        
        num_colors = sum(1 for c in range(max_colors) if y[c].solution_value() == 1)
        
        return {
            'coloring': coloring,
            'num_colors': int(num_colors),
            'algorithm': 'ILP (OR-Tools)'
        }


class DynamicRecoloring:
    """Handle graph modifications with dynamic recoloring."""
    
    def __init__(self, graph: Graph, initial_coloring: Dict):
        """
        Initialize dynamic recoloring.
        
        Args:
            graph: Graph
            initial_coloring: Initial coloring
        """
        self.graph = graph
        self.coloring = initial_coloring.copy()
        self.solver = ColoringSolver(graph)
    
    def add_vertex(self, vertex, neighbors: List) -> Dict:
        """
        Add vertex and recolor if necessary.
        
        Args:
            vertex: New vertex
            neighbors: List of neighbors
            
        Returns:
            Updated coloring
        """
        # Add vertex and edges
        self.graph.add_vertex(vertex)
        for neighbor in neighbors:
            if neighbor in self.graph.get_vertices():
                try:
                    self.graph.add_edge(vertex, neighbor)
                except ValueError as e:
                    # Skip if edge already exists or self-loop
                    pass
        
        # Find available color
        neighbor_colors = {self.coloring[n] for n in neighbors if n in self.coloring}
        color = 0
        while color in neighbor_colors:
            color += 1
        
        self.coloring[vertex] = color
        return self.coloring
    
    def remove_vertex(self, vertex) -> Dict:
        """
        Remove vertex.
        
        Args:
            vertex: Vertex to remove
            
        Returns:
            Updated coloring
        """
        self.graph.remove_vertex(vertex)
        if vertex in self.coloring:
            del self.coloring[vertex]
        return self.coloring
    
    def add_edge(self, u, v) -> Dict:
        """
        Add edge and recolor if necessary.
        
        Args:
            u: First vertex
            v: Second vertex
            
        Returns:
            Updated coloring
        """
        if u == v:
            raise ValueError("Cannot add self-loop")
        
        if u not in self.graph.get_vertices() or v not in self.graph.get_vertices():
            raise ValueError(f"One or both vertices ({u}, {v}) not found in graph")
        
        # Check if edge already exists
        if (u, v) in self.graph.get_edges() or (v, u) in self.graph.get_edges():
            return self.coloring  # Edge already exists, no change needed
        
        self.graph.add_edge(u, v)
        
        # Check if recoloring is needed
        if u in self.coloring and v in self.coloring and self.coloring.get(u) == self.coloring.get(v):
            # Recolor one of the vertices
            # Use greedy approach: recolor the vertex with smaller degree
            deg_u = self.graph.get_degree(u)
            deg_v = self.graph.get_degree(v)
            
            if deg_u <= deg_v:
                vertex_to_recolor = u
            else:
                vertex_to_recolor = v
            
            # Find available color
            neighbors = self.graph.get_neighbors(vertex_to_recolor)
            neighbor_colors = {self.coloring[n] for n in neighbors if n in self.coloring and n != vertex_to_recolor}
            color = 0
            while color in neighbor_colors:
                color += 1
            
            self.coloring[vertex_to_recolor] = color
        
        return self.coloring
    
    def remove_edge(self, u, v) -> Dict:
        """
        Remove edge (no recoloring needed).
        
        Args:
            u: First vertex
            v: Second vertex
            
        Returns:
            Updated coloring
        """
        if (u, v) not in self.graph.get_edges() and (v, u) not in self.graph.get_edges():
            raise ValueError(f"Edge ({u}, {v}) not found in graph")
        
        self.graph.remove_edge(u, v)
        return self.coloring

