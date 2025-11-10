"""
Graph coloring algorithms implementation.
"""

import random
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict, deque
import networkx as nx
from .graph import Graph


class ColoringAlgorithm:
    """Base class for graph coloring algorithms."""
    
    def __init__(self, graph: Graph):
        """
        Initialize coloring algorithm.
        
        Args:
            graph: Graph to color
        """
        self.graph = graph
        self.nx_graph = graph.get_networkx_graph()
        self.coloring: Dict = {}
        self.num_colors: int = 0
    
    def solve(self) -> Dict:
        """
        Solve the coloring problem.
        
        Returns:
            Dictionary mapping vertices to colors
        """
        raise NotImplementedError
    
    def validate_coloring(self, coloring: Optional[Dict] = None) -> Tuple[bool, Optional[str]]:
        """
        Validate that no two adjacent vertices have the same color.
        
        Args:
            coloring: Coloring to validate (uses self.coloring if None)
            
        Returns:
            (is_valid, error_message)
        """
        if coloring is None:
            coloring = self.coloring
        
        for edge in self.nx_graph.edges():
            u, v = edge
            if u in coloring and v in coloring:
                if coloring[u] == coloring[v]:
                    return False, f"Adjacent vertices {u} and {v} have the same color {coloring[u]}"
        
        return True, None
    
    def get_num_colors(self) -> int:
        """Get number of colors used."""
        if not self.coloring:
            return 0
        return len(set(self.coloring.values()))


class GreedyColoring(ColoringAlgorithm):
    """Greedy coloring algorithm with various vertex ordering strategies."""
    
    def __init__(self, graph: Graph, ordering: str = 'largest_first'):
        """
        Initialize greedy coloring.
        
        Args:
            graph: Graph to color
            ordering: Ordering strategy ('largest_first', 'smallest_last', 'random')
        """
        super().__init__(graph)
        self.ordering = ordering
    
    def solve(self) -> Dict:
        """Solve using greedy coloring."""
        vertices = self._get_ordered_vertices()
        self.coloring = {}
        used_colors = defaultdict(set)  # color -> set of vertices
        
        for vertex in vertices:
            # Find the smallest available color
            color = self._find_available_color(vertex, used_colors)
            self.coloring[vertex] = color
            used_colors[color].add(vertex)
        
        self.num_colors = len(used_colors)
        return self.coloring
    
    def _get_ordered_vertices(self) -> List:
        """Get vertices in the specified order."""
        vertices = list(self.nx_graph.nodes())
        
        if self.ordering == 'largest_first':
            # Sort by degree (descending)
            degrees = dict(self.nx_graph.degree())
            vertices.sort(key=lambda v: degrees[v], reverse=True)
        elif self.ordering == 'smallest_last':
            # Smallest-last ordering (remove vertices with smallest degree iteratively)
            vertices = self._smallest_last_ordering()
        elif self.ordering == 'random':
            random.shuffle(vertices)
        else:
            raise ValueError(f"Unknown ordering: {self.ordering}")
        
        return vertices
    
    def _smallest_last_ordering(self) -> List:
        """Compute smallest-last ordering."""
        g = self.nx_graph.copy()
        ordering = []
        
        while g.number_of_nodes() > 0:
            # Find vertex with minimum degree
            degrees = dict(g.degree())
            if not degrees:
                break
            min_degree = min(degrees.values())
            candidates = [v for v, d in degrees.items() if d == min_degree]
            vertex = min(candidates)  # Break ties deterministically
            
            ordering.append(vertex)
            g.remove_node(vertex)
        
        return list(reversed(ordering))  # Reverse to get smallest-last
    
    def _find_available_color(self, vertex: int, used_colors: Dict) -> int:
        """Find the smallest available color for a vertex."""
        neighbors = list(self.nx_graph.neighbors(vertex))
        neighbor_colors = {self.coloring[n] for n in neighbors if n in self.coloring}
        
        color = 0
        while color in neighbor_colors:
            color += 1
        
        return color


class WelshPowell(ColoringAlgorithm):
    """Welsh-Powell algorithm that sorts vertices by degree."""
    
    def solve(self) -> Dict:
        """Solve using Welsh-Powell algorithm."""
        # Sort vertices by degree (descending)
        degrees = dict(self.nx_graph.degree())
        vertices = sorted(self.nx_graph.nodes(), key=lambda v: degrees[v], reverse=True)
        
        self.coloring = {}
        color = 0
        
        while len(self.coloring) < len(vertices):
            # Try to color as many uncolored vertices as possible with current color
            for vertex in vertices:
                if vertex not in self.coloring:
                    # Check if we can use current color
                    neighbors = list(self.nx_graph.neighbors(vertex))
                    can_use_color = all(
                        neighbor not in self.coloring or self.coloring[neighbor] != color
                        for neighbor in neighbors
                    )
                    
                    if can_use_color:
                        self.coloring[vertex] = color
            
            color += 1
        
        self.num_colors = color
        return self.coloring


class BacktrackingColoring(ColoringAlgorithm):
    """Backtracking algorithm for finding exact chromatic number."""
    
    def __init__(self, graph: Graph, max_colors: Optional[int] = None):
        """
        Initialize backtracking coloring.
        
        Args:
            graph: Graph to color
            max_colors: Maximum number of colors to try (None = no limit)
        """
        super().__init__(graph)
        self.max_colors = max_colors
        self.best_coloring: Optional[Dict] = None
        self.best_num_colors = float('inf')
    
    def solve(self) -> Dict:
        """Solve using backtracking."""
        vertices = list(self.nx_graph.nodes())
        if not vertices:
            return {}
        
        # Try with increasing number of colors
        max_colors = self.max_colors if self.max_colors else len(vertices)
        
        for num_colors in range(1, max_colors + 1):
            self.coloring = {}
            if self._backtrack(vertices, 0, num_colors):
                self.num_colors = num_colors
                return self.coloring
        
        # If no solution found, return greedy solution as fallback
        greedy = GreedyColoring(self.graph, 'largest_first')
        self.coloring = greedy.solve()
        self.num_colors = greedy.get_num_colors()
        return self.coloring
    
    def _backtrack(self, vertices: List, index: int, num_colors: int) -> bool:
        """Backtracking helper function."""
        if index == len(vertices):
            return True
        
        vertex = vertices[index]
        neighbors = list(self.nx_graph.neighbors(vertex))
        neighbor_colors = {self.coloring[n] for n in neighbors if n in self.coloring}
        
        for color in range(num_colors):
            if color not in neighbor_colors:
                self.coloring[vertex] = color
                if self._backtrack(vertices, index + 1, num_colors):
                    return True
                del self.coloring[vertex]
        
        return False


class DSATUR(ColoringAlgorithm):
    """DSATUR (Degree of Saturation) algorithm."""
    
    def solve(self) -> Dict:
        """Solve using DSATUR algorithm."""
        self.coloring = {}
        uncolored = set(self.nx_graph.nodes())
        saturation = {v: 0 for v in self.nx_graph.nodes()}  # Number of different colors in neighbors
        
        while uncolored:
            # Find uncolored vertex with maximum saturation degree
            # Break ties by choosing vertex with maximum degree
            degrees = dict(self.nx_graph.degree())
            candidates = [(v, saturation[v], degrees[v]) for v in uncolored]
            candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
            vertex = candidates[0][0]
            
            # Find available color
            neighbors = list(self.nx_graph.neighbors(vertex))
            neighbor_colors = {self.coloring[n] for n in neighbors if n in self.coloring}
            
            color = 0
            while color in neighbor_colors:
                color += 1
            
            self.coloring[vertex] = color
            uncolored.remove(vertex)
            
            # Update saturation degrees of neighbors
            for neighbor in neighbors:
                if neighbor in uncolored:
                    neighbor_colors_set = {
                        self.coloring[n] for n in self.nx_graph.neighbors(neighbor)
                        if n in self.coloring
                    }
                    saturation[neighbor] = len(neighbor_colors_set)
        
        self.num_colors = len(set(self.coloring.values()))
        return self.coloring


class ColoringSolver:
    """Orchestrator for graph coloring algorithms."""
    
    ALGORITHMS = {
        'greedy_largest_first': lambda g: GreedyColoring(g, 'largest_first'),
        'greedy_smallest_last': lambda g: GreedyColoring(g, 'smallest_last'),
        'greedy_random': lambda g: GreedyColoring(g, 'random'),
        'welsh_powell': WelshPowell,
        'backtracking': BacktrackingColoring,
        'dsatur': DSATUR,
    }
    
    def __init__(self, graph: Graph):
        """
        Initialize coloring solver.
        
        Args:
            graph: Graph to color
        """
        self.graph = graph
        self.results: Dict[str, Dict] = {}
    
    def solve(self, algorithm: str, **kwargs) -> Dict:
        """
        Solve coloring problem using specified algorithm.
        
        Args:
            algorithm: Algorithm name
            **kwargs: Additional algorithm-specific parameters
            
        Returns:
            Coloring result dictionary
        """
        if algorithm not in self.ALGORITHMS:
            raise ValueError(f"Unknown algorithm: {algorithm}. Available: {list(self.ALGORITHMS.keys())}")
        
        import time
        start_time = time.time()
        
        solver = self.ALGORITHMS[algorithm](self.graph, **kwargs)
        coloring = solver.solve()
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        is_valid, error = solver.validate_coloring()
        
        result = {
            'coloring': coloring,
            'num_colors': solver.get_num_colors(),
            'execution_time': execution_time,
            'is_valid': is_valid,
            'error': error,
            'algorithm': algorithm
        }
        
        self.results[algorithm] = result
        return result
    
    def solve_all(self) -> Dict[str, Dict]:
        """Solve using all available algorithms."""
        for algorithm in self.ALGORITHMS.keys():
            try:
                self.solve(algorithm)
            except Exception as e:
                self.results[algorithm] = {
                    'error': str(e),
                    'algorithm': algorithm
                }
        return self.results
    
    def get_best_solution(self) -> Optional[Dict]:
        """Get the solution with minimum number of colors."""
        valid_results = {
            k: v for k, v in self.results.items()
            if v.get('is_valid', False) and 'num_colors' in v
        }
        
        if not valid_results:
            return None
        
        best = min(valid_results.items(), key=lambda x: x[1]['num_colors'])
        return best[1]

