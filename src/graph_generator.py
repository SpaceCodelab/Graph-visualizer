"""
Graph generator for creating various types of test graphs.
"""

from typing import Optional, Tuple
import numpy as np
import networkx as nx
from .graph import Graph


class GraphGenerator:
    """Generator for various types of graphs."""
    
    @staticmethod
    def complete_graph(n: int) -> Graph:
        """
        Generate a complete graph K_n.
        
        Args:
            n: Number of vertices
            
        Returns:
            Complete graph with n vertices
        """
        g = nx.complete_graph(n)
        return Graph(g)
    
    @staticmethod
    def bipartite_graph(n1: int, n2: int, p: float = 1.0) -> Graph:
        """
        Generate a bipartite graph.
        
        Args:
            n1: Number of vertices in first partition
            n2: Number of vertices in second partition
            p: Probability of edge between partitions (1.0 = complete bipartite)
            
        Returns:
            Bipartite graph
        """
        if p == 1.0:
            g = nx.complete_bipartite_graph(n1, n2)
        else:
            try:
                # Try using bipartite.random_graph
                from networkx.algorithms import bipartite
                g = bipartite.random_graph(n1, n2, p, seed=None, directed=False)
            except (ImportError, AttributeError):
                # Fallback: create bipartite graph manually
                g = nx.Graph()
                # Add nodes
                for i in range(n1):
                    g.add_node(i, bipartite=0)
                for i in range(n1, n1 + n2):
                    g.add_node(i, bipartite=1)
                # Add edges with probability p
                import random
                for i in range(n1):
                    for j in range(n1, n1 + n2):
                        if random.random() < p:
                            g.add_edge(i, j)
        return Graph(g)
    
    @staticmethod
    def cycle_graph(n: int) -> Graph:
        """
        Generate a cycle graph C_n.
        
        Args:
            n: Number of vertices
            
        Returns:
            Cycle graph
        """
        g = nx.cycle_graph(n)
        return Graph(g)
    
    @staticmethod
    def path_graph(n: int) -> Graph:
        """
        Generate a path graph P_n.
        
        Args:
            n: Number of vertices
            
        Returns:
            Path graph
        """
        g = nx.path_graph(n)
        return Graph(g)
    
    @staticmethod
    def planar_graph(n: int, m: Optional[int] = None) -> Graph:
        """
        Generate a planar graph.
        
        Args:
            n: Number of vertices
            m: Number of edges (if None, uses a reasonable default)
            
        Returns:
            Planar graph
        """
        if n < 3:
            raise ValueError("Planar graph requires at least 3 vertices")
        
        try:
            g = nx.random_planar_graph(n, seed=None)
            return Graph(g)
        except Exception:
            # Fallback: create a simple planar graph (triangular grid)
            g = nx.Graph()
            # Create a simple cycle (always planar)
            for i in range(n):
                g.add_node(i)
            for i in range(n):
                g.add_edge(i, (i + 1) % n)
            return Graph(g)
    
    @staticmethod
    def random_graph(n: int, p: float, seed: Optional[int] = None) -> Graph:
        """
        Generate a random graph (Erdős–Rényi model).
        
        Args:
            n: Number of vertices
            p: Probability of edge creation
            seed: Random seed
            
        Returns:
            Random graph
        """
        g = nx.erdos_renyi_graph(n, p, seed=seed)
        return Graph(g)
    
    @staticmethod
    def random_graph_by_density(n: int, density: float, seed: Optional[int] = None) -> Graph:
        """
        Generate a random graph with specified density.
        
        Args:
            n: Number of vertices
            density: Edge density (0.0 to 1.0)
            seed: Random seed
            
        Returns:
            Random graph
        """
        max_edges = n * (n - 1) // 2
        m = int(density * max_edges)
        g = nx.gnm_random_graph(n, m, seed=seed)
        return Graph(g)
    
    @staticmethod
    def petersen_graph() -> Graph:
        """Generate Petersen graph."""
        g = nx.petersen_graph()
        return Graph(g)
    
    @staticmethod
    def wheel_graph(n: int) -> Graph:
        """
        Generate a wheel graph W_n.
        
        Args:
            n: Number of vertices in the cycle (total vertices = n + 1)
            
        Returns:
            Wheel graph
        """
        g = nx.wheel_graph(n)
        return Graph(g)
    
    @staticmethod
    def star_graph(n: int) -> Graph:
        """
        Generate a star graph S_n.
        
        Args:
            n: Number of leaves (total vertices = n + 1)
            
        Returns:
            Star graph
        """
        g = nx.star_graph(n)
        return Graph(g)
    
    @staticmethod
    def grid_graph(m: int, n: int) -> Graph:
        """
        Generate an m x n grid graph.
        
        Args:
            m: Number of rows
            n: Number of columns
            
        Returns:
            Grid graph
        """
        if m < 1 or n < 1:
            raise ValueError("Grid dimensions must be at least 1")
        
        g = nx.grid_2d_graph(m, n)
        # Convert to simple node labels
        nodes_list = sorted(g.nodes())
        mapping = {node: i for i, node in enumerate(nodes_list)}
        g = nx.relabel_nodes(g, mapping)
        return Graph(g)
    
    @staticmethod
    def tree_graph(n: int, seed: Optional[int] = None) -> Graph:
        """
        Generate a random tree.
        
        Args:
            n: Number of vertices
            seed: Random seed
            
        Returns:
            Tree graph
        """
        g = nx.random_tree(n, seed=seed)
        return Graph(g)
    
    @staticmethod
    def regular_graph(n: int, d: int, seed: Optional[int] = None) -> Graph:
        """
        Generate a d-regular graph.
        
        Args:
            n: Number of vertices
            d: Degree of each vertex
            seed: Random seed
            
        Returns:
            Regular graph
        """
        # Validate parameters
        if n < 1:
            raise ValueError("Number of vertices must be at least 1")
        if d < 0:
            raise ValueError("Degree must be non-negative")
        if d >= n:
            raise ValueError(f"Degree {d} must be less than number of vertices {n}")
        if (n * d) % 2 != 0:
            raise ValueError(f"Cannot create {d}-regular graph with {n} vertices: n*d must be even")
        
        try:
            g = nx.random_regular_graph(d, n, seed=seed)
            return Graph(g)
        except (nx.NetworkXError, ValueError) as e:
            raise ValueError(f"Cannot create {d}-regular graph with {n} vertices: {e}")
    
    @staticmethod
    def barbell_graph(m1: int, m2: int) -> Graph:
        """
        Generate a barbell graph.
        
        Args:
            m1: Number of vertices in each complete graph
            m2: Number of vertices in the path connecting them
            
        Returns:
            Barbell graph
        """
        g = nx.barbell_graph(m1, m2)
        return Graph(g)
    
    @staticmethod
    def lollipop_graph(m: int, n: int) -> Graph:
        """
        Generate a lollipop graph.
        
        Args:
            m: Number of vertices in the complete graph
            n: Number of vertices in the path
            
        Returns:
            Lollipop graph
        """
        g = nx.lollipop_graph(m, n)
        return Graph(g)

