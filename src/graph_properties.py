"""
Graph properties calculator.
"""

from typing import Dict, List, Optional, Tuple
import networkx as nx
from .graph import Graph


class GraphProperties:
    """Calculate various graph properties."""
    
    def __init__(self, graph: Graph):
        """
        Initialize graph properties calculator.
        
        Args:
            graph: Graph to analyze
        """
        self.graph = graph
        self.nx_graph = graph.get_networkx_graph()
    
    def chromatic_number_lower_bound(self) -> int:
        """
        Calculate lower bound for chromatic number.
        
        Returns:
            Lower bound (clique number)
        """
        return self.clique_number()
    
    def chromatic_number_upper_bound(self) -> int:
        """
        Calculate upper bound for chromatic number.
        
        Returns:
            Upper bound (max degree + 1)
        """
        if self.graph.num_vertices() == 0:
            return 0
        return self.graph.get_max_degree() + 1
    
    def clique_number(self) -> int:
        """
        Calculate clique number (size of largest clique).
        
        Returns:
            Clique number
        """
        if self.graph.num_vertices() == 0:
            return 0
        cliques = list(nx.find_cliques(self.nx_graph))
        if not cliques:
            return 0
        return max(len(clique) for clique in cliques)
    
    def independence_number(self) -> int:
        """
        Calculate independence number (size of largest independent set).
        
        Returns:
            Independence number
        """
        if self.graph.num_vertices() == 0:
            return 0
        try:
            # max_weight_clique returns (clique, weight) tuple
            result = nx.max_weight_clique(nx.complement(self.nx_graph))
            if result and len(result) > 0:
                clique = result[0] if isinstance(result, tuple) else result
                return len(clique)
            return 0
        except Exception:
            # Fallback: use approximation
            try:
                # Try to find maximum independent set using approximation
                independent_set = nx.maximal_independent_set(self.nx_graph)
                return len(independent_set)
            except Exception:
                return 0
    
    def chromatic_polynomial_approx(self) -> Optional[Dict]:
        """
        Approximate chromatic polynomial coefficients.
        
        Note: Exact calculation is computationally expensive.
        This returns an approximation based on graph structure.
        
        Returns:
            Dictionary with polynomial information
        """
        n = self.graph.num_vertices()
        if n == 0:
            return None
        
        # For trees: P(G, k) = k * (k-1)^(n-1)
        if nx.is_tree(self.nx_graph):
            return {
                'type': 'tree',
                'formula': f'k * (k-1)^{n-1}',
                'degree': n
            }
        
        # For complete graphs: P(G, k) = k * (k-1) * ... * (k-n+1)
        if nx.is_complete(self.nx_graph):
            return {
                'type': 'complete',
                'formula': f'k * (k-1) * ... * (k-{n-1}+1)',
                'degree': n
            }
        
        # For cycles: P(C_n, k) = (k-1)^n + (-1)^n * (k-1)
        if nx.is_regular(self.nx_graph) and self.graph.get_max_degree() == 2:
            return {
                'type': 'cycle',
                'formula': f'(k-1)^{n} + (-1)^{n} * (k-1)',
                'degree': n
            }
        
        return {
            'type': 'general',
            'degree': n,
            'note': 'Exact calculation requires exponential time'
        }
    
    def get_all_properties(self) -> Dict:
        """Get all graph properties."""
        try:
            num_vertices = self.graph.num_vertices()
            num_edges = self.graph.num_edges()
            
            # Calculate degrees safely
            degrees = dict(self.nx_graph.degree())
            min_degree = min(degrees.values()) if degrees else 0
            max_degree = self.graph.get_max_degree()
            avg_degree = 2 * num_edges / num_vertices if num_vertices > 0 else 0.0
            
            # Calculate properties with error handling
            try:
                is_connected = self.graph.is_connected()
            except Exception:
                is_connected = False
            
            try:
                is_bipartite = self.graph.is_bipartite()
            except Exception:
                is_bipartite = False
            
            try:
                is_tree = nx.is_tree(self.nx_graph)
            except Exception:
                is_tree = False
            
            try:
                clique_num = self.clique_number()
            except Exception:
                clique_num = 0
            
            try:
                indep_num = self.independence_number()
            except Exception:
                indep_num = 0
            
            try:
                chrom_lower = self.chromatic_number_lower_bound()
            except Exception:
                chrom_lower = 0
            
            try:
                chrom_upper = self.chromatic_number_upper_bound()
            except Exception:
                chrom_upper = max_degree + 1 if max_degree > 0 else 0
            
            try:
                num_components = nx.number_connected_components(self.nx_graph)
            except Exception:
                num_components = 1
            
            return {
                'num_vertices': num_vertices,
                'num_edges': num_edges,
                'max_degree': max_degree,
                'min_degree': min_degree,
                'average_degree': round(avg_degree, 3),
                'is_connected': is_connected,
                'is_bipartite': is_bipartite,
                'is_tree': is_tree,
                'is_planar': self._is_planar(),
                'clique_number': clique_num,
                'independence_number': indep_num,
                'chromatic_lower_bound': chrom_lower,
                'chromatic_upper_bound': chrom_upper,
                'num_components': num_components,
                'density': round(self._calculate_density(), 3),
                'chromatic_polynomial': self.chromatic_polynomial_approx()
            }
        except Exception as e:
            # Return minimal properties if calculation fails
            return {
                'num_vertices': self.graph.num_vertices(),
                'num_edges': self.graph.num_edges(),
                'max_degree': self.graph.get_max_degree(),
                'min_degree': 0,
                'average_degree': 0.0,
                'is_connected': False,
                'is_bipartite': False,
                'is_tree': False,
                'is_planar': False,
                'clique_number': 0,
                'independence_number': 0,
                'chromatic_lower_bound': 0,
                'chromatic_upper_bound': 0,
                'num_components': 1,
                'density': 0.0,
                'chromatic_polynomial': None,
                'error': str(e)
            }
    
    def _is_planar(self) -> bool:
        """Check if graph is planar."""
        try:
            return nx.is_planar(self.nx_graph)
        except:
            return False
    
    def _calculate_density(self) -> float:
        """Calculate graph density."""
        n = self.graph.num_vertices()
        if n <= 1:
            return 0.0
        max_edges = n * (n - 1) / 2
        return self.graph.num_edges() / max_edges if max_edges > 0 else 0.0

