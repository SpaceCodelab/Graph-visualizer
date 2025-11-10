"""
Graph data structure and input/output handling.
"""

import json
import csv
from typing import Dict, List, Set, Tuple, Optional, Union
import numpy as np
import networkx as nx
from pathlib import Path


class Graph:
    """
    Graph class for representing and manipulating graphs.
    
    Supports multiple input formats: adjacency matrix, adjacency list, edge list,
    and file loading (CSV, JSON).
    """
    
    def __init__(self, graph_data: Optional[Union[nx.Graph, Dict, List, np.ndarray]] = None):
        """
        Initialize a graph.
        
        Args:
            graph_data: Can be a NetworkX graph, adjacency dict, edge list, or adjacency matrix
        """
        self._graph = nx.Graph()
        self._directed = False
        
        if graph_data is not None:
            if isinstance(graph_data, nx.Graph):
                self._graph = graph_data.copy()
            elif isinstance(graph_data, dict):
                self._from_adjacency_dict(graph_data)
            elif isinstance(graph_data, list):
                self._from_edge_list(graph_data)
            elif isinstance(graph_data, np.ndarray):
                self._from_adjacency_matrix(graph_data)
    
    def _from_adjacency_dict(self, adj_dict: Dict):
        """Load graph from adjacency dictionary."""
        self._graph = nx.Graph()
        for node, neighbors in adj_dict.items():
            if not isinstance(neighbors, (list, tuple, set)):
                raise ValueError(f"Neighbors for node {node} must be a list, tuple, or set")
            for neighbor in neighbors:
                if node == neighbor:
                    continue  # Skip self-loops
                self._graph.add_edge(node, neighbor)
    
    def _from_edge_list(self, edges: List[Tuple]):
        """Load graph from edge list."""
        self._graph = nx.Graph()
        for edge in edges:
            if len(edge) < 2:
                continue  # Skip invalid edges
            u, v = edge[0], edge[1]
            if u == v:
                continue  # Skip self-loops
            if len(edge) == 2:
                self._graph.add_edge(u, v)
            elif len(edge) == 3:
                self._graph.add_edge(u, v, weight=edge[2])
    
    def _from_adjacency_matrix(self, matrix: np.ndarray):
        """Load graph from adjacency matrix."""
        self._graph = nx.Graph()
        n = matrix.shape[0]
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Adjacency matrix must be square")
        for i in range(n):
            for j in range(i + 1, n):
                if matrix[i, j] != 0:
                    self._graph.add_edge(i, j, weight=float(matrix[i, j]))
    
    @classmethod
    def from_adjacency_matrix(cls, matrix: np.ndarray) -> 'Graph':
        """Create graph from adjacency matrix."""
        return cls(matrix)
    
    @classmethod
    def from_adjacency_list(cls, adj_list: Dict) -> 'Graph':
        """Create graph from adjacency list."""
        return cls(adj_list)
    
    @classmethod
    def from_edge_list(cls, edges: List[Tuple]) -> 'Graph':
        """Create graph from edge list."""
        return cls(edges)
    
    @classmethod
    def from_file(cls, filepath: str) -> 'Graph':
        """
        Load graph from file (CSV or JSON).
        
        CSV format: Each line contains two nodes (comma-separated)
        JSON format: {"edges": [[node1, node2], ...]} or {"adjacency": {...}}
        """
        path = Path(filepath)
        if path.suffix.lower() == '.csv':
            return cls._from_csv(filepath)
        elif path.suffix.lower() == '.json':
            return cls._from_json(filepath)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    @classmethod
    def _from_csv(cls, filepath: str) -> 'Graph':
        """Load graph from CSV file."""
        edges = []
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    try:
                        # Try to convert to integers if possible
                        node1 = int(row[0]) if row[0].isdigit() else row[0]
                        node2 = int(row[1]) if row[1].isdigit() else row[1]
                        edges.append((node1, node2))
                    except ValueError:
                        edges.append((row[0], row[1]))
        return cls(edges)
    
    @classmethod
    def _from_json(cls, filepath: str) -> 'Graph':
        """Load graph from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'edges' in data:
            return cls(data['edges'])
        elif 'adjacency' in data:
            return cls(data['adjacency'])
        else:
            raise ValueError("JSON must contain 'edges' or 'adjacency' key")
    
    def to_adjacency_matrix(self) -> np.ndarray:
        """Convert graph to adjacency matrix."""
        nodes = sorted(self._graph.nodes())
        n = len(nodes)
        matrix = np.zeros((n, n), dtype=int)
        node_to_index = {node: i for i, node in enumerate(nodes)}
        
        for edge in self._graph.edges():
            i = node_to_index[edge[0]]
            j = node_to_index[edge[1]]
            matrix[i, j] = 1
            matrix[j, i] = 1
        
        return matrix
    
    def to_adjacency_list(self) -> Dict:
        """Convert graph to adjacency list."""
        adj_list = {}
        for node in self._graph.nodes():
            adj_list[node] = list(self._graph.neighbors(node))
        return adj_list
    
    def to_edge_list(self) -> List[Tuple]:
        """Convert graph to edge list."""
        return list(self._graph.edges())
    
    def save_to_file(self, filepath: str, format: str = 'json'):
        """
        Save graph to file.
        
        Args:
            filepath: Path to save file
            format: 'json' or 'csv'
        """
        path = Path(filepath)
        if format == 'json' or path.suffix.lower() == '.json':
            data = {
                'edges': self.to_edge_list(),
                'adjacency': self.to_adjacency_list()
            }
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        elif format == 'csv' or path.suffix.lower() == '.csv':
            with open(filepath, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                for edge in self.to_edge_list():
                    writer.writerow(edge)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def add_vertex(self, vertex):
        """Add a vertex to the graph."""
        self._graph.add_node(vertex)
    
    def remove_vertex(self, vertex):
        """Remove a vertex from the graph."""
        self._graph.remove_node(vertex)
    
    def add_edge(self, u, v, weight: Optional[float] = None):
        """Add an edge between vertices u and v."""
        if u == v:
            raise ValueError("Self-loops are not allowed in simple graphs")
        if weight is not None:
            self._graph.add_edge(u, v, weight=weight)
        else:
            self._graph.add_edge(u, v)
    
    def remove_edge(self, u, v):
        """Remove an edge between vertices u and v."""
        self._graph.remove_edge(u, v)
    
    def get_neighbors(self, vertex) -> List:
        """Get neighbors of a vertex."""
        return list(self._graph.neighbors(vertex))
    
    def get_degree(self, vertex) -> int:
        """Get degree of a vertex."""
        return self._graph.degree(vertex)
    
    def get_vertices(self) -> List:
        """Get all vertices."""
        return list(self._graph.nodes())
    
    def get_edges(self) -> List[Tuple]:
        """Get all edges."""
        return list(self._graph.edges())
    
    def num_vertices(self) -> int:
        """Get number of vertices."""
        return self._graph.number_of_nodes()
    
    def num_edges(self) -> int:
        """Get number of edges."""
        return self._graph.number_of_edges()
    
    def is_connected(self) -> bool:
        """Check if graph is connected."""
        return nx.is_connected(self._graph)
    
    def is_bipartite(self) -> bool:
        """Check if graph is bipartite."""
        return nx.is_bipartite(self._graph)
    
    def get_networkx_graph(self) -> nx.Graph:
        """Get underlying NetworkX graph."""
        return self._graph
    
    def get_max_degree(self) -> int:
        """Get maximum degree in the graph."""
        if self.num_vertices() == 0:
            return 0
        return max(dict(self._graph.degree()).values())
    
    def get_degrees(self) -> Dict:
        """Get degree of each vertex."""
        return dict(self._graph.degree())
    
    def get_components(self) -> List[List]:
        """Get connected components."""
        return [list(comp) for comp in nx.connected_components(self._graph)]
    
    def validate(self) -> Tuple[bool, Optional[str]]:
        """
        Validate graph structure.
        
        Returns:
            (is_valid, error_message)
        """
        # Check for self-loops
        for edge in self._graph.edges():
            if edge[0] == edge[1]:
                return False, f"Self-loop detected at vertex {edge[0]}"
        
        return True, None

