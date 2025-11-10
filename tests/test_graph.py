"""
Tests for Graph class.
"""

import pytest
import numpy as np
from src.graph import Graph
from src.graph_generator import GraphGenerator


def test_graph_creation():
    """Test graph creation."""
    graph = Graph()
    assert graph.num_vertices() == 0
    assert graph.num_edges() == 0


def test_graph_from_edge_list():
    """Test graph creation from edge list."""
    edges = [(0, 1), (1, 2), (2, 0)]
    graph = Graph.from_edge_list(edges)
    assert graph.num_vertices() == 3
    assert graph.num_edges() == 3


def test_graph_from_adjacency_matrix():
    """Test graph creation from adjacency matrix."""
    matrix = np.array([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0]
    ])
    graph = Graph.from_adjacency_matrix(matrix)
    assert graph.num_vertices() == 3
    assert graph.num_edges() == 3


def test_graph_from_adjacency_list():
    """Test graph creation from adjacency list."""
    adj_list = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
    graph = Graph.from_adjacency_list(adj_list)
    assert graph.num_vertices() == 3
    assert graph.num_edges() == 3


def test_add_remove_vertex():
    """Test adding and removing vertices."""
    graph = Graph()
    graph.add_vertex(0)
    graph.add_vertex(1)
    assert graph.num_vertices() == 2
    
    graph.remove_vertex(0)
    assert graph.num_vertices() == 1


def test_add_remove_edge():
    """Test adding and removing edges."""
    graph = Graph()
    graph.add_vertex(0)
    graph.add_vertex(1)
    graph.add_edge(0, 1)
    assert graph.num_edges() == 1
    
    graph.remove_edge(0, 1)
    assert graph.num_edges() == 0


def test_validate():
    """Test graph validation."""
    graph = Graph()
    graph.add_vertex(0)
    graph.add_vertex(1)
    graph.add_edge(0, 1)
    
    is_valid, error = graph.validate()
    assert is_valid
    assert error is None


def test_complete_graph():
    """Test complete graph generation."""
    generator = GraphGenerator()
    graph = generator.complete_graph(5)
    assert graph.num_vertices() == 5
    assert graph.num_edges() == 10  # C(5,2) = 10


def test_cycle_graph():
    """Test cycle graph generation."""
    generator = GraphGenerator()
    graph = generator.cycle_graph(5)
    assert graph.num_vertices() == 5
    assert graph.num_edges() == 5


def test_bipartite_graph():
    """Test bipartite graph generation."""
    generator = GraphGenerator()
    graph = generator.bipartite_graph(3, 3, 1.0)
    assert graph.num_vertices() == 6
    assert graph.num_edges() == 9  # Complete bipartite K_{3,3}


def test_is_connected():
    """Test connectivity check."""
    graph = Graph.from_edge_list([(0, 1), (1, 2)])
    assert graph.is_connected()
    
    graph = Graph.from_edge_list([(0, 1), (2, 3)])
    assert not graph.is_connected()


def test_is_bipartite():
    """Test bipartite check."""
    generator = GraphGenerator()
    graph = generator.bipartite_graph(3, 3, 1.0)
    assert graph.is_bipartite()
    
    graph = generator.complete_graph(3)
    assert not graph.is_bipartite()

