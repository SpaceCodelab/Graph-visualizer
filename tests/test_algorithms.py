"""
Tests for coloring algorithms.
"""

import pytest
from src.graph import Graph
from src.graph_generator import GraphGenerator
from src.algorithms import (
    GreedyColoring, WelshPowell, BacktrackingColoring,
    DSATUR, ColoringSolver
)


def test_greedy_coloring():
    """Test greedy coloring."""
    generator = GraphGenerator()
    graph = generator.complete_graph(5)
    
    greedy = GreedyColoring(graph, 'largest_first')
    coloring = greedy.solve()
    
    assert len(coloring) == 5
    is_valid, error = greedy.validate_coloring(coloring)
    assert is_valid
    assert greedy.get_num_colors() == 5


def test_welsh_powell():
    """Test Welsh-Powell algorithm."""
    generator = GraphGenerator()
    graph = generator.complete_graph(5)
    
    wp = WelshPowell(graph)
    coloring = wp.solve()
    
    assert len(coloring) == 5
    is_valid, error = wp.validate_coloring(coloring)
    assert is_valid
    assert wp.get_num_colors() == 5


def test_backtracking():
    """Test backtracking algorithm."""
    generator = GraphGenerator()
    graph = generator.complete_graph(4)
    
    bt = BacktrackingColoring(graph)
    coloring = bt.solve()
    
    assert len(coloring) == 4
    is_valid, error = bt.validate_coloring(coloring)
    assert is_valid
    assert bt.get_num_colors() == 4


def test_dsatur():
    """Test DSATUR algorithm."""
    generator = GraphGenerator()
    graph = generator.complete_graph(5)
    
    dsatur = DSATUR(graph)
    coloring = dsatur.solve()
    
    assert len(coloring) == 5
    is_valid, error = dsatur.validate_coloring(coloring)
    assert is_valid
    assert dsatur.get_num_colors() == 5


def test_coloring_solver():
    """Test ColoringSolver."""
    generator = GraphGenerator()
    graph = generator.complete_graph(5)
    
    solver = ColoringSolver(graph)
    result = solver.solve('dsatur')
    
    assert result['is_valid']
    assert result['num_colors'] == 5
    assert 'coloring' in result


def test_coloring_solver_all():
    """Test solving with all algorithms."""
    generator = GraphGenerator()
    graph = generator.complete_graph(5)
    
    solver = ColoringSolver(graph)
    results = solver.solve_all()
    
    assert len(results) > 0
    for alg, result in results.items():
        if 'error' not in result:
            assert result['is_valid']


def test_coloring_validation():
    """Test coloring validation."""
    generator = GraphGenerator()
    graph = generator.complete_graph(3)
    
    greedy = GreedyColoring(graph, 'largest_first')
    coloring = greedy.solve()
    
    # Valid coloring
    is_valid, error = greedy.validate_coloring(coloring)
    assert is_valid
    
    # Invalid coloring (force same color for adjacent vertices)
    invalid_coloring = {0: 0, 1: 0, 2: 1}
    is_valid, error = greedy.validate_coloring(invalid_coloring)
    assert not is_valid
    assert error is not None


def test_greedy_orderings():
    """Test different greedy orderings."""
    generator = GraphGenerator()
    graph = generator.complete_graph(5)
    
    for ordering in ['largest_first', 'smallest_last', 'random']:
        greedy = GreedyColoring(graph, ordering)
        coloring = greedy.solve()
        is_valid, error = greedy.validate_coloring(coloring)
        assert is_valid


def test_tree_coloring():
    """Test coloring on trees (should be 2-colorable)."""
    generator = GraphGenerator()
    graph = generator.tree_graph(10)
    
    solver = ColoringSolver(graph)
    result = solver.solve('dsatur')
    
    assert result['is_valid']
    assert result['num_colors'] <= 2


def test_bipartite_coloring():
    """Test coloring on bipartite graphs (should be 2-colorable)."""
    generator = GraphGenerator()
    graph = generator.bipartite_graph(5, 5, 1.0)
    
    solver = ColoringSolver(graph)
    result = solver.solve('dsatur')
    
    assert result['is_valid']
    assert result['num_colors'] <= 2

