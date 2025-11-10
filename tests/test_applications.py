"""
Tests for real-world applications.
"""

import pytest
import numpy as np
from src.applications import (
    RegisterAllocation, ExamScheduling, MapColoring,
    FrequencyAssignment, SudokuSolver
)


def test_register_allocation():
    """Test register allocation."""
    variables = ['a', 'b', 'c', 'd']
    conflicts = [('a', 'b'), ('b', 'c'), ('c', 'd'), ('d', 'a')]
    
    app = RegisterAllocation(variables, conflicts)
    result = app.solve()
    
    assert 'registers' in result
    assert 'num_registers' in result
    assert result['num_registers'] >= 2


def test_exam_scheduling():
    """Test exam scheduling."""
    courses = ['Math', 'Physics', 'Chemistry']
    students = {
        'Alice': ['Math', 'Physics'],
        'Bob': ['Physics', 'Chemistry'],
        'Charlie': ['Math', 'Chemistry']
    }
    
    app = ExamScheduling(courses, students)
    result = app.solve()
    
    assert 'schedule' in result
    assert 'num_time_slots' in result
    assert result['num_time_slots'] >= 2


def test_map_coloring():
    """Test map coloring."""
    regions = ['A', 'B', 'C', 'D']
    borders = [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'A')]
    
    app = MapColoring(regions, borders)
    result = app.solve()
    
    assert 'region_colors' in result
    assert 'num_colors' in result
    assert result['num_colors'] >= 2


def test_frequency_assignment():
    """Test frequency assignment."""
    stations = ['S1', 'S2', 'S3', 'S4']
    interference = [('S1', 'S2'), ('S2', 'S3'), ('S3', 'S4'), ('S4', 'S1')]
    
    app = FrequencyAssignment(stations, interference)
    result = app.solve()
    
    assert 'frequencies' in result
    assert 'num_frequencies' in result
    assert result['num_frequencies'] >= 2


def test_sudoku_solver():
    """Test Sudoku solver."""
    # Simple Sudoku puzzle
    grid = np.array([
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ])
    
    app = SudokuSolver(grid)
    solution = app.solve()
    
    assert solution is not None
    assert np.all(solution > 0)
    assert np.all(solution <= 9)

