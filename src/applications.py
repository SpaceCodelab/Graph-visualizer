"""
Real-world applications of graph coloring.
"""

from typing import Dict, List, Tuple, Optional, Set
import numpy as np
from .graph import Graph
from .algorithms import ColoringSolver
from .graph_generator import GraphGenerator


class RegisterAllocation:
    """Register allocation simulation using graph coloring."""
    
    def __init__(self, variables: List[str], conflicts: List[Tuple[str, str]]):
        """
        Initialize register allocation problem.
        
        Args:
            variables: List of variable names
            conflicts: List of tuples (var1, var2) indicating variables that conflict
        """
        self.variables = variables
        self.conflicts = conflicts
        self.graph = self._build_graph()
        self.solver = ColoringSolver(self.graph)
    
    def _build_graph(self) -> Graph:
        """Build conflict graph."""
        edges = []
        for var1, var2 in self.conflicts:
            edges.append((var1, var2))
        return Graph.from_edge_list(edges)
    
    def solve(self, algorithm: str = 'dsatur') -> Dict:
        """
        Solve register allocation.
        
        Args:
            algorithm: Coloring algorithm to use
            
        Returns:
            Dictionary with register assignment
        """
        result = self.solver.solve(algorithm)
        coloring = result['coloring']
        
        # Map colors to register names
        registers = {}
        for var, color in coloring.items():
            registers[var] = f'R{color}'
        
        return {
            'registers': registers,
            'num_registers': result['num_colors'],
            'coloring': coloring,
            'algorithm': algorithm
        }


class ExamScheduling:
    """Exam scheduling using graph coloring."""
    
    def __init__(self, courses: List[str], students: Dict[str, List[str]]):
        """
        Initialize exam scheduling problem.
        
        Args:
            courses: List of course names
            students: Dictionary mapping student ID to list of courses they take
        """
        self.courses = courses
        self.students = students
        self.graph = self._build_graph()
        self.solver = ColoringSolver(self.graph)
    
    def _build_graph(self) -> Graph:
        """Build conflict graph (courses that share students)."""
        course_pairs = set()
        
        # Find courses that share at least one student
        student_courses = {}
        for student, courses in self.students.items():
            for course in courses:
                if course not in student_courses:
                    student_courses[course] = set()
                student_courses[course].add(student)
        
        # Create edges between conflicting courses
        courses_list = list(student_courses.keys())
        for i, course1 in enumerate(courses_list):
            for course2 in courses_list[i+1:]:
                if student_courses[course1] & student_courses[course2]:  # Intersection
                    course_pairs.add((course1, course2))
        
        edges = list(course_pairs)
        return Graph.from_edge_list(edges)
    
    def solve(self, algorithm: str = 'dsatur') -> Dict:
        """
        Solve exam scheduling.
        
        Args:
            algorithm: Coloring algorithm to use
            
        Returns:
            Dictionary with exam schedule
        """
        result = self.solver.solve(algorithm)
        coloring = result['coloring']
        
        # Map colors to time slots
        schedule = {}
        for course, color in coloring.items():
            schedule[course] = f'Time Slot {color + 1}'
        
        return {
            'schedule': schedule,
            'num_time_slots': result['num_colors'],
            'coloring': coloring,
            'algorithm': algorithm
        }


class MapColoring:
    """Map coloring problem."""
    
    def __init__(self, regions: List[str], borders: List[Tuple[str, str]]):
        """
        Initialize map coloring problem.
        
        Args:
            regions: List of region names
            borders: List of tuples (region1, region2) indicating shared borders
        """
        self.regions = regions
        self.borders = borders
        self.graph = self._build_graph()
        self.solver = ColoringSolver(self.graph)
    
    def _build_graph(self) -> Graph:
        """Build adjacency graph."""
        edges = list(self.borders)
        return Graph.from_edge_list(edges)
    
    def solve(self, algorithm: str = 'dsatur') -> Dict:
        """
        Solve map coloring.
        
        Args:
            algorithm: Coloring algorithm to use
            
        Returns:
            Dictionary with region colors
        """
        result = self.solver.solve(algorithm)
        coloring = result['coloring']
        
        # Map colors to color names
        color_names = ['Red', 'Blue', 'Green', 'Yellow', 'Orange', 'Purple', 'Pink', 'Cyan']
        region_colors = {}
        for region, color in coloring.items():
            region_colors[region] = color_names[color % len(color_names)]
        
        return {
            'region_colors': region_colors,
            'num_colors': result['num_colors'],
            'coloring': coloring,
            'algorithm': algorithm
        }


class FrequencyAssignment:
    """Frequency assignment in wireless networks."""
    
    def __init__(self, stations: List[str], interference: List[Tuple[str, str]]):
        """
        Initialize frequency assignment problem.
        
        Args:
            stations: List of station names
            interference: List of tuples (station1, station2) indicating interference
        """
        self.stations = stations
        self.interference = interference
        self.graph = self._build_graph()
        self.solver = ColoringSolver(self.graph)
    
    def _build_graph(self) -> Graph:
        """Build interference graph."""
        edges = list(self.interference)
        return Graph.from_edge_list(edges)
    
    def solve(self, algorithm: str = 'dsatur') -> Dict:
        """
        Solve frequency assignment.
        
        Args:
            algorithm: Coloring algorithm to use
            
        Returns:
            Dictionary with frequency assignment
        """
        result = self.solver.solve(algorithm)
        coloring = result['coloring']
        
        # Map colors to frequencies
        frequencies = {}
        base_frequency = 900  # MHz
        for station, color in coloring.items():
            frequencies[station] = base_frequency + color * 10  # 10 MHz spacing
        
        return {
            'frequencies': frequencies,
            'num_frequencies': result['num_colors'],
            'coloring': coloring,
            'algorithm': algorithm
        }


class SudokuSolver:
    """Sudoku solver using graph coloring concepts."""
    
    def __init__(self, grid: Optional[np.ndarray] = None):
        """
        Initialize Sudoku solver.
        
        Args:
            grid: 9x9 numpy array with initial values (0 = empty)
        """
        if grid is None:
            self.grid = np.zeros((9, 9), dtype=int)
        else:
            self.grid = grid.copy()
    
    def _build_graph(self) -> Graph:
        """Build constraint graph for Sudoku."""
        # Each cell is a vertex
        # Edges connect cells in same row, column, or 3x3 box
        edges = []
        
        for i in range(9):
            for j in range(9):
                cell1 = (i, j)
                # Same row
                for k in range(9):
                    if k != j:
                        edges.append((cell1, (i, k)))
                # Same column
                for k in range(9):
                    if k != i:
                        edges.append((cell1, (k, j)))
                # Same 3x3 box
                box_i = (i // 3) * 3
                box_j = (j // 3) * 3
                for bi in range(box_i, box_i + 3):
                    for bj in range(box_j, box_j + 3):
                        if (bi, bj) != (i, j):
                            edges.append((cell1, (bi, bj)))
        
        # Remove duplicates
        edges = list(set(edges))
        return Graph.from_edge_list(edges)
    
    def solve(self) -> Optional[np.ndarray]:
        """
        Solve Sudoku using backtracking.
        
        Returns:
            Solved grid or None if unsolvable
        """
        return self._backtrack(self.grid.copy())
    
    def _backtrack(self, grid: np.ndarray) -> Optional[np.ndarray]:
        """Backtracking helper."""
        # Find empty cell
        empty = self._find_empty(grid)
        if empty is None:
            return grid
        
        row, col = empty
        
        # Try each number
        for num in range(1, 10):
            if self._is_valid(grid, row, col, num):
                grid[row, col] = num
                result = self._backtrack(grid)
                if result is not None:
                    return result
                grid[row, col] = 0
        
        return None
    
    def _find_empty(self, grid: np.ndarray) -> Optional[Tuple[int, int]]:
        """Find empty cell."""
        for i in range(9):
            for j in range(9):
                if grid[i, j] == 0:
                    return (i, j)
        return None
    
    def _is_valid(self, grid: np.ndarray, row: int, col: int, num: int) -> bool:
        """Check if number is valid in position."""
        # Check row
        if num in grid[row, :]:
            return False
        
        # Check column
        if num in grid[:, col]:
            return False
        
        # Check 3x3 box
        box_i = (row // 3) * 3
        box_j = (col // 3) * 3
        if num in grid[box_i:box_i+3, box_j:box_j+3]:
            return False
        
        return True

