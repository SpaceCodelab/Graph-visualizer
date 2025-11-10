"""
Performance benchmarking and comparison module.
"""

import time
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from .graph import Graph
from .algorithms import ColoringSolver
from .graph_properties import GraphProperties


class Benchmarker:
    """Benchmark and compare graph coloring algorithms."""
    
    def __init__(self, graph: Graph):
        """
        Initialize benchmarker.
        
        Args:
            graph: Graph to benchmark
        """
        self.graph = graph
        self.solver = ColoringSolver(graph)
        self.properties = GraphProperties(graph)
        self.results: List[Dict] = []
    
    def benchmark_all(self) -> pd.DataFrame:
        """
        Benchmark all algorithms on the graph.
        
        Returns:
            DataFrame with benchmark results
        """
        results = []
        
        for algorithm in self.solver.ALGORITHMS.keys():
            try:
                result = self.solver.solve(algorithm)
                
                # Ensure all required fields are present
                num_colors = result.get('num_colors', None)
                exec_time = result.get('execution_time', None)
                is_valid = result.get('is_valid', False)
                error = result.get('error', None)
                
                results.append({
                    'algorithm': algorithm,
                    'num_colors': num_colors if num_colors is not None else None,
                    'execution_time': exec_time if exec_time is not None else None,
                    'is_valid': is_valid,
                    'error': error
                })
            except Exception as e:
                import traceback
                error_msg = f"{str(e)}\n{traceback.format_exc()}"
                results.append({
                    'algorithm': algorithm,
                    'num_colors': None,
                    'execution_time': None,
                    'is_valid': False,
                    'error': error_msg
                })
        
        self.results = results
        df = pd.DataFrame(results)
        return df
    
    def compare_algorithms(self, algorithms: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compare specific algorithms.
        
        Args:
            algorithms: List of algorithm names (None = all)
            
        Returns:
            DataFrame with comparison results
        """
        if algorithms is None:
            algorithms = list(self.solver.ALGORITHMS.keys())
        
        results = []
        for algorithm in algorithms:
            try:
                result = self.solver.solve(algorithm)
                results.append({
                    'algorithm': algorithm,
                    'num_colors': result['num_colors'],
                    'execution_time': result['execution_time'],
                    'is_valid': result['is_valid']
                })
            except Exception as e:
                results.append({
                    'algorithm': algorithm,
                    'num_colors': None,
                    'execution_time': None,
                    'is_valid': False,
                    'error': str(e)
                })
        
        return pd.DataFrame(results)
    
    def plot_comparison_matplotlib(self, save_path: Optional[str] = None) -> None:
        """
        Plot comparison charts using Matplotlib.
        
        Args:
            save_path: Path to save figure (None = display)
        """
        if not self.results:
            self.benchmark_all()
        
        df = pd.DataFrame(self.results)
        valid_df = df[df['is_valid'] == True]
        
        if len(valid_df) == 0:
            print("No valid results to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot number of colors
        ax1.bar(valid_df['algorithm'], valid_df['num_colors'], color='skyblue', edgecolor='black')
        ax1.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Number of Colors', fontsize=12, fontweight='bold')
        ax1.set_title('Number of Colors Comparison', fontsize=14, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot execution time
        ax2.bar(valid_df['algorithm'], valid_df['execution_time'], color='lightcoral', edgecolor='black')
        ax2.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
        ax2.set_title('Execution Time Comparison', fontsize=14, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def plot_comparison_plotly(self) -> go.Figure:
        """
        Plot comparison charts using Plotly (interactive).
        
        Returns:
            Plotly figure
        """
        if not self.results:
            self.benchmark_all()
        
        df = pd.DataFrame(self.results)
        
        # Filter valid results and handle NaN values
        valid_df = df[df['is_valid'] == True].copy()
        valid_df = valid_df.dropna(subset=['num_colors', 'execution_time'])
        
        if len(valid_df) == 0:
            # Return empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="No valid results to display",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16)
            )
            return fig
        
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Number of Colors Comparison', 'Execution Time Comparison'),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Number of colors - ensure values are numeric
        num_colors = pd.to_numeric(valid_df['num_colors'], errors='coerce').fillna(0).astype(int)
        fig.add_trace(
            go.Bar(
                x=valid_df['algorithm'],
                y=num_colors,
                name='Number of Colors',
                marker_color='skyblue',
                text=num_colors.astype(str),
                textposition='outside'
            ),
            row=1, col=1
        )
        
        # Execution time - ensure values are numeric
        exec_times = pd.to_numeric(valid_df['execution_time'], errors='coerce').fillna(0.0)
        fig.add_trace(
            go.Bar(
                x=valid_df['algorithm'],
                y=exec_times,
                name='Execution Time (s)',
                marker_color='lightcoral',
                text=[f"{t:.4f}" if t > 0 else "N/A" for t in exec_times],
                textposition='outside'
            ),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="Algorithm", row=1, col=1)
        fig.update_xaxes(title_text="Algorithm", row=1, col=2)
        fig.update_yaxes(title_text="Number of Colors", row=1, col=1)
        fig.update_yaxes(title_text="Execution Time (seconds)", row=1, col=2)
        
        fig.update_layout(
            title_text="Algorithm Comparison",
            showlegend=False,
            height=500
        )
        
        return fig
    
    def export_to_csv(self, filepath: str) -> None:
        """
        Export benchmark results to CSV.
        
        Args:
            filepath: Path to save CSV file
        """
        if not self.results:
            self.benchmark_all()
        
        df = pd.DataFrame(self.results)
        
        # Add graph properties
        props = self.properties.get_all_properties()
        for key, value in props.items():
            df[f'graph_{key}'] = value
        
        df.to_csv(filepath, index=False)
        print(f"Results exported to {filepath}")
    
    def export_to_pdf(self, filepath: str) -> None:
        """
        Export benchmark results to PDF.
        
        Args:
            filepath: Path to save PDF file
        """
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.lib import colors
            
            doc = SimpleDocTemplate(filepath, pagesize=letter)
            story = []
            styles = getSampleStyleSheet()
            
            # Title
            title = Paragraph("Graph Coloring Benchmark Results", styles['Title'])
            story.append(title)
            story.append(Spacer(1, 12))
            
            # Graph properties
            props = self.properties.get_all_properties()
            props_text = f"<b>Graph Properties:</b><br/>"
            props_text += f"Vertices: {props['num_vertices']}<br/>"
            props_text += f"Edges: {props['num_edges']}<br/>"
            props_text += f"Max Degree: {props['max_degree']}<br/>"
            props_text += f"Connected: {props['is_connected']}<br/>"
            props_text += f"Chromatic Lower Bound: {props['chromatic_lower_bound']}<br/>"
            props_text += f"Chromatic Upper Bound: {props['chromatic_upper_bound']}<br/>"
            
            story.append(Paragraph(props_text, styles['Normal']))
            story.append(Spacer(1, 12))
            
            # Results table
            if not self.results:
                self.benchmark_all()
            
            df = pd.DataFrame(self.results)
            data = [['Algorithm', 'Num Colors', 'Time (s)', 'Valid']]
            
            for _, row in df.iterrows():
                data.append([
                    row['algorithm'],
                    str(row['num_colors']) if pd.notna(row['num_colors']) else 'N/A',
                    f"{row['execution_time']:.4f}" if pd.notna(row['execution_time']) else 'N/A',
                    str(row['is_valid'])
                ])
            
            table = Table(data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(table)
            doc.build(story)
            print(f"Results exported to {filepath}")
        except ImportError:
            print("reportlab not available. Install with: pip install reportlab")
        except Exception as e:
            print(f"Error exporting to PDF: {e}")

