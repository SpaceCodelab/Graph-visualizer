"""
Graph visualization using Matplotlib and Plotly.
"""

from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .graph import Graph


class Visualizer:
    """Visualize graphs and colorings."""
    
    def __init__(self, graph: Graph):
        """
        Initialize visualizer.
        
        Args:
            graph: Graph to visualize
        """
        self.graph = graph
        self.nx_graph = graph.get_networkx_graph()
    
    def get_color_map(self, num_colors: int) -> List[str]:
        """
        Get color map for given number of colors.
        
        Args:
            num_colors: Number of colors needed
            
        Returns:
            List of color names/hex codes
        """
        # Use distinct colors
        if num_colors <= 10:
            colors = plt.cm.tab10(np.linspace(0, 1, 10))[:num_colors]
        elif num_colors <= 20:
            colors = plt.cm.tab20(np.linspace(0, 1, 20))[:num_colors]
        else:
            colors = plt.cm.rainbow(np.linspace(0, 1, num_colors))
        
        return [mcolors.rgb2hex(color) for color in colors]
    
    def visualize_coloring_matplotlib(
        self,
        coloring: Dict,
        title: str = "Graph Coloring",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8)
    ) -> None:
        """
        Visualize graph coloring using Matplotlib.
        
        Args:
            coloring: Dictionary mapping vertices to colors
            title: Plot title
            save_path: Path to save figure (None = display)
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        # Get color map
        num_colors = len(set(coloring.values()))
        color_map = self.get_color_map(num_colors)
        
        # Create color list for nodes
        node_colors = [color_map[coloring.get(node, 0) % len(color_map)] for node in self.nx_graph.nodes()]
        
        # Layout
        pos = nx.spring_layout(self.nx_graph, k=1, iterations=50)
        
        # Draw graph
        nx.draw(
            self.nx_graph,
            pos,
            with_labels=True,
            node_color=node_colors,
            node_size=1000,
            font_size=10,
            font_weight='bold',
            edge_color='gray',
            width=2,
            alpha=0.7
        )
        
        # Create legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[i],
                      markersize=15, label=f'Color {i}')
            for i in range(num_colors)
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def visualize_coloring_plotly(
        self,
        coloring: Dict,
        title: str = "Graph Coloring"
    ) -> go.Figure:
        """
        Visualize graph coloring using Plotly (interactive).
        
        Args:
            coloring: Dictionary mapping vertices to colors
            title: Plot title
            
        Returns:
            Plotly figure
        """
        # Get color map
        num_colors = len(set(coloring.values()))
        color_map = self.get_color_map(num_colors)
        
        # Layout
        pos = nx.spring_layout(self.nx_graph, k=1, iterations=50)
        
        # Create edge trace
        edge_x = []
        edge_y = []
        for edge in self.nx_graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        node_colors_list = []
        
        for node in self.nx_graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            color_idx = coloring.get(node, 0)
            node_colors_list.append(color_map[color_idx % len(color_map)])
            node_text.append(f'Node: {node}<br>Color: {color_idx}')
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[str(node) for node in self.nx_graph.nodes()],
            textposition="middle center",
            textfont=dict(size=12, color='white', family='Arial Black'),
            hovertext=node_text,
            marker=dict(
                showscale=False,
                color=node_colors_list,
                size=30,
                line=dict(width=2, color='white')
            )
        )
        
        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=dict(
                    text=title,
                    x=0.5,
                    font=dict(size=20)
                ),
                showlegend=True,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=[
                    dict(
                        text=f"Number of colors: {num_colors}",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002,
                        xanchor='left', yanchor='bottom',
                        font=dict(size=12)
                    )
                ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
        )
        
        # Add legend
        for i in range(num_colors):
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=15, color=color_map[i]),
                name=f'Color {i}',
                showlegend=True
            ))
        
        return fig
    
    def visualize_uncolored_matplotlib(
        self,
        title: str = "Graph",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8)
    ) -> None:
        """Visualize uncolored graph using Matplotlib."""
        plt.figure(figsize=figsize)
        
        pos = nx.spring_layout(self.nx_graph, k=1, iterations=50)
        
        nx.draw(
            self.nx_graph,
            pos,
            with_labels=True,
            node_color='lightblue',
            node_size=1000,
            font_size=10,
            font_weight='bold',
            edge_color='gray',
            width=2,
            alpha=0.7
        )
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def visualize_uncolored_plotly(self, title: str = "Graph") -> go.Figure:
        """Visualize uncolored graph using Plotly."""
        pos = nx.spring_layout(self.nx_graph, k=1, iterations=50)
        
        # Create edge trace
        edge_x = []
        edge_y = []
        for edge in self.nx_graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        
        for node in self.nx_graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(f'Node: {node}<br>Degree: {self.nx_graph.degree(node)}')
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[str(node) for node in self.nx_graph.nodes()],
            textposition="middle center",
            textfont=dict(size=12, color='white', family='Arial Black'),
            hovertext=node_text,
            marker=dict(
                showscale=False,
                color='lightblue',
                size=30,
                line=dict(width=2, color='white')
            )
        )
        
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=dict(
                    text=title,
                    x=0.5,
                    font=dict(size=20)
                ),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
        )
        
        return fig

