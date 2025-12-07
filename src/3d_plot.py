import torch
from torch_geometric.datasets import LRGBDataset
from torch_geometric.utils import to_networkx
from src.common.common_lpca import load_dataset

import networkx as nx
import plotly.graph_objects as go


def graph_to_3d_fig(g, title="Peptides-func graph (3D layout)"):
    # Convert PyG Data -> NetworkX graph
    G = to_networkx(g, to_undirected=True)

    # 3D spring layout (purely topological, not real molecule coordinates)
    pos = nx.spring_layout(G, dim=3, seed=42)

    # Extract node coordinates
    xs, ys, zs = [], [], []
    for node in G.nodes():
        x, y, z = pos[node]
        xs.append(x)
        ys.append(y)
        zs.append(z)

    # Node trace
    node_trace = go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="markers",
        marker=dict(
            size=4,
            opacity=0.8,
        ),
        # Hover shows node index
        text=[str(n) for n in G.nodes()],
        hoverinfo="text",
        name="nodes",
    )

    # Edge coordinates (as line segments)
    edge_x, edge_y, edge_z = [], [], []
    for u, v in G.edges():
        x0, y0, z0 = pos[u]
        x1, y1, z1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_z += [z0, z1, None]

    edge_trace = go.Scatter3d(
        x=edge_x,
        y=edge_y,
        z=edge_z,
        mode="lines",
        line=dict(
            width=1,
        ),
        hoverinfo="none",
        name="edges",
    )

    fig = go.Figure(data=[edge_trace, node_trace])

    fig.update_layout(
        title=title,
        showlegend=False,
        scene=dict(
            xaxis=dict(showbackground=False, showticklabels=False, visible=False),
            yaxis=dict(showbackground=False, showticklabels=False, visible=False),
            zaxis=dict(showbackground=False, showticklabels=False, visible=False),
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    return fig


def main():
    # Load Peptides-func (train split)
    dataset = load_dataset("Peptides")
    print(dataset)
    print("Number of graphs:", len(dataset))

    # Pick any graph, e.g. index 0
    g = dataset[449]
    print(g)

    fig = graph_to_3d_fig(g, title="Peptides-func – graph 0 (3D spring layout)")
    fig.show(renderer="browser")  # opens in browser / interactive window


if __name__ == "__main__":
    main()
