import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from torch_geometric.utils import from_networkx
from torch_geometric.nn import GCNConv


############################################
# Universe Graph Generator
############################################

def generate_universe(
        n_nodes=400,
        visible_prob=0.02,
        hidden_prob=0.01,
        feature_dim=16):

    """
    Creates a universe graph.

    visible_prob -> normal interactions
    hidden_prob  -> dark matter connectivity
    """

    # visible interactions
    G = nx.erdos_renyi_graph(n_nodes, visible_prob)

    # hidden edges (dark matter)
    hidden_edges = int(hidden_prob * n_nodes)

    for _ in range(hidden_edges):

        a = np.random.randint(0, n_nodes)
        b = np.random.randint(0, n_nodes)

        G.add_edge(a, b)

    data = from_networkx(G)

    # node states (physical fields)
    data.x = torch.randn(n_nodes, feature_dim)

    return data


############################################
# Graph Neural Universe Model
############################################

class UniverseGNN(torch.nn.Module):

    """
    Implements the update rule

    H_{t+1} = σ(A H W)

    from the paper.
    """

    def __init__(self, feature_dim=16, hidden=64):

        super().__init__()

        self.conv1 = GCNConv(feature_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.conv3 = GCNConv(hidden, feature_dim)

    def forward(self, data):

        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = self.conv3(x, edge_index)

        return x


############################################
# Training
############################################

def train_universe(model, data, epochs=200):

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):

        optimizer.zero_grad()

        out = model(data)

        # stability loss
        loss = ((out - data.x) ** 2).mean()

        loss.backward()

        optimizer.step()

        if epoch % 20 == 0:
            print("Epoch:", epoch, "Loss:", loss.item())


############################################
# Universe Simulation
############################################

def simulate_universe(model, data, steps=20):

    states = []

    x = data.x

    for step in range(steps):

        data.x = x

        x = model(data)

        states.append(x.detach())

    return states


############################################
# Visualization
############################################

def visualize_graph(data):

    G = nx.Graph()

    edges = data.edge_index.numpy()

    for i in range(edges.shape[1]):
        G.add_edge(int(edges[0, i]), int(edges[1, i]))

    plt.figure(figsize=(7,7))

    nx.draw(G, node_size=10)

    plt.title("Universe Interaction Graph")

    plt.show()


############################################
# Clustering Analysis
############################################

def clustering_analysis(data):

    G = nx.Graph()

    edges = data.edge_index.numpy()

    for i in range(edges.shape[1]):
        G.add_edge(int(edges[0,i]), int(edges[1,i]))

    clustering = nx.clustering(G)

    avg = sum(clustering.values()) / len(clustering)

    print("Average clustering coefficient:", avg)

    return avg


############################################
# Main
############################################

def main():

    print("\nGenerating universe graph...")

    data = generate_universe()

    visualize_graph(data)

    print("\nBuilding universe neural model...")

    model = UniverseGNN()

    print("\nTraining universe dynamics...")

    train_universe(model, data)

    print("\nSimulating universe evolution...")

    states = simulate_universe(model, data)

    print("Simulation steps:", len(states))

    print("\nAnalyzing cosmic clustering...")

    clustering_analysis(data)


if __name__ == "__main__":
    main()
