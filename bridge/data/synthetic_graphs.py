import networkx as nx
import numpy as np
import scipy as sp


def generate_planar_graphs(num_graphs, min_size, max_size, seed=0):
    """Generate planar graphs using Delauney triangulation."""
    rng = np.random.default_rng(seed)
    graphs = []

    for _ in range(num_graphs):
        n = rng.integers(min_size, max_size, endpoint=True)
        points = rng.random((n, 2))
        tri = sp.spatial.Delaunay(points)
        adj = sp.sparse.lil_array((n, n), dtype=np.int32)
        for t in tri.simplices:
            adj[t[0], t[1]] = 1
            adj[t[1], t[2]] = 1
            adj[t[2], t[0]] = 1
            adj[t[1], t[0]] = 1
            adj[t[2], t[1]] = 1
            adj[t[0], t[2]] = 1
        G = nx.from_scipy_sparse_array(adj)
        graphs.append(G)

    return graphs


def generate_tree_graphs(num_graphs, min_size, max_size, seed=0):
    """Generate tree graphs using the networkx library."""
    rng = np.random.default_rng(seed)
    graphs = []

    for _ in range(num_graphs):
        n = rng.integers(min_size, max_size, endpoint=True)
        G = nx.random_tree(n, seed=rng)
        graphs.append(G)

    return graphs


def generate_sbm_graphs(
    num_graphs,
    min_num_communities,
    max_num_communities,
    min_community_size,
    max_community_size,
    intra_prob=0.005,
    inter_prob=0.3,
    seed=0,
):
    """Generate SBM graphs using the networkx library."""
    rng = np.random.default_rng(seed)
    graphs = []

    while len(graphs) < num_graphs:
        num_communities = rng.integers(
            min_num_communities, max_num_communities, endpoint=True
        )
        community_sizes = rng.integers(
            min_community_size, max_community_size, size=num_communities
        )
        probs = np.ones([num_communities, num_communities]) * intra_prob
        probs[np.arange(num_communities), np.arange(num_communities)] = inter_prob
        G = nx.stochastic_block_model(community_sizes, probs, seed=rng)
        if nx.is_connected(G):
            graphs.append(G)

    return graphs


def generate_split_sbm_graphs(
    num_graphs,
    num_communities,
    intra_prob=0.005,
    inter_prob=0.3,
    seed=0,
):
    """Generate SBM graphs using the networkx library."""
    rng = np.random.default_rng(seed)
    graphs = []

    while len(graphs) < num_graphs:
        if num_communities == 2:
            community_sizes = np.array([15, 20])
        elif num_communities == 3:# Define the arrays
            array1 = np.array([15, 10, 10])
            # Combine the arrays into a list
            # arrays = [array1, array2, array3]
            # Randomly choose one of the arrays
            # community_sizes = rng.choice(arrays)
            community_sizes = array1

        probs = np.ones([num_communities, num_communities]) * intra_prob
        probs[np.arange(num_communities), np.arange(num_communities)] = inter_prob
        probs[0, 0] = inter_prob * 2
        G = nx.stochastic_block_model(community_sizes, probs, seed=rng)
        if nx.is_connected(G):
            graphs.append(G)

    return graphs



def generate_small_split_sbm_graphs(
    num_graphs,
    num_communities,
    intra_prob=0.005,
    inter_prob=0.3,
    seed=0,
):
    """Generate SBM graphs using the networkx library."""
    rng = np.random.default_rng(seed)
    graphs = []

    while len(graphs) < num_graphs:
        if num_communities == 2:
            community_sizes = np.array([7, 10])
        elif num_communities == 3:# Define the arrays
            array1 = np.array([7, 5, 5])
            # Combine the arrays into a list
            # arrays = [array1, array2, array3]
            # Randomly choose one of the arrays
            # community_sizes = rng.choice(arrays)
            community_sizes = array1

        probs = np.ones([num_communities, num_communities]) * intra_prob
        probs[np.arange(num_communities), np.arange(num_communities)] = inter_prob
        probs[0, 0] = inter_prob * 2
        G = nx.stochastic_block_model(community_sizes, probs, seed=rng)
        if nx.is_connected(G):
            graphs.append(G)

    return graphs
def generate_sbm_graphs_fixed_size(
    num_graphs,
    num_nodes,
    min_num_communities,
    max_num_communities,
    min_community_size,
    max_community_size,
    intra_prob=0.005,
    inter_prob=0.3,
    seed=0,
):
    """Generate SBM graphs using the networkx library."""
    rng = np.random.default_rng(seed)
    graphs = []

    while len(graphs) < num_graphs:
        num_communities = rng.integers(
            min_num_communities, max_num_communities, endpoint=True
        )
        # sample community sizes under a certain graph size
        community_sizes = np.ones(num_communities) * min_community_size
        nodes_left = num_nodes - community_sizes.sum()
        max_num_nodes_to_add = max_community_size - min_community_size
        for i in range(num_communities):
            nodes_to_add_i = min(
                np.random.choice(max_num_nodes_to_add + 1, 1), nodes_left
            )
            community_sizes[i] = community_sizes[i] + nodes_to_add_i
            nodes_left = nodes_left - nodes_to_add_i

        if community_sizes.sum() != num_nodes:
            continue

        print("created", len(graphs), "graphs", num_communities, community_sizes)

        community_sizes = community_sizes.astype(int)

        probs = np.ones([num_communities, num_communities]) * intra_prob
        probs[np.arange(num_communities), np.arange(num_communities)] = inter_prob
        G = nx.stochastic_block_model(community_sizes, probs, seed=rng)
        if nx.is_connected(G):
            graphs.append(G)

    return graphs
