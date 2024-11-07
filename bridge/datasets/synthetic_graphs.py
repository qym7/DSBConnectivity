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


def generate_planar_edge_remove_graphs(
    num_graphs, num_nodes, edge_removal, degree=False, seed=0
):
    """Generate planar graphs using Delauney triangulation (with or without edge removal)."""
    rng = np.random.default_rng(seed)
    graphs = []

    for _ in range(num_graphs):
        points = rng.random((num_nodes, 2))
        tri = sp.spatial.Delaunay(points)
        adj = sp.sparse.lil_array((num_nodes, num_nodes), dtype=np.int32)

        for t in tri.simplices:
            adj[t[0], t[1]] = 1
            adj[t[1], t[2]] = 1
            adj[t[2], t[0]] = 1
            adj[t[1], t[0]] = 1
            adj[t[2], t[1]] = 1
            adj[t[0], t[2]] = 1

        G = nx.from_scipy_sparse_array(adj)

        if edge_removal != 0:
            # calculate the number of edges to remove
            num_edges = G.number_of_edges()
            print("before removal", num_edges)
            edges_to_remove = int(num_edges * (edge_removal / 100.0))

            if degree:
                removed_edges = 0
                while removed_edges < edges_to_remove:
                    # sort edges by the sum of degrees of their endpoints in descending order
                    edges = sorted(
                        G.edges(),
                        key=lambda e: G.degree[e[0]] + G.degree[e[1]],
                        reverse=True,
                    )

                    for edge in edges:
                        G.remove_edge(*edge)
                        if nx.is_connected(G) and nx.check_planarity(G)[0]:
                            removed_edges += 1
                            if removed_edges >= edges_to_remove:
                                break
                        else:
                            G.add_edge(
                                *edge
                            )  # add the edge again if removing it breaks connectivity or planarity
                    if removed_edges >= edges_to_remove:
                        break
            else:
                # randomly remove edges while maintaining planarity
                edges = list(G.edges())
                rng.shuffle(edges)

                for _ in range(edges_to_remove):
                    edge = edges.pop()
                    G.remove_edge(*edge)
                    if not nx.check_planarity(G)[0]:
                        G.add_edge(
                            *edge
                        )  # add the edge again if removing it breaks planarity

            new_num_edges = G.number_of_edges()
            print("after removal", new_num_edges)
            print("removed proportion", (num_edges - new_num_edges) / num_edges)
        graphs.append(G)

    return graphs


def generate_planar_edge_add_graphs(
    num_graphs, num_nodes, avg_degree=3, shortest_path=False, seed=0
):
    """Generate planar graphs using Delaunay triangulation and add edges based on shortest path distance."""
    rng = np.random.default_rng(seed)
    graphs = []

    for _ in range(num_graphs):
        points = rng.random((num_nodes, 2))
        tri = sp.spatial.Delaunay(points)
        adj = sp.sparse.lil_matrix((num_nodes, num_nodes), dtype=np.int32)

        for t in tri.simplices:
            adj[t[0], t[1]] = 1
            adj[t[1], t[2]] = 1
            adj[t[2], t[0]] = 1
            adj[t[1], t[0]] = 1
            adj[t[2], t[1]] = 1
            adj[t[0], t[2]] = 1

        G = nx.from_scipy_sparse_matrix(adj)

        if avg_degree > 0:
            if shortest_path:
                # Add edges based on shortest path distance
                potential_edges = [
                    (u, v)
                    for u in range(num_nodes)
                    for v in range(u + 1, num_nodes)
                    if not G.has_edge(u, v)
                ]
                potential_edges = sorted(
                    potential_edges,
                    key=lambda edge: nx.shortest_path_length(
                        G, edge[0], edge[1]
                    ),
                )

                for u, v in potential_edges:
                    G.add_edge(u, v)
                    if not nx.check_planarity(G)[0]:
                        G.remove_edge(u, v)
            else:
                # Add edges to increase density while maintaining average node degree
                max_edges = avg_degree * num_nodes // 2
                potential_edges = [
                    (u, v)
                    for u in range(num_nodes)
                    for v in range(u + 1, num_nodes)
                    if not G.has_edge(u, v)
                ]
                rng.shuffle(potential_edges)

                for u, v in potential_edges:
                    if G.number_of_edges() >= max_edges:
                        break
                    G.add_edge(u, v)
                    if (
                        not nx.check_planarity(G)[0]
                        or G.number_of_edges() > max_edges
                    ):
                        G.remove_edge(u, v)

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
        probs[np.arange(num_communities), np.arange(num_communities)] = (
            inter_prob
        )
        G = nx.stochastic_block_model(community_sizes, probs, seed=rng)
        if nx.is_connected(G):
            graphs.append(G)

    return graphs


def generate_split_2_to_3_sbm_graphs(
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
        elif num_communities == 3:  # Define the arrays
            array1 = np.array([15, 10, 10])
            community_sizes = array1

        probs = np.ones([num_communities, num_communities]) * intra_prob
        probs[np.arange(num_communities), np.arange(num_communities)] = (
            inter_prob
        )
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
        elif num_communities == 3:  # Define the arrays
            array1 = np.array([7, 5, 5])
            # Combine the arrays into a list
            # arrays = [array1, array2, array3]
            # Randomly choose one of the arrays
            # community_sizes = rng.choice(arrays)
            community_sizes = array1

        probs = np.ones([num_communities, num_communities]) * intra_prob
        probs[np.arange(num_communities), np.arange(num_communities)] = (
            inter_prob
        )
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

        print(
            "created", len(graphs), "graphs", num_communities, community_sizes
        )

        community_sizes = community_sizes.astype(int)

        probs = np.ones([num_communities, num_communities]) * intra_prob
        probs[np.arange(num_communities), np.arange(num_communities)] = (
            inter_prob
        )
        G = nx.stochastic_block_model(community_sizes, probs, seed=rng)
        if nx.is_connected(G):
            graphs.append(G)

    return graphs
