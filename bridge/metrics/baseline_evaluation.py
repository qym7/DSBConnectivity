from ..analysis.rdkit_functions import BasicMolecularMetrics
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity
from types import SimpleNamespace
from scipy.optimize import linear_sum_assignment
import numpy as np


def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    nodes = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    edges = [
        (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        for bond in mol.GetBonds()
    ]
    return {"nodes": nodes, "edges": edges}


def compute_cost_alpha(d, alpha):
    """
    Compute the base cost for either nodes or edges
    based on their cardinality d and the parameter alpha
    """
    return -np.log(((d - 1) * alpha + 1) / d), -np.log((-alpha + 1) / d)


def node_cost(v1, v2, alpha, dV):
    """
    Node replacement cost function.
    If nodes are the same, return a low cost; otherwise, return the replacement cost.
    """
    if v1 == v2:
        return compute_cost_alpha(dV, alpha)[0]
    else:
        return compute_cost_alpha(dV, alpha)[1]


def edge_cost(e1, e2, alpha, dE):
    """
    Edge replacement cost function.
    If edges are the same, return a low cost; otherwise, return the replacement cost.
    """
    if e1 == e2:
        return compute_cost_alpha(dE, alpha)[0]
    else:
        return compute_cost_alpha(dE, alpha)[1]


def nll_graph_edit_distance(G1, G2, alpha):
    """
    Compute the NLL between two graphs G1 and G2.
    """
    dV = max(len(G1["nodes"]), len(G2["nodes"]))
    dE = max(len(G1["edges"]), len(G2["edges"]))

    node_cost_matrix = np.zeros((dV, dV))

    for i, v1 in enumerate(G1["nodes"]):
        for j, v2 in enumerate(G2["nodes"]):
            node_cost_matrix[i, j] = node_cost(v1, v2, alpha, dV)

    for i in range(len(G1["nodes"]), dV):
        for j in range(len(G2["nodes"]), dV):
            node_cost_matrix[i, j] = compute_cost_alpha(dV, alpha)[
                1
            ]  # Cost for dummy nodes

    edge_cost_matrix = np.zeros((dE, dE))

    G1_edges = list(G1["edges"])
    G2_edges = list(G2["edges"])

    for i, e1 in enumerate(G1_edges):
        for j, e2 in enumerate(G2_edges):
            edge_cost_matrix[i, j] = edge_cost(e1, e2, alpha, dE)

    for i in range(len(G1_edges), dE):
        for j in range(len(G2_edges), dE):
            edge_cost_matrix[i, j] = compute_cost_alpha(dE, alpha)[1]

    node_row_ind, node_col_ind = linear_sum_assignment(node_cost_matrix)
    edge_row_ind, edge_col_ind = linear_sum_assignment(edge_cost_matrix)

    total_node_cost = node_cost_matrix[node_row_ind, node_col_ind].sum()
    total_edge_cost = edge_cost_matrix[edge_row_ind, edge_col_ind].sum()

    return total_node_cost + total_edge_cost


def smiles_similarity(smiles1, smiles2):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    if mol1 is None or mol2 is None:
        raise ValueError("Invalid SMILES string(s)")

    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2)

    similarity = TanimotoSimilarity(fp1, fp2)

    return similarity


if __name__ == "__main__":
    # input from user
    dataset_name = "qm9"
    remove_h = True
    file_path = (
        "/scratch/uceeosm/EvoMol/evomol/results/test_qm9_noh/all_batches.csv"
    )
    alpha = 0.4

    data = pd.read_csv(file_path)
    original_smiles = data["Original_SMILES"].tolist()
    optimized_smiles = data["Optimized_SMILES"].to_list()

    # atom decoder
    if dataset_name == "qm9":
        if remove_h:
            atom_decoder = ["C", "N", "O", "F"]
        else:
            atom_decoder = ["H", "C", "N", "O", "F"]
    elif dataset_name == "zinc":
        atom_decoder = ["C", "F", "S", "I", "Cl", "P", "Br", "O", "N"]

    dataset_info = SimpleNamespace(atom_decoder=atom_decoder)

    metrics = BasicMolecularMetrics(
        dataset_info=dataset_info,
        test_smiles=optimized_smiles,
        train_smiles=original_smiles,
    )
    valid_optimized, _, _, _, valid_indices = metrics.evaluate_baselines(
        optimized_smiles
    )

    filtered_original_molecules = [original_smiles[i] for i in valid_indices]

    similarity = []
    graph_edit_distances = []

    for original, optimized in zip(
        filtered_original_molecules, valid_optimized
    ):
        similarity_score = smiles_similarity(original, optimized)
        change_score = 1 - similarity_score
        similarity.append(change_score)
        edit_distances = nll_graph_edit_distance(original, optimized, alpha)
        graph_edit_distances.append(edit_distances)

    similarity_mean = sum(similarity) / len(similarity)
    ged_mean = sum(graph_edit_distances) / len(graph_edit_distances)

    print(f"Average Tanimoto Distance Similarity is: ", {similarity_mean})
    print(f"NLL wrt Graph Edit Distance is: ", {ged_mean})
