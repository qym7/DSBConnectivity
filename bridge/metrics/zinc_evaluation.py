import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.metrics import mean_absolute_error
from rdkit import Chem
from rdkit.Chem import QED, Descriptors, RDConfig, rdchem
import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
import pandas as pd
from scipy.optimize import linear_sum_assignment
from fcd_torch import FCD

def mol2smiles(mol):
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return None
    return Chem.MolToSmiles(mol)

def compute_validity_smiles(generated):
    valid = []
    num_components = []
    all_smiles = []
    valid_indices = []

    for idx, smiles in enumerate(generated):
        try:
            mol = Chem.MolFromSmiles(smiles)
            smiles = mol2smiles(mol)
        except:
            pass
        try:
            mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
            num_components.append(len(mol_frags))
        except:
            pass

        if smiles is not None and not pd.isna(smiles):
            try:
                mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
                largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
                smiles = mol2smiles(largest_mol)
                valid.append(smiles)
                all_smiles.append(smiles)
                valid_indices.append(idx)  # Record index of valid molecule
            except Chem.rdchem.AtomValenceException:
                print("Valence error in GetMolFrags")
                all_smiles.append(None)
            except Chem.rdchem.KekulizeException:
                print("Can't kekulize molecule")
                all_smiles.append(None)
        else:
            all_smiles.append(None)
    return valid, len(valid) / len(generated), np.array(num_components), all_smiles, valid_indices

def compute_relaxed_validity_smiles(generated):
    valid = []
    for smiles in generated:
        try:
            mol = Chem.MolFromSmiles(smiles)
            smiles = mol2smiles(mol)
        except:
            pass
        if smiles is not None and not pd.isna(smiles):
            try:
                mol_frags = Chem.rdmolops.GetMolFrags(
                    mol, asMols=True, sanitizeFrags=True
                )
                largest_mol = max(
                    mol_frags, default=mol, key=lambda m: m.GetNumAtoms()
                )
                smiles = mol2smiles(largest_mol)
                valid.append(smiles)
            except Chem.rdchem.AtomValenceException:
                print("Valence error in GetmolFrags")
            except Chem.rdchem.KekulizeException:
                print("Can't kekulize molecule")
    return valid, len(valid) / len(generated)

def compute_w1(property_real, property_generated):
    return wasserstein_distance(property_real, property_generated)

def compute_mad(property_real, property_generated):
    min_length = min(len(property_real), len(property_generated))
    property_real_new = property_real[:min_length]
    property_generated_new = property_generated[:min_length]
    return mean_absolute_error(property_real_new, property_generated_new)

def compute_molecular_properties(molecules):
    logP_values = []
    qed_values = []
    sa_scores = []

    for smiles in molecules:
        mol = Chem.MolFromSmiles(smiles)
        logP_values.append(Descriptors.MolLogP(mol))
        qed_values.append(QED.qed(mol))
        sa_scores.append(sascorer.calculateScore(mol))

    return np.array(logP_values), np.array(qed_values), np.array(sa_scores)

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    nodes = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    edges = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds()]
    return {'nodes': nodes, 'edges': edges}

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
    dV = max(len(G1['nodes']), len(G2['nodes']))
    dE = max(len(G1['edges']), len(G2['edges']))

    node_cost_matrix = np.zeros((dV, dV))

    for i, v1 in enumerate(G1['nodes']):
        for j, v2 in enumerate(G2['nodes']):
            node_cost_matrix[i, j] = node_cost(v1, v2, alpha, dV)

    for i in range(len(G1['nodes']), dV):
        for j in range(len(G2['nodes']), dV):
            node_cost_matrix[i, j] = compute_cost_alpha(dV, alpha)[1]

    edge_cost_matrix = np.zeros((dE, dE))

    G1_edges = list(G1['edges'])
    G2_edges = list(G2['edges'])

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

def nll_total_graph_edit_distance(source, target, alpha):
    nll_total = []
    for smiles1, smiles2 in zip(source, target):
        try:
            G1 = smiles_to_graph(smiles1)
            G2 = smiles_to_graph(smiles2)
            nll = nll_graph_edit_distance(G1, G2, alpha)
            nll_total.append(nll)
        except:
            pass

    return sum(nll_total) / len(nll_total)

if __name__ == "__main__":
    real_smiles = '/scratch/uceeosm/DSBConnectivity/data/zinc/zinc_source/zinc_pyg/raw/test_zinc.csv'
    gen_smiles_path = '/scratch/uceeosm/DSBConnectivity/experiments/2024-10-15/zinc_gen_backward_114/17-19-51'
    gen_smiles = os.path.join(gen_smiles_path, 'final_smiles.csv')
    gen_smiles_optimized = os.path.join(gen_smiles_path, 'SA_values.csv')

    alpha = 0.4

    real_data = pd.read_csv(real_smiles)
    gen_data = pd.read_csv(gen_smiles)

    real_smiles_data = real_data['SMILES'].tolist()
    gen_smiles_data = gen_data['SMILES'].to_list()

    gen_valid, gen_validity, _, _, _ = compute_validity_smiles(gen_smiles_data)
    print(f"Validity over {len(gen_smiles_data)} molecules: {gen_validity * 100 :.2f}%")

    gen_valid_relax, gen_validity_relax = compute_relaxed_validity_smiles(gen_smiles_data)
    print(f"Relaxed Validity over {len(gen_smiles_data)} molecules: {gen_validity_relax * 100 :.2f}%")

    logP_real, qed_real, sa_real = compute_molecular_properties(real_smiles_data)
    logP_gen, qed_gen, sa_gen = compute_molecular_properties(gen_valid)

    w1_logP = compute_w1(logP_real, logP_gen)
    print(f"W1 LogP is: {w1_logP}")
    mad_qed = compute_mad(qed_real, qed_gen)
    print(f"MAD QED is: {mad_qed}")
    mad_sa = compute_mad(sa_real, sa_gen)
    print(f"MAD SA is: {mad_sa}")

    gen_smiles_opt = pd.read_csv(gen_smiles_optimized)
    gen_smiles_opt_filter = gen_smiles_opt[gen_smiles_opt['target_smiles'].isin(gen_valid)]
    gen_smiles_opt_source = gen_smiles_opt_filter['source_smiles'].tolist()
    gen_smiles_opt_target = gen_smiles_opt_filter['target_smiles'].tolist()
    nll = nll_total_graph_edit_distance(gen_smiles_opt_source, gen_smiles_opt_target, alpha)
    print(f"NLL Score is: {nll}")

    fcd = FCD(device='cuda', n_jobs=8)
    fcd_score = fcd(real_smiles_data, gen_valid)
    print(f"FCD Score is: {fcd_score}")







