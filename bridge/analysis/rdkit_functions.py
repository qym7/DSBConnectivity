import numpy as np
import torch
import re
import wandb
import os
import sys
from scipy.stats import wasserstein_distance
from sklearn.metrics import mean_absolute_error
from fcd_torch import FCD
import functools
from rdkit.DataStructs import TanimotoSimilarity
import pygmtools as pygm

pygm.set_backend("pytorch")
_ = torch.manual_seed(1)

try:
    from rdkit import Chem, DataStructs, RDLogger
    print("Found rdkit, all good")
    from rdkit.Chem import (
        RDConfig,
        QED,
        Crippen,
        AllChem,
        MolFromSmiles,
        MolToSmiles,
        Descriptors,
    )

    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer
except ModuleNotFoundError as e:
    use_rdkit = False
    from warnings import warn

    warn("Didn't find rdkit, this will fail")
    assert use_rdkit, "Didn't find rdkit"

RDLogger.DisableLog("rdApp.*")


allowed_bonds = {
    "H": 1,
    "C": 4,
    "N": 3,
    "O": 2,
    "F": 1,
    "B": 3,
    "Al": 3,
    "Si": 4,
    "P": [3, 5],
    "S": 4,
    "Cl": 1,
    "As": 3,
    "Br": 1,
    "I": 1,
    "Hg": [1, 2],
    "Bi": [3, 5],
    "Se": [2, 4, 6],
}
bond_dict = [
    None,
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
ATOM_VALENCY = {
    6: 4,
    7: 3,
    8: 2,
    9: 1,
    15: 3,
    16: 2,
    17: 1,
    35: 1,
    53: 1,
}


class BasicMolecularMetrics(object):
    def __init__(self, dataset_info, test_smiles=None, train_smiles=None):
        self.atom_decoder = dataset_info.atom_decoder
        self.dataset_info = dataset_info
        self.name = dataset_info.name
        # Retrieve dataset smiles only for qm9 currently.
        self.test_smiles = test_smiles
        self.dataset_smiles_list = train_smiles
        self.test_sa_avg, _, self.test_sa_success = self.compute_sascore(
            test_smiles
        )
        self.train_sa_avg, _, self.train_sa_success = self.compute_sascore(
            train_smiles
        )

    def compute_validity(self, generated):
        """generated: list of couples (positions, atom_types)"""
        valid = []
        num_components = []
        all_smiles = []
        for graph in generated:
            atom_types, edge_types = graph
            mol = build_molecule(
                atom_types,
                edge_types,
                self.dataset_info.atom_decoder,
            )
            smiles = mol2smiles(mol)
            try:
                mol_frags = Chem.rdmolops.GetMolFrags(
                    mol, asMols=True, sanitizeFrags=True
                )
                num_components.append(len(mol_frags))
            except:
                pass
            if smiles is not None:
                try:
                    mol_frags = Chem.rdmolops.GetMolFrags(
                        mol, asMols=True, sanitizeFrags=True
                    )
                    largest_mol = max(
                        mol_frags,
                        default=mol,
                        key=lambda m: m.GetNumAtoms(),
                    )
                    smiles = mol2smiles(largest_mol)
                    valid.append(smiles)
                    all_smiles.append(smiles)
                except Chem.rdchem.AtomValenceException:
                    print("Valence error in GetmolFrags")
                    all_smiles.append(None)
                except Chem.rdchem.KekulizeException:
                    print("Can't kekulize molecule")
                    all_smiles.append(None)
            else:
                all_smiles.append(None)

        return (
            valid,
            len(valid) / len(generated),
            np.array(num_components),
            all_smiles,
        )

    def compute_validity_smiles(self, generated):
        valid = []
        num_components = []
        all_smiles = []
        valid_indices = []  # To store indices of valid molecules

        for idx, smiles in enumerate(generated):
            mol = MolFromSmiles(smiles)
            smiles = mol2smiles(mol)
            try:
                mol_frags = Chem.rdmolops.GetMolFrags(
                    mol, asMols=True, sanitizeFrags=True
                )
                num_components.append(len(mol_frags))
            except:
                pass

            if smiles is not None:
                try:
                    mol_frags = Chem.rdmolops.GetMolFrags(
                        mol, asMols=True, sanitizeFrags=True
                    )
                    largest_mol = max(
                        mol_frags, default=mol, key=lambda m: m.GetNumAtoms()
                    )
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

        return (
            valid,
            len(valid) / len(generated),
            np.array(num_components),
            all_smiles,
            valid_indices,
        )

    def compute_uniqueness(self, valid):
        """valid: list of SMILES strings."""
        return list(set(valid)), len(set(valid)) / len(valid)

    def compute_novelty(self, unique):
        num_novel = 0
        num_coverage = 0
        novel = []
        coverage = []
        if self.dataset_smiles_list is None:
            print("Dataset smiles is None, novelty computation skipped")
            return 1, 1
        for smiles in unique:
            if smiles not in self.dataset_smiles_list:
                novel.append(smiles)
                num_novel += 1
        for smiles in unique:
            if smiles in self.test_smiles:
                coverage.append(smiles)
                num_coverage += 1
        return (
            novel,
            num_novel / len(unique),
            coverage,
            num_coverage / len(unique),
        )

    def compute_relaxed_validity(self, generated):
        valid_smiles = []
        all_smiles = []
        for graph in generated:
            atom_types, edge_types = graph
            mol = build_molecule_with_partial_charges(
                atom_types,
                edge_types,
                self.dataset_info.atom_decoder,
            )
            smiles = mol2smiles(mol)
            if smiles is not None:
                try:
                    mol_frags = Chem.rdmolops.GetMolFrags(
                        mol, asMols=True, sanitizeFrags=True
                    )
                    largest_mol = max(
                        mol_frags,
                        default=mol,
                        key=lambda m: m.GetNumAtoms(),
                    )
                    smiles = mol2smiles(largest_mol)
                    valid_smiles.append(smiles)
                    all_smiles.append(smiles)
                except Chem.rdchem.AtomValenceException:
                    all_smiles.append(None)
                    print("Valence error in GetmolFrags")
                except Chem.rdchem.KekulizeException:
                    all_smiles.append(None)
                    print("Can't kekulize molecule")
                except:
                    all_smiles.append(None)
            else:
                all_smiles.append(None)

        return (
            valid_smiles,
            all_smiles,
            len(valid_smiles) / len(generated),
        )

    def compute_relaxed_validity_smiles(self, generated):
        valid = []
        all_smiles = []
        for smiles in generated:
            mol = MolFromSmiles(smiles)
            smiles = mol2smiles(mol)
            if smiles is not None:
                try:
                    mol_frags = Chem.rdmolops.GetMolFrags(
                        mol, asMols=True, sanitizeFrags=True
                    )
                    largest_mol = max(
                        mol_frags, default=mol, key=lambda m: m.GetNumAtoms()
                    )
                    smiles = mol2smiles(largest_mol)
                    valid.append(smiles)
                    all_smiles.append(smiles)
                except Chem.rdchem.AtomValenceException:
                    print("Valence error in GetmolFrags")
                    all_smiles.append(None)
                except Chem.rdchem.KekulizeException:
                    print("Can't kekulize molecule")
                    all_smiles.append(None)
                except:
                    all_smiles.append(None)
            else:
                all_smiles.append(None)
        return (
            valid,
            all_smiles,
            len(valid) / len(generated),
        )

    def compute_sascore(self, all_smiles):
        count_true_sa = 0
        sa_values = []

        for smiles in all_smiles:
            try:
                mol = MolFromSmiles(smiles)
                sa_score = sascorer.calculateScore(mol)
            except:
                sa_score = 100

            sa_values.append(sa_score)

            if sa_score <= 3:
                count_true_sa += 1

        v_sa_values = [sa for sa in sa_values if sa != 100]
        if len(v_sa_values) > 0:
            v_sa_avg = sum(v_sa_values) / len(v_sa_values)
        else:
            v_sa_avg = 100

        return (
            v_sa_avg,
            sa_values,
            count_true_sa / len(all_smiles),
        )

    def smiles_similarity(self, smiles1, smiles2):
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)

        if mol1 is None or mol2 is None:
            raise ValueError("Invalid SMILES string(s)")

        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2)

        similarity = TanimotoSimilarity(fp1, fp2)

        return similarity


    def check_nll(self, generated, source, eps=1e-18):
        # get maximum number of nodes across all graphs in the batch
        max_nodes_0 = max(graph[0].size(0) for graph in generated)
        max_nodes_T = max(graph[0].size(0) for graph in source)

        X_0_padded = []
        E_0_padded = []
        X_T_padded = []
        E_T_padded = []
        node_masks = []

        # pad each graph to the maximum size in the batch
        for (X_0, E_0), (X_T, E_T) in zip(generated, source):
            n_nodes_0 = X_0.size(0)
            n_nodes_T = X_T.size(0)

            X_0_pad = torch.zeros(
                (max_nodes_0,), dtype=X_0.dtype, device=X_0.device
            )
            X_T_pad = torch.zeros(
                (max_nodes_T,), dtype=X_T.dtype, device=X_T.device
            )
            X_0_pad[:n_nodes_0] = X_0
            X_T_pad[:n_nodes_T] = X_T

            E_0_pad = torch.zeros(
                (max_nodes_0, max_nodes_0), dtype=E_0.dtype, device=E_0.device
            )
            E_T_pad = torch.zeros(
                (max_nodes_T, max_nodes_T), dtype=E_T.dtype, device=E_T.device
            )
            E_0_pad[:n_nodes_0, :n_nodes_0] = E_0
            E_T_pad[:n_nodes_T, :n_nodes_T] = E_T

            X_0_padded.append(X_0_pad)
            E_0_padded.append(E_0_pad)
            X_T_padded.append(X_T_pad)
            E_T_padded.append(E_T_pad)

            # create node masks
            node_masks.append(
                (
                    torch.arange(
                        max(max_nodes_0, max_nodes_T), device=X_0.device
                    )
                    < n_nodes_0
                ).float()
            )

        X_0_padded = torch.stack(X_0_padded)
        E_0_padded = torch.stack(E_0_padded)
        X_T_padded = torch.stack(X_T_padded)
        E_T_padded = torch.stack(E_T_padded)
        node_masks = torch.stack(node_masks)

        n0 = torch.tensor(
            [X.shape[0] for X, _ in generated], device=X_0_padded.device
        ).unsqueeze(1)
        nT = torch.tensor(
            [X.shape[0] for X, _ in source], device=X_T_padded.device
        ).unsqueeze(1)

        conn0, edge0, ne0 = pygm.utils.dense_to_sparse(E_0_padded)
        connT, edgeT, neT = pygm.utils.dense_to_sparse(E_T_padded)

        gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=0.1)

        K = pygm.utils.build_aff_mat(
            X_0_padded.unsqueeze(-1),
            edge0,
            conn0,
            X_T_padded.unsqueeze(-1),
            edgeT,
            connT,
            n0,
            ne0,
            nT,
            neT,
            node_aff_fn=gaussian_aff,
            edge_aff_fn=gaussian_aff,
        )

        Y = pygm.sm(K, n0, nT)
        Y = pygm.hungarian(Y)
        Y = Y.to(X_T_padded.dtype)

        # align the node and edge feature matrices using the matching matrices
        X_T_aligned = torch.matmul(Y, X_T_padded.unsqueeze(-1)).squeeze(
            -1
        )  # Align node values
        E_T_aligned = torch.matmul(
            torch.matmul(Y, E_T_padded), Y.transpose(1, 2)
        )

        nll_X = -torch.log(X_0_padded + eps) * X_T_aligned
        nll_E = -torch.log(E_0_padded + eps) * E_T_aligned

        # mask the padded entries
        nll_X = nll_X * node_masks  # Mask node values
        nll_E = (
            nll_E * node_masks.unsqueeze(-1) * node_masks.unsqueeze(-2)
        )  # Mask edge values

        nll_X = nll_X.sum(dim=-1)
        nll_E = nll_E.view(E_0_padded.size(0), -1).sum(dim=-1)
        nll_graph = nll_X + nll_E

        return nll_graph.mean().item(), nll_graph.tolist()

    def compute_w1(self, generated_values, source_values):
        return wasserstein_distance(generated_values, source_values)

    def compute_mad(self, generated_values, source_values):
        min_length = min(len(generated_values), len(source_values))
        property_gen = generated_values[:min_length]
        property_source = source_values[:min_length]
        return mean_absolute_error(property_gen, property_source), min_length

    def compute_properties(self, molecules):
        logP_values = []
        qed_values = []
        sa_scores = []

        for smiles in molecules:
            mol = Chem.MolFromSmiles(smiles)
            logP_values.append(Descriptors.MolLogP(mol))
            qed_values.append(QED.qed(mol))
            sa_scores.append(sascorer.calculateScore(mol))

        return np.array(logP_values), np.array(qed_values), np.array(sa_scores)

    def calculate_fcd(self, generated, source):
        fcd = FCD(device="cuda", n_jobs=8)
        fcd_score = fcd(generated, source)
        return fcd_score

    def evaluate(self, generated, source):
        """generated: list of pairs (atom_types: n [int], edge_types: n * n)"""

        print(f"Number of generated molecules: {len(generated)}")
        valid, validity, num_components, _ = self.compute_validity(generated)

        if source is not None:
            _, all_relaxed_smiles_source, _ = self.compute_relaxed_validity(
                source
            )
            v_sa_avg_source, sa_values_source, sa_v_source = (
                self.compute_sascore(all_relaxed_smiles_source)
            )
        else:
            all_relaxed_smiles_source = None
            v_sa_source = None
            sa_values_source = None

        nc_mu = num_components.mean() if len(num_components) > 0 else 0
        nc_min = num_components.min() if len(num_components) > 0 else 0
        nc_max = num_components.max() if len(num_components) > 0 else 0
        print(
            f"Validity over {len(generated)} molecules: {validity * 100 :.2f}%"
        )
        print(
            f"Number of connected components of {len(generated)} molecules: min:{nc_min:.2f} mean:{nc_mu:.2f} max:{nc_max:.2f}"
        )

        connectivity = (num_components == 1).sum() / len(num_components)
        print(
            f"Connectivity over {len(generated)} molecules: {connectivity * 100 :.2f}%"
        )

        relaxed_valid_smiles, all_smiles, relaxed_validity = (
            self.compute_relaxed_validity(generated)
        )
        print(
            f"Relaxed validity over {len(generated)} molecules: {relaxed_validity * 100 :.2f}%"
        )
        if relaxed_validity > 0:
            unique_smiles, uniqueness = self.compute_uniqueness(
                relaxed_valid_smiles
            )
            print(
                f"Uniqueness over {len(relaxed_valid_smiles)} valid molecules: {uniqueness * 100 :.2f}%"
            )

            if self.dataset_smiles_list is not None:
                novel_smiles, novelty, _, coverage = self.compute_novelty(
                    unique_smiles
                )
                print(
                    f"Novelty over {len(unique_smiles)} unique valid molecules: {novelty * 100 :.2f}%"
                )
                print(
                    f"Coverage over {len(unique_smiles)} unique valid molecules: {coverage * 100 :.2f}%"
                )
            else:
                novelty = -1.0

            v_sa_avg, sa_values, v_sa = self.compute_sascore(all_smiles)
            _, _, vu_sa = self.compute_sascore(unique_smiles)
            vu_sa = vu_sa * len(unique_smiles) / len(all_smiles)
            _, _, vun_sa = self.compute_sascore(novel_smiles)
            vun_sa = vun_sa * len(novel_smiles) / len(all_smiles)

            print(
                f"SA Average Value over {len(relaxed_valid_smiles)} unique valid molecules: {v_sa_avg :.2f}"
            )
            print(f"SA V.: {v_sa :.2f}")
            print(f"SA.V.U.: {vu_sa :.2f}")
            print(f"SA.V.U.N.: {vun_sa :.2f}")

        else:
            novelty = -1.0
            uniqueness = 0.0
            v_sa_avg = 0.0
            v_sa = 0.0
            vu_sa = 0.0
            vun_sa = 0.0
            coverage = 0.0
            unique_smiles = []
            sa_values = [100 for smile in all_smiles]

        return (
            [
                validity,
                relaxed_validity,
                uniqueness,
                novelty,
                connectivity,
                self.train_sa_avg,
                self.train_sa_success,
                self.test_sa_avg,
                self.test_sa_success,
                coverage,
                v_sa_avg,
                v_sa,
                vu_sa,
                vun_sa,
            ],
            unique_smiles,
            dict(nc_min=nc_min, nc_max=nc_max, nc_mu=nc_mu),
            (all_relaxed_smiles_source, all_smiles),
            (sa_values_source, sa_values),
        )

    def evaluate_zinc(self, generated, source):
        """generated: list of pairs (atom_types: n [int], edge_types: n * n)"""

        print(f"Number of generated molecules: {len(generated)}")
        valid, validity, num_components, all_smiles = self.compute_validity(
            generated
        )

        if source is not None:
            valid_source, _, _, all_smiles_source = self.compute_validity(
                source
            )
            nll, nll_vals = self.check_nll(generated, source)
            print(
                f"NLL Score over {len(generated)} total generated molecules is: {nll}"
            )
        else:
            valid_source = None
            all_smiles_source = None
            nll = -1.0

        nc_mu = num_components.mean() if len(num_components) > 0 else 0
        nc_min = num_components.min() if len(num_components) > 0 else 0
        nc_max = num_components.max() if len(num_components) > 0 else 0
        print(
            f"Validity over {len(generated)} molecules: {validity * 100 :.2f}%"
        )
        print(
            f"Number of connected components of {len(generated)} molecules: min:{nc_min:.2f} mean:{nc_mu:.2f} max:{nc_max:.2f}"
        )

        connectivity = (num_components == 1).sum() / len(num_components)
        print(
            f"Connectivity over {len(generated)} molecules: {connectivity * 100 :.2f}%"
        )

        relaxed_valid, all_smiles, relaxed_validity = (
            self.compute_relaxed_validity(generated)
        )
        print(
            f"Relaxed validity over {len(generated)} molecules: {relaxed_validity * 100 :.2f}%"
        )
        if relaxed_validity > 0:
            unique, uniqueness = self.compute_uniqueness(relaxed_valid)
            print(
                f"Uniqueness over {len(relaxed_valid)} valid molecules: {uniqueness * 100 :.2f}%"
            )

            if self.dataset_smiles_list is not None:
                _, novelty, _, coverage = self.compute_novelty(unique)
                print(
                    f"Novelty over {len(unique)} unique valid molecules: {novelty * 100 :.2f}%"
                )
                print(
                    f"Coverage over {len(unique)} unique valid molecules: {coverage * 100 :.2f}%"
                )
            else:
                novelty = -1.0
                coverage = -1.0

            if valid_source is not None:
                logP_gen, qed_gen, sa_gen = self.compute_properties(
                    relaxed_valid
                )
                logP_source, qed_source, sa_source = self.compute_properties(
                    valid_source
                )

                w1_logP = self.compute_w1(logP_gen, logP_source)
                print(
                    f"W1 LogP over {len(relaxed_valid)} valid molecules is: {w1_logP}"
                )
                mad_qed, qed_length = self.compute_mad(qed_gen, qed_source)
                print(
                    f"MAD QED over {qed_length} (a portion) valid molecules is: {mad_qed}"
                )
                mad_sa, sa_length = self.compute_mad(sa_gen, sa_source)
                print(
                    f"MAD SA {sa_length} (a portion) valid molecules is: {mad_sa}"
                )

                fcd_score = self.calculate_fcd(relaxed_valid, valid_source)
                print(
                    f"FCD Score over {len(relaxed_valid)} valid molecules is: {fcd_score}"
                )
            else:
                w1_logP = -1.0
                mad_qed = -1.0
                mad_sa = -1.0
                fcd_score = -1.0
        else:
            novelty = -1.0
            uniqueness = 0.0
            w1_logP = -1.0
            mad_qed = -1.0
            mad_sa = -1.0
            fcd_score = -1.0
            coverage = 0.0
            unique = []
        return (
            [
                validity,
                relaxed_validity,
                uniqueness,
                novelty,
                connectivity,
                w1_logP,
                mad_qed,
                mad_sa,
                fcd_score,
                coverage,
                nll,
            ],
            unique,
            dict(nc_min=nc_min, nc_max=nc_max, nc_mu=nc_mu),
            (all_smiles_source, all_smiles),
        )

    def evaluate_baselines(self, generated):
        """generated: list of pairs (atom_types: n [int], edge_types: n * n)"""

        print(f"Number of molecules to evaluate: {len(generated)}")
        valid, validity, num_components, all_smiles, valid_indices = (
            self.compute_validity_smiles(generated)
        )

        nc_mu = num_components.mean() if len(num_components) > 0 else 0
        nc_min = num_components.min() if len(num_components) > 0 else 0
        nc_max = num_components.max() if len(num_components) > 0 else 0
        print(
            f"Validity: {validity * 100 :.2f}%"
        )
        print(
            f"Number of connected components of {len(generated)} molecules: min:{nc_min:.2f} mean:{nc_mu:.2f} max:{nc_max:.2f}"
        )

        connectivity = (num_components == 1).sum() / len(num_components)
        print(
            f"Connectivity: {connectivity * 100 :.2f}%"
        )

        relaxed_valid, all_smiles, relaxed_validity = self.compute_relaxed_validity_smiles(
            generated
        )
        print(
            f"Relaxed validity: {relaxed_validity * 100 :.2f}%"
        )
        if relaxed_validity > 0:
            unique, uniqueness = self.compute_uniqueness(relaxed_valid)
            print(
                f"Uniqueness: {uniqueness * 100 :.2f}%"
            )
            vu = uniqueness * len(relaxed_valid) / len(all_smiles)
            print(
                f"VU: {vu * 100 :.2f}%"
            )

            if self.dataset_smiles_list is not None:
                novel, novelty, _, coverage = self.compute_novelty(unique)
                print(
                    f"Novelty: {novelty * 100 :.2f}%"
                )
                print(
                    f"Coverage: {coverage * 100 :.2f}%"
                )
                vun = novelty * len(unique) / len(all_smiles)
                print(
                    f"VUN: {vun * 100 :.2f}%"
                )

            sav_avg, _, sav_success = self.compute_sascore(relaxed_valid)
            savu_avg, _, savu_success = self.compute_sascore(unique)
            vu_sa = savu_success * len(unique) / len(all_smiles)
            savun_avg, _, savun_success = self.compute_sascore(novel)
            vun_sa = savun_success * len(novel) / len(all_smiles)

            print(
                f"SA_V Success Rate (<3): {sav_success * 100 :.2f}%"
            )
            print(
                f"SA_VU Success Rate (<3): {vu_sa * 100 :.2f}%"
            )
            print(
                f"SA_VUN Success Rate (<3): {vun_sa * 100 :.2f}%"
            )

            print(
                f"SA_V Average: {sav_avg :.2f}"
            )
            print(
                f"SA_VU Average: {savu_avg :.2f}"
            )
            print(
                f"SA_VUN Average: {savun_avg :.2f}"
            )

        return relaxed_valid, unique, novel, valid, valid_indices

def mol2smiles(mol):
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return None
    return Chem.MolToSmiles(mol)


def build_molecule(atom_types, edge_types, atom_decoder, verbose=False):
    if verbose:
        print("building new molecule")

    mol = Chem.RWMol()
    for atom in atom_types:
        a = Chem.Atom(atom_decoder[atom.item()])
        mol.AddAtom(a)
        if verbose:
            print(
                "Atom added: ",
                atom.item(),
                atom_decoder[atom.item()],
            )

    edge_types = torch.triu(edge_types)
    all_bonds = torch.nonzero(edge_types)
    for i, bond in enumerate(all_bonds):
        if bond[0].item() != bond[1].item():
            mol.AddBond(
                bond[0].item(),
                bond[1].item(),
                bond_dict[edge_types[bond[0], bond[1]].item()],
            )
            if verbose:
                print(
                    "bond added:",
                    bond[0].item(),
                    bond[1].item(),
                    edge_types[bond[0], bond[1]].item(),
                    bond_dict[edge_types[bond[0], bond[1]].item()],
                )
    return mol


def build_molecule_with_partial_charges(
    atom_types, edge_types, atom_decoder, verbose=False
):
    if verbose:
        print("\nbuilding new molecule")

    mol = Chem.RWMol()
    for atom in atom_types:
        a = Chem.Atom(atom_decoder[atom.item()])
        mol.AddAtom(a)
        if verbose:
            print(
                "Atom added: ",
                atom.item(),
                atom_decoder[atom.item()],
            )
    edge_types = torch.triu(edge_types)
    all_bonds = torch.nonzero(edge_types)

    for i, bond in enumerate(all_bonds):
        if bond[0].item() != bond[1].item():
            mol.AddBond(
                bond[0].item(),
                bond[1].item(),
                bond_dict[edge_types[bond[0], bond[1]].item()],
            )
            if verbose:
                print(
                    "bond added:",
                    bond[0].item(),
                    bond[1].item(),
                    edge_types[bond[0], bond[1]].item(),
                    bond_dict[edge_types[bond[0], bond[1]].item()],
                )
            # add formal charge to atom: e.g. [O+], [N+], [S+]
            # not support [O-], [N-], [S-], [NH+] etc.
            flag, atomid_valence = check_valency(mol)
            if verbose:
                print("flag, valence", flag, atomid_valence)
            if flag:
                continue
            else:
                assert len(atomid_valence) == 2
                idx = atomid_valence[0]
                v = atomid_valence[1]
                an = mol.GetAtomWithIdx(idx).GetAtomicNum()
                if verbose:
                    print(
                        "atomic num of atom with a large valence",
                        an,
                    )
                if an in (7, 8, 16) and (v - ATOM_VALENCY[an]) == 1:
                    mol.GetAtomWithIdx(idx).SetFormalCharge(1)
                    # print("Formal charge added")

    return mol


# Functions from GDSS
def check_valency(mol):
    try:
        Chem.SanitizeMol(
            mol,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES,
        )
        return True, None
    except ValueError as e:
        e = str(e)
        p = e.find("#")
        e_sub = e[p:]
        atomid_valence = list(map(int, re.findall(r"\d+", e_sub)))
        return False, atomid_valence


def correct_mol(m):
    # xsm = Chem.MolToSmiles(x, isomericSmiles=True)
    mol = m

    #####
    no_correct = False
    flag, _ = check_valency(mol)
    if flag:
        no_correct = True

    while True:
        flag, atomid_valence = check_valency(mol)
        if flag:
            break
        else:
            assert len(atomid_valence) == 2
            idx = atomid_valence[0]
            v = atomid_valence[1]
            queue = []
            check_idx = 0
            for b in mol.GetAtomWithIdx(idx).GetBonds():
                type = int(b.GetBondType())
                queue.append(
                    (
                        b.GetIdx(),
                        type,
                        b.GetBeginAtomIdx(),
                        b.GetEndAtomIdx(),
                    )
                )
                if type == 12:
                    check_idx += 1
            queue.sort(key=lambda tup: tup[1], reverse=True)

            if queue[-1][1] == 12:
                return None, no_correct
            elif len(queue) > 0:
                start = queue[check_idx][2]
                end = queue[check_idx][3]
                t = queue[check_idx][1] - 1
                mol.RemoveBond(start, end)
                if t >= 1:
                    mol.AddBond(start, end, bond_dict[t])
    return mol, no_correct


def valid_mol_can_with_seg(m, largest_connected_comp=True):
    if m is None:
        return None
    sm = Chem.MolToSmiles(m, isomericSmiles=True)
    if largest_connected_comp and "." in sm:
        vsm = [
            (s, len(s)) for s in sm.split(".")
        ]  # 'C.CC.CCc1ccc(N)cc1CCC=O'.split('.')
        vsm.sort(key=lambda tup: tup[1], reverse=True)
        mol = Chem.MolFromSmiles(vsm[0][0])
    else:
        mol = Chem.MolFromSmiles(sm)
    return mol


if __name__ == "__main__":
    smiles_mol = "C1CCC1"
    print("Smiles mol %s" % smiles_mol)
    chem_mol = Chem.MolFromSmiles(smiles_mol)
    block_mol = Chem.MolToMolBlock(chem_mol)
    print("Block mol:")
    print(block_mol)

use_rdkit = True


def check_stability(
    atom_types,
    edge_types,
    dataset_info,
    debug=False,
    atom_decoder=None,
):
    if atom_decoder is None:
        atom_decoder = dataset_info.atom_decoder

    n_bonds = np.zeros(len(atom_types), dtype="int")

    for i in range(len(atom_types)):
        for j in range(i + 1, len(atom_types)):
            n_bonds[i] += abs((edge_types[i, j] + edge_types[j, i]) / 2)
            n_bonds[j] += abs((edge_types[i, j] + edge_types[j, i]) / 2)
    n_stable_bonds = 0
    for atom_type, atom_n_bond in zip(atom_types, n_bonds):
        possible_bonds = allowed_bonds[atom_decoder[atom_type]]
        if type(possible_bonds) == int:
            is_stable = possible_bonds == atom_n_bond
        else:
            is_stable = atom_n_bond in possible_bonds
        if not is_stable and debug:
            print(
                "Invalid bonds for molecule %s with %d bonds"
                % (atom_decoder[atom_type], atom_n_bond)
            )
        n_stable_bonds += int(is_stable)

    molecule_stable = n_stable_bonds == len(atom_types)
    return molecule_stable, n_stable_bonds, len(atom_types)


def compute_molecular_metrics(
    molecule_list,
    test_smiles,
    train_smiles,
    dataset_info,
    fb,
    source_molecule_list,
):
    """molecule_list: (dict)"""

    if not dataset_info.remove_h:
        print(f"Analyzing molecule stability...")

        molecule_stable = 0
        nr_stable_bonds = 0
        n_atoms = 0
        n_molecules = len(molecule_list)

        for i, mol in enumerate(molecule_list):
            atom_types, edge_types = mol

            validity_results = check_stability(
                atom_types, edge_types, dataset_info
            )

            molecule_stable += int(validity_results[0])
            nr_stable_bonds += int(validity_results[1])
            n_atoms += int(validity_results[2])

        # Validity
        fraction_mol_stable = molecule_stable / float(n_molecules)
        fraction_atm_stable = nr_stable_bonds / float(n_atoms)
        validity_dict = {
            f"mol_metrics_charts_{fb}/mol_stable": fraction_mol_stable,
            f"mol_metrics_charts_{fb}/atm_stable": fraction_atm_stable,
        }
        if wandb.run:
            wandb.log(validity_dict)
    else:
        validity_dict = {
            f"mol_metrics_charts_{fb}/mol_stable": -1,
            f"mol_metrics_charts_{fb}/atm_stable": -1,
        }

    metrics = BasicMolecularMetrics(dataset_info, test_smiles, train_smiles)
    if dataset_info.name == "zinc":
        rdkit_metrics = metrics.evaluate_zinc(
            molecule_list, source_molecule_list
        )

        all_smiles = rdkit_metrics[-1]
        nc = rdkit_metrics[-2]
        dic = {
            f"mol_metrics_charts_{fb}/Validity": rdkit_metrics[0][0],
            f"mol_metrics_charts_{fb}/Relaxed Validity": rdkit_metrics[0][1],
            f"mol_metrics_charts_{fb}/Uniqueness": rdkit_metrics[0][2],
            f"mol_metrics_charts_{fb}/Novelty": rdkit_metrics[0][3],
            f"mol_metrics_charts_{fb}/Connectivity": rdkit_metrics[0][4],
            f"mol_metrics_charts_{fb}/W1 LogP": float(rdkit_metrics[0][5]),
            f"mol_metrics_charts_{fb}/MAD QED": float(rdkit_metrics[0][6]),
            f"mol_metrics_charts_{fb}/MAD SA": float(rdkit_metrics[0][7]),
            f"mol_metrics_charts_{fb}/FCD": float(rdkit_metrics[0][8]),
            f"mol_metrics_charts_{fb}/NLL GED": float(rdkit_metrics[0][10]),
            f"mol_metrics_charts_{fb}/coverage except training": float(
                rdkit_metrics[0][9]
            ),
            f"mol_metrics_charts_{fb}/nc_max": nc["nc_max"],
            f"mol_metrics_charts_{fb}/nc_mu": nc["nc_mu"],
        }
        if wandb.run:
            dic = dic
            wandb.log(dic)

        return validity_dict, rdkit_metrics, all_smiles, dic, (None, None)
    else:
        rdkit_metrics = metrics.evaluate(molecule_list, source_molecule_list)

    all_smiles = rdkit_metrics[-2]
    nc = rdkit_metrics[-3]
    all_sa_values_source = rdkit_metrics[-1][0]
    all_sa_values = rdkit_metrics[-1][1]
    dic = {
        f"mol_metrics_charts_{fb}/Validity": rdkit_metrics[0][0],
        f"mol_metrics_charts_{fb}/Relaxed Validity": rdkit_metrics[0][1],
        f"mol_metrics_charts_{fb}/Uniqueness": rdkit_metrics[0][2],
        f"mol_metrics_charts_{fb}/Novelty": rdkit_metrics[0][3],
        f"mol_metrics_charts_{fb}/Connectivity": rdkit_metrics[0][4],
        f"mol_metrics_charts_{fb}/train SA Average Value": float(
            rdkit_metrics[0][5]
        ),
        f"mol_metrics_charts_{fb}/train SA Success": float(rdkit_metrics[0][6]),
        f"mol_metrics_charts_{fb}/test SA Average Value": float(
            rdkit_metrics[0][7]
        ),
        f"mol_metrics_charts_{fb}/test SA Success": float(rdkit_metrics[0][8]),
        f"mol_metrics_charts_{fb}/coverage except training": float(
            rdkit_metrics[0][9]
        ),
        f"mol_metrics_charts_{fb}/SA V. Average": float(rdkit_metrics[0][10]),
        f"mol_metrics_charts_{fb}/SA V.": float(rdkit_metrics[0][11]),
        f"mol_metrics_charts_{fb}/SA.V.U.": float(rdkit_metrics[0][12]),
        f"mol_metrics_charts_{fb}/SA.V.U.N.": float(rdkit_metrics[0][13]),
        f"mol_metrics_charts_{fb}/V.": rdkit_metrics[0][1],
        f"mol_metrics_charts_{fb}/V.U.": rdkit_metrics[0][1]
        * rdkit_metrics[0][2],
        f"mol_metrics_charts_{fb}/V.U.N.": rdkit_metrics[0][1]
        * rdkit_metrics[0][2]
        * rdkit_metrics[0][3],
        f"mol_metrics_charts_{fb}/nc_max": nc["nc_max"],
        f"mol_metrics_charts_{fb}/nc_mu": nc["nc_mu"],
    }
    if wandb.run:
        dic = dic
        wandb.log(dic)

    return (
        validity_dict,
        rdkit_metrics,
        all_smiles,
        dic,
        (all_sa_values_source, all_sa_values),
    )
