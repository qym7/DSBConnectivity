import numpy as np
import torch
import re
import wandb
import os
import sys

try:
    from rdkit import Chem, DataStructs, RDLogger
    print("Found rdkit, all good")
    from rdkit.Chem import RDConfig, QED, Crippen, AllChem, MolFromSmiles, MolToSmiles
    sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
    import sascorer
except ModuleNotFoundError as e:
    use_rdkit = False
    from warnings import warn
    warn("Didn't find rdkit, this will fail")
    assert use_rdkit, "Didn't find rdkit"

RDLogger.DisableLog('rdApp.*')


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
ATOM_VALENCY = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}


class BasicMolecularMetrics(object):
    def __init__(self, dataset_info, test_smiles=None, train_smiles=None):
        self.atom_decoder = dataset_info.atom_decoder
        self.dataset_info = dataset_info
        # Retrieve dataset smiles only for qm9 currently.
        self.test_smiles = test_smiles
        self.dataset_smiles_list = train_smiles
        _, self.test_sa_avg, self.test_sa_success = self.compute_sascore(test_smiles)
        _, self.train_sa_avg, self.train_sa_success = self.compute_sascore(train_smiles)

    def compute_validity(self, generated):
        """generated: list of couples (positions, atom_types)"""
        valid = []
        num_components = []
        all_smiles = []
        for graph in generated:
            atom_types, edge_types = graph
            mol = build_molecule(atom_types, edge_types, self.dataset_info.atom_decoder)
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
                except Chem.rdchem.AtomValenceException:
                    print("Valence error in GetmolFrags")
                    all_smiles.append(None)
                except Chem.rdchem.KekulizeException:
                    print("Can't kekulize molecule")
                    all_smiles.append(None)
            else:
                all_smiles.append(None)

        return valid, len(valid) / len(generated), np.array(num_components), all_smiles

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
        return novel, num_novel / len(unique), coverage, num_coverage / len(unique)

    def compute_relaxed_validity(self, generated):
        valid = []
        for graph in generated:
            atom_types, edge_types = graph
            mol = build_molecule_with_partial_charges(
                atom_types, edge_types, self.dataset_info.atom_decoder
            )
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
                except Chem.rdchem.AtomValenceException:
                    print("Valence error in GetmolFrags")
                except Chem.rdchem.KekulizeException:
                    print("Can't kekulize molecule")
        return valid, len(valid) / len(generated)

    def compute_sascore(self, all_smiles):
        count_true_sa = 0
        sa_values = []
        for smiles in all_smiles:
            mol = MolFromSmiles(smiles)
            sa_score = sascorer.calculateScore(mol)
            sa_values.append(sa_score)
            if sa_score <= 3:
                count_true_sa += 1

        sa_avg = sum(sa_values) / float(len(sa_values))
        return sa_values, sa_avg, count_true_sa / len(all_smiles)

    def evaluate(self, generated):
        """generated: list of pairs (positions: n x 3, atom_types: n [int])
        the positions and atom types should already be masked."""
        
        print(f"Number of generated molecules: {len(generated)}")
        valid, validity, num_components, all_smiles = self.compute_validity(generated)
        nc_mu = num_components.mean() if len(num_components) > 0 else 0
        nc_min = num_components.min() if len(num_components) > 0 else 0
        nc_max = num_components.max() if len(num_components) > 0 else 0
        print(f"Validity over {len(generated)} molecules: {validity * 100 :.2f}%")
        print(
            f"Number of connected components of {len(generated)} molecules: min:{nc_min:.2f} mean:{nc_mu:.2f} max:{nc_max:.2f}"
        )

        connectivity = (num_components == 1).sum()/len(num_components)
        print(f"Connectivity over {len(generated)} molecules: {connectivity * 100 :.2f}%")

        relaxed_valid, relaxed_validity = self.compute_relaxed_validity(generated)
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

            # print((num_components==1).sum())
            existing_smiles = [smiles for smiles in all_smiles if smiles is not None]
            # print(len(existing_smiles))
            # unique = all_smiles[num_components == 1]
            # unique = list(set(unique_connected))
            _, sa_avg, sa_success = self.compute_sascore(unique)
            _, _, vun_sa = self.compute_sascore(list(set(unique)))
            vun_sa = vun_sa * len(list(set(unique))) / len(generated)
            
            print(f"SA Score Success Rate (<3) over {len(unique)} unique valid molecules: {sa_success * 100 :.2f}%")
            print(f"SA Average Value over {len(unique)} unique valid molecules: {sa_avg :.2f}")
            print(f"SA.V.U.N. over {len(unique)} unique valid molecules: {vun_sa :.2f}")

        else:
            novelty = -1.0
            uniqueness = 0.0
            sa_success = 0.0
            sa_avg = 0.0
            vun_sa = 0.0
            coverage = 0.0
            unique = []
        return (
            [validity, relaxed_validity, uniqueness, novelty, connectivity, sa_success, sa_avg, self.train_sa_avg, self.train_sa_success, self.test_sa_avg, self.test_sa_success, coverage, vun_sa],
            unique,
            dict(nc_min=nc_min, nc_max=nc_max, nc_mu=nc_mu),
            all_smiles,
        )


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
            print("Atom added: ", atom.item(), atom_decoder[atom.item()])

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
            print("Atom added: ", atom.item(), atom_decoder[atom.item()])
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
                    print("atomic num of atom with a large valence", an)
                if an in (7, 8, 16) and (v - ATOM_VALENCY[an]) == 1:
                    mol.GetAtomWithIdx(idx).SetFormalCharge(1)
                    # print("Formal charge added")

    return mol


# Functions from GDSS
def check_valency(mol):
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
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
                queue.append((b.GetIdx(), type, b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
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
    atom_types, edge_types, dataset_info, debug=False, atom_decoder=None
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


def compute_molecular_metrics(molecule_list, test_smiles, train_smiles, dataset_info, fb):
    """molecule_list: (dict)"""

    if not dataset_info.remove_h:
        print(f"Analyzing molecule stability...")

        molecule_stable = 0
        nr_stable_bonds = 0
        n_atoms = 0
        n_molecules = len(molecule_list)

        for i, mol in enumerate(molecule_list):
            atom_types, edge_types = mol

            validity_results = check_stability(atom_types, edge_types, dataset_info)

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
        validity_dict = {f"mol_metrics_charts_{fb}/mol_stable": -1, f"mol_metrics_charts_{fb}/atm_stable": -1}

    metrics = BasicMolecularMetrics(dataset_info, test_smiles, train_smiles)
    rdkit_metrics = metrics.evaluate(molecule_list)
    all_smiles = rdkit_metrics[-1]

    nc = rdkit_metrics[-2]
    dic = {
        f"mol_metrics_charts_{fb}/Validity": rdkit_metrics[0][0],
        f"mol_metrics_charts_{fb}/Relaxed Validity": rdkit_metrics[0][1],
        f"mol_metrics_charts_{fb}/Uniqueness": rdkit_metrics[0][2],
        f"mol_metrics_charts_{fb}/Novelty": rdkit_metrics[0][3],
        f"mol_metrics_charts_{fb}/Connectivity": rdkit_metrics[0][4],
        f"mol_metrics_charts_{fb}/SA Score Success Rate (<3)": float(rdkit_metrics[0][5]),
        f"mol_metrics_charts_{fb}/SA Average Value": float(rdkit_metrics[0][6]),
        f"mol_metrics_charts_{fb}/train_SA Average Value": float(rdkit_metrics[0][7]),
        f"mol_metrics_charts_{fb}/train_SA Score Success Rate (<3)": float(rdkit_metrics[0][8]),
        f"mol_metrics_charts_{fb}/test_SA Average Value": float(rdkit_metrics[0][9]),
        f"mol_metrics_charts_{fb}/test_SA Score Success Rate (<3)": float(rdkit_metrics[0][10]),
        f"mol_metrics_charts_{fb}/coverage except training": float(rdkit_metrics[0][11]),
        f"mol_metrics_charts_{fb}/SA.V.U.N.": float(rdkit_metrics[0][-1]),
        f"mol_metrics_charts_{fb}/SA Average Value": float(rdkit_metrics[0][6]),
        f"mol_metrics_charts_{fb}/nc_max": nc["nc_max"],
        f"mol_metrics_charts_{fb}/nc_mu": nc["nc_mu"],
    }
    if wandb.run:
        dic = dic
        wandb.log(dic)

    return validity_dict, rdkit_metrics, all_smiles, dic
