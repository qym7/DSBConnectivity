from ..analysis.rdkit_functions import BasicMolecularMetrics
import pandas as pd
from types import SimpleNamespace

if __name__ == "__main__":
    # input from user
    dataset_name = "qm9"
    remove_h = True
    file_path = (
        "/scratch/uceeosm/DST/result/test_qm9_1k.csv"
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

    dataset_info = SimpleNamespace(atom_decoder=atom_decoder, name=dataset_name)

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
    # graph_edit_distances = []

    for original, optimized in zip(
        filtered_original_molecules, valid_optimized
    ):
        similarity_score = metrics.smiles_similarity(original, optimized)
        change_score = 1 - similarity_score
        similarity.append(change_score)

    similarity_mean = sum(similarity) / len(similarity)

    print(f"Average Tanimoto Distance Similarity is: ", {similarity_mean})

    fcd_score = metrics.calculate_fcd(optimized_smiles, original_smiles)
    print(f"FCD score is: ", {fcd_score})


