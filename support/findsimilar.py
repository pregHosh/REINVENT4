# %%
import argparse

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.DataStructs import TanimotoSimilarity
from tqdm import tqdm


def to_scaffold(smiles, chirality=True):
    """
    Return a scaffold SMILES string of this molecule.

    Parameters:
        chirality (bool, optional): consider chirality in the scaffold or not

    Returns:
        str
    """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(smiles, includeChirality=chirality)
    return scaffold


def calculate_similarity(smiles1, smiles2):
    # Convert SMILES to RDKit molecule objects
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    # Generate Morgan fingerprints
    # fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=3)
    # fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=3)
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 3, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 3, nBits=2048)
    # Calculate Tanimoto similarity
    similarity = TanimotoSimilarity(fp1, fp2)
    return similarity


def get_most_similar(smiles, ref_smiles, threhold=0.85):
    similarity = 0
    most_similar = None
    for ref in ref_smiles:
        sim = calculate_similarity(smiles, ref)
        if sim > similarity:
            similarity = sim
            most_similar = ref
        if similarity > threhold:
            break
    return most_similar, similarity


def get_scaffold(smiles_list, threshold=0.69):
    scaffolds = []
    duplicate = False
    for smiles in tqdm(smiles_list, desc="Finding scaffolds", total=len(smiles_list)):
        scaffold = to_scaffold(smiles)
        if scaffold not in scaffolds:
            for id, i in enumerate(scaffolds):
                try:
                    similarity = calculate_similarity(i, scaffold)
                    if similarity > threshold:
                        print(f"Found similar scaffold: {scaffold} with {scaffolds[id]}")
                        duplicate = True
                        break
                except Exception:
                    duplicate = False
            if not duplicate:
                scaffolds.append(scaffold)
    return scaffolds


def save_smiles_to_smi(smiles_list, filename):
    with open(filename, "w") as f:
        for smiles in smiles_list:
            f.write(smiles + "\n")


# %%


def main():
    parser = argparse.ArgumentParser(
        description="Find the most similar molecules in the dataset to the sampled molecules."
    )
    parser.add_argument(
        "--filename",
        dest="filename",
        type=str,
        default=None,
        help="Path to the .csv file containing the sampled molecules (default=None).",
    )
    parser.add_argument(
        "--ref",
        dest="ref",
        type=str,
        default=None,
        help="Path to the reference .smi file (default=None).",
    )
    # ref = "../data/Data_FORMED.smi"
    # filename = "../cl_exp/nav2_2/sampling.csv"
    args = parser.parse_args()
    filename = args.filename
    ref = args.ref

    smiles_ref = []
    with open(ref, "r") as file:
        for line in file:
            smiles_ref.append(line.strip())
    smiles_sampled = pd.read_csv(filename)["SMILES"].tolist()

    most_similars = []
    scores = []
    for smiles in tqdm(smiles_sampled, desc="Finding most similar", total=len(smiles_sampled)):
        most_similar, similarity = get_most_similar(smiles, smiles_ref, threhold=0.87)
        most_similars.append(most_similar)
        scores.append(similarity)
        print(f"Most similar: {most_similar}, Similarity: {similarity} for {smiles}")

    dict = {"SMILES": smiles_sampled, "Most similar": most_similars, "Similarity": scores}
    df = pd.DataFrame(dict)
    df.to_csv("sampling_similarity.csv", index=False)

    df_sorted = df.sort_values(by="Similarity", ascending=False)
    print(df_sorted.head(10))


if __name__ == "__main__":
    main()
