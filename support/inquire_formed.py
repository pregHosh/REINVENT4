import argparse

import pandas as pd
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols


def inquire_from_csv(csv_file, smiles_input):
    df = pd.read_csv(csv_file)
    required_columns = ["smiles", "name", "S1_exc", "T1_exc", "S1_ehdist"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"The required column '{col}' is not in the CSV file.")

    result = df[df["smiles"] == smiles_input]
    if result.empty:
        print("No match found for the given SMILES.")
    else:
        for index, row in result.iterrows():
            print(f"\nMatch found:")
            print(f"  Filename     : {row['name']}")
            print(f"  S1_exc       : {row['S1_exc']}")
            print(f"  T1_exc       : {row['T1_exc']}")
            print(f"  S1_ehdistc   : {row['S1_ehdist']}")


def find_most_similar_smiles(csv_file, smiles_input):
    df = pd.read_csv(csv_file)
    required_columns = ["smiles", "name", "S1_exc", "T1_exc", "S1_ehdist"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"The required column '{col}' is not in the CSV file.")

    input_mol = Chem.MolFromSmiles(smiles_input)
    if input_mol is None:
        raise ValueError("Invalid SMILES string provided.")

    input_fp = FingerprintMols.FingerprintMol(input_mol)
    similarities = []

    for index, row in df.iterrows():
        if not isinstance(row["smiles"], str) or not row["smiles"]:
            continue
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol is not None:
            fp = FingerprintMols.FingerprintMol(mol)
            similarity = DataStructs.FingerprintSimilarity(input_fp, fp)
            similarities.append((index, similarity))

    if not similarities:
        print("No valid SMILES found in the database.")
        return

    similarities.sort(key=lambda x: x[1], reverse=True)
    top_similarities = similarities[:100]
    top_similarities.to_csv("top_similarities.csv")

    print("\nTop 10 similar matches found:")
    for index, similarity in top_similarities:
        row = df.loc[index]
        print(f"\nSimilarity    : {similarity:.4f}")
        print(f"  Filename     : {row['name']}")
        print(f"  S1_exc       : {row['S1_exc']}")
        print(f"  T1_exc       : {row['T1_exc']}")
        print(f"  S1_ehdistc   : {row['S1_ehdist']}")
        print(f"  SMILES       : {row['smiles']}")
    # most_similar_index = max(similarities, key=lambda x: x[1])[0]
    # most_similar_row = df.loc[most_similar_index]
    # print(f"\nMost similar match found:")
    # print(f"  Filename     : {most_similar_row['name']}")
    # print(f"  S1_exc       : {most_similar_row['S1_exc']}")
    # print(f"  T1_exc       : {most_similar_row['T1_exc']}")
    # print(f"  S1_ehdistc   : {most_similar_row['S1_ehdist']}")
    # print(f"  SMILES       : {most_similar_row['smiles']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inquire from CSV using SMILES input.")
    parser.add_argument(
        "--csv_filename",
        default="data/Data_FORMED_scored.csv",
        type=str,
        help="Path to the CSV file.",
    )
    parser.add_argument(
        "--find_similar", action="store_true", help="Find the most similar SMILES in the database."
    )
    parser.add_argument("--input_smiles", type=str, help="SMILES string to inquire.")
    args = parser.parse_args()

    if args.find_similar:
        find_most_similar_smiles(args.csv_filename, args.input_smiles)
    else:
        inquire_from_csv(args.csv_filename, args.input_smiles)
