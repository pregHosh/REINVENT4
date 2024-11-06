from model_analysis import calculate_similarity
from tqdm import tqdm

ref = "../data/Data_FORMED.smi"
output = "../data/Data_FORMED_unique.smi"
unique_threshold = 0.35

smiles_ref = []
with open(ref, "r") as file:
    for line in file:
        smiles_ref.append(line.strip())

smiles_unique = []
for i, smiles in tqdm(enumerate(smiles_ref), total=len(smiles_ref)):
    if i == 0:
        smiles_unique.append(smiles)
    else:
        for j, unique in enumerate(smiles_unique):
            sim = calculate_similarity(smiles, unique)
            if sim > unique_threshold:
                print(f"Similarity between {smiles} and {unique}: {sim}")
                break
            else:
                if j == len(smiles_unique) - 1:
                    smiles_unique.append(smiles)
                    break

with open(output, "w") as file:
    for smiles in smiles_unique:
        file.write(smiles + "\n")
