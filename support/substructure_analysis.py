import numpy as np

i
from model_analysis import calculate_similarity, to_scaffold

ref = "data/Data_FORMED.smi"
sampled = "data/Data_FORMED_sampled.smi"
similarity_measurement = "tanimoto"
verb = 0

scaffold_ref = []
with open(ref, "r") as file:
    for line in file:
        scaffold_ref.append(line.strip())

smiles_sampled = []
with open(sampled, "r") as file:
    for line in file:
        smiles_sampled.append(line.strip())

match_scaffolds = np.zeros((len(smiles_sampled)))
dmaxs = np.zeros((len(smiles_sampled)))
for i, smi in enumerate(smiles_sampled):
    d = np.zeros((len(scaffold_ref)))
    scaffold = to_scaffold(smi)
    for j, scaffold_r in enumerate(scaffold_ref):
        similarity = calculate_similarity(scaffold, scaffold_r)
        d[j] = similarity
        if verb > 0:
            print(f"Sampled: {i}, Reference: {j}, Similarity: {similarity}")
    scaffold_max = scaffold_ref[np.argmax(d)]
    dmax = np.max(d)
    match_scaffolds[i] = scaffold_max
    dmaxs[i] = dmax

np.savetxt("match_scaffolds.txt", match_scaffolds, fmt="%s")
np.savetxt("dmaxs.txt", dmaxs, fmt="%s")
