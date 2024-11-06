__all__ = ["DiversityTanioto"]
import itertools
from typing import List

import numpy as np
from pydantic.dataclasses import dataclass
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity

from reinvent_plugins.mol_cache import molcache

from .add_tag import add_tag
from .component_results import ComponentResults


def calculate_similarity(smiles1, smiles2, radius):
    # Convert SMILES to RDKit molecule objects
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    # Generate Morgan fingerprints
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=2)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=2)

    # Calculate Tanimoto similarity
    similarity = TanimotoSimilarity(fp1, fp2)
    return similarity


@add_tag("__parameters")
@dataclass
class Parameters:
    radius: int = 3


@add_tag("__component")
class DiversityTanioto:
    def __init__(self, parameters: Parameters):
        self.radius = parameters.radius

    @molcache
    def __call__(self, mols: List[Chem.Mol]) -> np.array:

        smiles_all = list([Chem.MolToSmiles(mol) for mol in mols])

        smiles_pairs = []
        for i in range(len(smiles_all)):
            for j in range(i + 1, len(smiles_all)):
                smiles_pairs.append((smiles_all[i], smiles_all[j]))

        similarities = []
        for i, smiles_pair in enumerate(smiles_pairs):
            try:
                similarities.append(
                    calculate_similarity(smiles_pair[0], smiles_pair[1], self.radius)
                )
            except Exception as e:
                print(f"Error calculating similarity: {e}")
                pass
        similarities_avg = np.mean(similarities)

        scores = np.full(len(mols), similarities_avg)

        return ComponentResults([scores])
