"""Compute various scores with RDKit"""

from typing import Callable, List

import numpy as np
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski

from reinvent_plugins.mol_cache import molcache

from ..component_results import ComponentResults


def compute_scores(mols: List[Chem.Mol], func: Callable) -> np.array:
    """Compute scores using a RDKit function

    :param mols: a list of RDKit molecules
    :param func: a callable that computes the score for a RDKit molecule
    :returns: a numpy array with the scores
    """

    scores = []

    for mol in mols:
        try:
            score = func(mol)
        except ValueError:
            score = np.nan

        scores.append(score)

    return ComponentResults([np.array(scores, dtype=float)])


def num_sp(mol: Chem.Mol) -> int:
    num_sp_atoms = len(
        [atom for atom in mol.GetAtoms() if atom.GetHybridization() == Chem.HybridizationType.SP]
    )

    return num_sp_atoms


def num_sp2(mol: Chem.Mol) -> int:
    num_sp2_atoms = len(
        [atom for atom in mol.GetAtoms() if atom.GetHybridization() == Chem.HybridizationType.SP2]
    )

    return num_sp2_atoms


def num_sp3(mol: Chem.Mol) -> int:
    num_sp3_atoms = len(
        [atom for atom in mol.GetAtoms() if atom.GetHybridization() == Chem.HybridizationType.SP3]
    )
    return num_sp3_atoms


def graph_length(mol: Chem.Mol) -> int:
    return int(np.max(Chem.GetDistanceMatrix(mol)))


def formal_charge(mol: Chem.Mol) -> int:
    return sum([atom.GetFormalCharge() for atom in mol.GetAtoms()])


def force_neutral(mol: Chem.Mol) -> int:

    formal_c = formal_charge(mol)
    if formal_c == 0:
        return 1
    else:
        return 0


def count_aromatic_rings(mol):
    # Get the ring information
    ring_info = mol.GetRingInfo()

    # Get atom indices of all rings
    rings = ring_info.AtomRings()

    # Count the number of aromatic rings
    aromatic_rings = 0
    for ring in rings:
        # Check if all atoms in the ring are aromatic
        if all(mol.GetAtomWithIdx(atom_idx).GetIsAromatic() for atom_idx in ring):
            aromatic_rings += 1

    return aromatic_rings


def enforce_aromaticity(mol: Chem.Mol) -> Chem.Mol:

    n_ring = count_aromatic_rings(mol)
    if n_ring > 1:
        n_ring = 1
    return n_ring


cls_func_map = {
    "ForceAromaticity": enforce_aromaticity,
    "FormalCharge": formal_charge,
    "ForceNeutral": force_neutral,
    "Qed": Descriptors.qed,
    "MolecularWeight": Descriptors.MolWt,
    "GraphLength": graph_length,
    "NumAtomStereoCenters": Chem.CalcNumAtomStereoCenters,
    "HBondAcceptors": Lipinski.NumHAcceptors,
    "HBondDonors": Lipinski.NumHDonors,
    "NumRotBond": Lipinski.NumRotatableBonds,
    "Csp3": Lipinski.FractionCSP3,
    "numsp": num_sp,
    "numsp2": num_sp2,
    "numsp3": num_sp3,
    "NumHeavyAtoms": Lipinski.HeavyAtomCount,
    "NumHeteroAtoms": Lipinski.NumHeteroatoms,
    "NumRings": Lipinski.RingCount,
    "NumAromaticRings": Lipinski.NumAromaticRings,
    "NumAliphaticRings": Lipinski.NumAliphaticRings,
    "SlogP": Crippen.MolLogP,
}

for cls_name, func in cls_func_map.items():

    class Temp:
        def __init__(self, *args, **kwargs):
            pass

        @molcache
        def __call__(self, mols: List[Chem.Mol]) -> np.array:
            return compute_scores(mols, self.desc_function.__func__)

    Temp.__name__ = cls_name
    Temp.__qualname__ = cls_name

    globals()[cls_name] = Temp
    del Temp

    cls = globals()[cls_name]
    setattr(cls, "__component", True)
    setattr(cls, "desc_function", func)
