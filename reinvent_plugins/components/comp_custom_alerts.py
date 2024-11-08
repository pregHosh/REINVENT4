"""Compute scores with RDKit's QED"""

__all__ = ["CustomAlerts"]
from typing import List

import numpy as np
from pydantic.dataclasses import dataclass
from rdkit import Chem

from reinvent_plugins.mol_cache import molcache

from .add_tag import add_tag
from .component_results import ComponentResults


@add_tag("__parameters")
@dataclass
class Parameters:
    smarts: List[List[str]]
    filename: List[str]


@add_tag("__component", "filter")
class CustomAlerts:

    def __init__(self, params: Parameters):
        # FIXME: read from file?
        self.filename = params.filename[0]
        if self.filename != "a":
            with open(self.filename, "r") as f:
                self.smarts = [line.strip() for line in f]
        else:
            self.smarts = params.smarts[0]  # assume there is only one endpoint...

    @molcache
    def __call__(self, mols: List[Chem.Mol]) -> np.array:
        match = []

        for mol in mols:
            if not mol:
                score = False  # FIXME: rely on pre-processing?
            else:
                score = any(
                    [
                        mol.HasSubstructMatch(Chem.MolFromSmarts(subst))
                        for subst in self.smarts
                        if Chem.MolFromSmarts(subst)
                    ]
                )

            match.append(score)

        scores = [1 - m for m in match]

        return ComponentResults([np.array(scores, dtype=float)])
