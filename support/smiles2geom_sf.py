import argparse
import logging
import multiprocessing
import os
import re
import shutil
import subprocess as sp
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolfiles import MolToSmiles as mol2smi
from tqdm import tqdm

logger = logging.getLogger(__name__)
lg = RDLogger.logger()
lg.setLevel(RDLogger.ERROR)
RDLogger.DisableLog("rdApp.*")

n_cores_ = str(multiprocessing.cpu_count())
os.environ["OMP_NUM_THREADS"] = n_cores_
os.environ["OPENBLAS_NUM_THREADS"] = n_cores_
os.environ["MKL_NUM_THREADS"] = n_cores_
os.environ["VECLIB_MAXIMUM_THREADS"] = n_cores_
os.environ["NUMEXPR_NUM_THREADS"] = n_cores_


def get_confs_ff(mol, maxiters=250):
    mol_copy = Chem.Mol(mol)
    mol_structure = Chem.Mol(mol)
    mol_structure.RemoveAllConformers()
    try:
        if Chem.rdForceFieldHelpers.MMFFHasAllMoleculeParams(mol):
            AllChem.MMFFSanitizeMolecule(mol)
            energies = AllChem.MMFFOptimizeMoleculeConfs(
                mol, maxIters=maxiters, nonBondedThresh=15.0
            )
            energies_list = [e[1] for e in energies]
            min_e_index = energies_list.index(min(energies_list))
            mol_structure.AddConformer(mol.GetConformer(min_e_index))
            return mol_structure
        else:
            logger.debug("Could not do complete MMFF typing. SMILES {0}".format(mol2smi(mol)))
    except ValueError:
        logger.debug("Conformational sampling led to crash. SMILES {0}".format(mol2smi(mol)))
        mol = mol_copy
    try:
        if Chem.rdForceFieldHelpers.UFFHasAllMoleculeParams(mol):
            energies = AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=maxiters, vdwThresh=15.0)
            energies_list = [e[1] for e in energies]
            min_e_index = energies_list.index(min(energies_list))
            mol_structure.AddConformer(mol.GetConformer(min_e_index))
            return mol_structure
        else:
            logger.debug("Could not do complete UFF typing. SMILES {0}".format(mol2smi(mol)))
    except ValueError:
        logger.debug("Conformational sampling led to crash. SMILES {0}".format(mol2smi(mol)))
        mol = mol_copy
    logger.debug("Conformational sampling not performed. SMILES {0}".format(mol2smi(mol)))
    return mol_copy


def get_structure_ff(mol, n_confs=5):
    """Generates a reasonable set of 3D structures
    using forcefields for a given rdkit.mol object.
    It will try several 3D generation approaches in rdkit.
    It will try to sample several conformations and get the minima.

    Parameters:
    :param mol: an rdkit mol object
    :type mol: rdkit.mol
    :param n_confs: number of conformations to sample
    :type n_confs: int

    Returns:
    :return mol_structure: mol with 3D coordinate information set
    """
    Chem.SanitizeMol(mol)
    mol = Chem.AddHs(mol)
    coordinates_added = False
    if not coordinates_added:
        try:
            conformer_ids = AllChem.EmbedMultipleConfs(
                mol,
                numConfs=n_confs,
                useExpTorsionAnglePrefs=True,
                useBasicKnowledge=True,
                pruneRmsThresh=1.25,
                enforceChirality=True,
            )
        except:
            logger.debug("Method 1 failed to generate conformations.")
        else:
            if all([conformer_id >= 0 for conformer_id in conformer_ids]):
                coordinates_added = True

    if not coordinates_added:
        try:
            params = params = AllChem.srETKDGv3()
            params.useSmallRingTorsions = True
            conformer_ids = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
        except:
            logger.debug("Method 2 failed to generate conformations.")
        else:
            if all([conformer_id >= 0 for conformer_id in conformer_ids]):
                coordinates_added = True

    if not coordinates_added:
        try:
            conformer_ids = AllChem.EmbedMultipleConfs(
                mol,
                numConfs=n_confs,
                useRandomCoords=True,
                useBasicKnowledge=True,
                maxAttempts=250,
                pruneRmsThresh=1.25,
                ignoreSmoothingFailures=True,
            )
        except:
            logger.debug("Method 3 failed to generate conformations.")
        else:
            if all([conformer_id >= 0 for conformer_id in conformer_ids]):
                coordinates_added = True
        finally:
            if not coordinates_added:
                diagnose_mol(mol)

    if not coordinates_added:
        logger.exception("Could not embed the molecule. SMILES {0}".format(mol2smi(mol)))
        return mol
    else:
        mol_structure = get_confs_ff(mol, maxiters=250)
        return mol_structure


def cleanup_mol(mol: Chem.Mol, n_confs: int = 20) -> Chem.Mol:
    """
    Clean up a molecule with potentially bad conformer information.

    Parameters:
    -----------
    mol : Chem.Mol
        The molecule with potentially bad conformer information.

    n_confs : int, optional
        The desired number of conformers to generate (default is 20).

    Returns:
    --------
    Chem.Mol
        A molecule object with cleaned conformer information.
    """

    try:
        # Attempt to generate conformers using 'get_structure_ff'
        mol = get_structure_ff(mol, n_confs)

    except Exception as e:
        print(e)
        # If 'get_structure_ff' fails, generate a single random conformer
        conf_id = AllChem.EmbedMultipleConfs(
            mol,
            numConfs=1,
            useRandomCoords=True,
            numThreads=multiprocessing.cpu_count(),
        )

    return mol


def xtb_opt(xyz, charge=0, unpaired_e=0, level=1):
    """
    Perform quick and reliable geometry optimization with xtb module
    Parameters:
    1. xyz: xyz file
    2. charge: (int)
    3. unpaired_e: number of unpaired electrons
    4. level: 0-2; Halmitonian level [gfn-1,2, gfnff], default=2

    Returns:
    none
    (filename_xtbopt.xyz)
    """
    execution = []
    if level == 0:
        execution = ["xtb", "--gfnff", xyz, "--opt"]
    elif level == 1:
        execution = [
            "xtb",
            "--gfn",
            "1",
            xyz,
            "--opt",
        ]
    elif level == 2:
        execution = [
            "xtb",
            "--gfn",
            "2",
            xyz,
            "--opt",
        ]

    if charge != 0:
        execution.extend(["--charge", str(charge)])
    if unpaired_e != 0:
        execution.extend(["--uhf", str(unpaired_e)])
    sp.call(execution, stdout=sp.DEVNULL, stderr=sp.STDOUT)

    name = xyz[:-4]
    try:
        os.rename("xtbopt.xyz", f"{name}_xtbopt.xyz")
    except Exception as e:
        os.rename("xtblast.xyz", f"{name}_xtbopt.xyz")


def smiles_to_3d(smiles, level: int = 2, filename: Optional[str] = None):

    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    MAX_ITERATIONS = 10
    for i in range(2, MAX_ITERATIONS + 1):
        if mol.GetNumConformers() != 0:
            break
        mol = cleanup_mol(mol, n_confs=10 + i * 5)
    else:
        return 0

    AllChem.MolToXYZFile(mol, filename)

    if os.path.getsize(filename) == 0:
        logging.error(f"Cannot generate mol object for {smiles}, return None")
        return 0

    if level != -1:
        xtb_opt(filename, level=level)
        shutil.move(f"{filename[:-4]}_xtbopt.xyz", filename)
        files_to_remove = ["charges", ".xtboptok", "wbo", "xtbopt.log", "xtbrestart", "xtbtopo.mol"]
        for f in files_to_remove:
            try:
                os.remove(f)
            except FileNotFoundError:
                pass
    return 1


def extract_filename(filepath):
    filename_with_extension = os.path.basename(filepath)
    filename, _ = os.path.splitext(filename_with_extension)
    return filename


def xyztoinput_sf_t1(xyzfile, charge=0, spin=1):
    """
    Convert xyz file to gaussian input file for T1 optimization
    Parameters:
    1. xyzfile: xyz file
    2. charge: (int)
    3. unpaired_e: number of unpaired electrons

    Returns:
    none
    (filename.com)
    """

    name = extract_filename(xyzfile)
    path_name = xyzfile[:-4]
    with open(f"{path_name}.com", "w") as f:
        f.write(f"%chk={name}.chk\n")
        f.write(f"%nprocshared={n_cores_}\n")
        f.write(f"%mem=4GB\n")
        f.write(
            f"#p wB97XD/6-31g*  opt tda=(nstates=3,root=1,triplets) scf=(maxconventionalcycles=120,xqc) Integral=NoXCTest\n"
        )
        f.write("\n")
        f.write("Suichan wa kyomo kawaii~~~~\n")
        f.write("\n")
        f.write(f"{charge} {spin}\n")
        with open(xyzfile, "r") as xyz:
            xyz_content = xyz.readlines()
        xyz_content = xyz_content[2:]
        for line in xyz_content:
            f.write(line)
        f.write("\n")
    return 1


def xyztoinput_sf_s1(xyzfile, charge=0, spin=1):
    """
    Convert xyz file to gaussian input file
    Parameters:
    1. xyzfile: xyz file
    2. charge: (int)
    3. unpaired_e: number of unpaired electrons

    Returns:
    none
    (filename.com)
    """

    name = extract_filename(xyzfile)
    path_name = xyzfile[:-4]
    with open(f"{path_name}.com", "w") as f:
        f.write(f"%chk={name}.chk\n")
        f.write(f"%nprocshared={n_cores_}\n")
        f.write(f"%mem=4GB\n")
        f.write(
            f"#p wB97XD/6-31g*  opt tda=(nstates=3,root=1,singlets) scf=(maxconventionalcycles=120,xqc) Integral=NoXCTest\n"
        )
        f.write("\n")
        f.write("Suichan wa kyomo kawaii~~~~\n")
        f.write("\n")
        f.write(f"{charge} {spin}\n")
        with open(xyzfile, "r") as xyz:
            xyz_content = xyz.readlines()
        xyz_content = xyz_content[2:]
        for line in xyz_content:
            f.write(line)
        f.write("\n")
    return 1


def xyztoinput_sf_vert(xyzfile, charge=0, spin=1):
    """
    Convert xyz file to gaussian input file for vertical excitation
    Parameters:
    1. xyzfile: xyz file
    2. charge: (int)
    3. unpaired_e: number of unpaired electrons

    Returns:
    none
    (filename.com)
    """

    name = extract_filename(xyzfile)
    path_name = xyzfile[:-4]
    with open(f"{path_name}.com", "w") as f:
        f.write(f"%chk={name}.chk\n")
        f.write(f"%nprocshared={n_cores_}\n")
        f.write(f"%mem=4GB\n")
        f.write(f"#p wb97xd/6-31g* opt SCF=(maxconventionalcycles=120,xqc) Integral=NoXCTest\n")
        f.write("\n")
        f.write("Suichan wa kyomo kawaii~~~~\n")
        f.write("\n")
        f.write(f"{charge} {spin}\n")
        with open(xyzfile, "r") as xyz:
            xyz_content = xyz.readlines()
        xyz_content = xyz_content[2:]
        for line in xyz_content:
            f.write(line)
        f.write("\n")
        f.write("--Link1--\n")
        f.write(f"%chk={name}.chk\n")
        f.write(f"%nprocshared={n_cores_}\n")
        f.write(f"%mem=4GB\n")
        f.write(
            f"#p wB97XD/6-31g* nosymm tda=(nstates=5,50-50) scf=(maxconventionalcycles=120,xqc) Integral=NoXCTest pop=full iop(9/40=3) geom=check guess=read\n"
        )
        f.write("\n")
        f.write("Suichan wa kyomo kawaii~~~~\n")
        f.write("\n")
        f.write(f"{charge} {spin}\n")
        f.write("\n")
    return 1


def main():
    parser = argparse.ArgumentParser(
        description="Convert SMILES in csv or smi to 3D geometry and gaussian input files."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="sampling.csv",
        help="Path to the input file containing SMILES (csv or smi)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="output",
        help="Path to the output directory",
    )
    parser.add_argument(
        "-l",
        "--level",
        type=int,
        default=2,
        help="xtb level [0-2] (default=2)",
    )

    args = parser.parse_args()
    input_file = args.input
    output_dir = args.output
    level = args.level

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if input_file.endswith(".csv"):
        df = pd.read_csv(input_file)
        try:
            smiles_list = df["SMILES"].tolist()
        except KeyError:
            smiles_list = df["smiles"].tolist()
    elif input_file.endswith(".smi"):
        with open(input_file, "r") as f:
            smiles_list = f.readlines()

    for i, smiles in tqdm(enumerate(smiles_list), total=len(smiles_list)):
        smiles = smiles.strip()
        filename = os.path.join(output_dir, f"SF_{i}.xyz")
        if smiles_to_3d(smiles, level=level, filename=filename):
            xyztoinput(filename)


if __name__ == "__main__":
    # main()
    output_dir = "sf_candidates"
    for i in range(25):
        filename = os.path.join(output_dir, f"SF_{i}.xyz")
        xyztoinput(filename)

    # smiles = "CCO"
    # filename = "test.xyz"
    # smiles_to_3d(smiles, filename=filename)
    # xyztoinput(filename)
    # print(f"3D structure of {smiles} is saved as {filename}")
