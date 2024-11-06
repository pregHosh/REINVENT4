import argparse
import json
import math
import os
import os.path as op
import pickle
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from os import devnull

import chemprop
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import toml
import torch
from excit_obj_func import energy_score, energy_score_sti
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.DataStructs import TanimotoSimilarity
from tqdm import tqdm

from reinvent_plugins.components.SAScore.sascorer import calculateScore as calculateSAScore

_fscores = None


# %% Model reader
def build_sample_toml(model_file, output_file, num_smiles):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = {
        "run_type": "sampling",
        "device": device,
        "json_out_config": "sampling.json",
        "parameters": {
            "model_file": model_file,
            "output_file": output_file,
            "num_smiles": num_smiles,
            "unique_molecules": True,
            "randomize_smiles": True,
        },
    }

    with open("sampling.toml", "w") as toml_file:
        toml.dump(data, toml_file)


def sample_smiles(model_file, output_file, num_smiles):
    build_sample_toml(model_file, output_file, num_smiles)
    os.system(f"reinvent sampling.toml")
    os.remove("sampling.toml")
    df = pd.read_csv(output_file)
    smiles = df["SMILES"].to_list()
    return smiles


def prop_prediction(pred_model_path, smiles_list):

    args = [
        "--checkpoint_dir",  # ChemProp models directory
        pred_model_path,
        "--test_path",
        "/dev/null",
        "--preds_path",
        "/dev/null",
    ]

    # Run the SMILES through ChemProp
    with suppress_output():
        chemprop_args = chemprop.args.PredictArgs().parse_args(args)
        chemprop_model = chemprop.train.load_model(args=chemprop_args)

    preds = chemprop.train.make_predictions(
        model_objects=chemprop_model,
        smiles=[[smiles] for smiles in smiles_list],
        args=chemprop_args,
        return_invalid_smiles=True,
        return_uncertainty=False,
    )

    # for smiles in tqdm(smiles_list):

    #     with suppress_output():
    #         chemprop_args = chemprop.args.PredictArgs().parse_args(args)
    #         chemprop_model = chemprop.train.load_model(args=chemprop_args)

    #         pred = chemprop.train.make_predictions(
    #             model_objects=chemprop_model,
    #             smiles=[[smiles]],
    #             args=chemprop_args,
    #             return_invalid_smiles=True,
    #             return_uncertainty=False,
    #         )
    #         preds.append(pred[0])

    return np.asanyarray(preds, dtype=float)


# %% phys chem functions
class BasicMolecularMetrics(object):
    """
    Valid amongst all generated molecules
    Uniqueness amongst valid molecules
    Novelty amongst unique molecules
    """

    def __init__(self, dataset_smiles_list):
        self.dataset_smiles_list = dataset_smiles_list

    # TODO consider change to cell2mol xyz2mol instead and able to skip fail mol
    def compute_validity(self, generated):
        """generated smiles"""
        valid = []

        for smiles in generated:
            mol = Chem.MolFromSmiles(smiles)
            mol = Chem.AddHs(mol)
            emd = AllChem.EmbedMolecule(mol)
            if emd == 0:
                valid.append(smiles)

        return valid, len(valid) / len(generated)

    def compute_uniqueness(self, valid):
        """valid: list of SMILES strings."""
        return list(set(valid)), len(set(valid)) / len(valid)

    def compute_novelty(self, unique):
        num_novel = 0
        novel = []
        for smiles in unique:
            if smiles not in self.dataset_smiles_list:
                novel.append(smiles)
                num_novel += 1
        return novel, num_novel / len(unique)

    def evaluate(self, generated):
        """generated: list of pairs (positions: n x 3, atom_types: n [int])
        the positions and atom types should already be masked."""
        valid, validity = self.compute_validity(generated)
        # print(f"Validity over {len(generated)} molecules: {validity * 100 :.2f}%")
        if validity > 0:
            unique, uniqueness = self.compute_uniqueness(valid)
            # print(
            #     f"Uniqueness over {len(valid)} valid molecules: {uniqueness * 100 :.2f}%"
            # )

            if self.dataset_smiles_list is not None:
                _, novelty = self.compute_novelty(unique)
                # print(
                #     f"Novelty over {len(unique)} unique valid molecules: {novelty * 100 :.2f}%"
                # )
            else:
                novelty = 0.0
        else:
            novelty = 0.0
            uniqueness = 0.0
            unique = None

        return [validity, uniqueness, novelty], unique


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
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=2)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=2)

    # Calculate Tanimoto similarity
    similarity = TanimotoSimilarity(fp1, fp2)
    return similarity


def get_scaffold(smiles_list, threshold=0.69):
    scaffolds = []
    duplicate = False
    for smiles in smiles_list:
        scaffold = to_scaffold(smiles)
        if scaffold not in scaffolds:
            for id, i in enumerate(scaffolds):
                similarity = calculate_similarity(i, scaffold)
                if similarity > threshold:
                    print(f"Found similar scaffold: {scaffold} with {scaffolds[id]}")
                    duplicate = True
                    break
            if not duplicate:
                scaffolds.append(scaffold)
    return scaffolds


def get_candidate(df_score: pd.DataFrame, n_select: int, theshold: float):

    smiles = df_score["SMILES"].to_list()
    score = df_score["Score"].to_list()

    df_candidate = {"SMILES": [], "Score": []}

    for i in range(n_select):
        if i == 0:
            df_candidate["SMILES"].append(smiles[0])
            df_candidate["Score"].append(score[0])
        else:
            for j in range(len(df_candidate["SMILES"])):
                if calculate_similarity(smiles[i], df_candidate["SMILES"][j]) > theshold:
                    pass
            else:
                df_candidate["SMILES"].append(smiles[i])
                df_candidate["Score"].append(score[i])
    df_candidate = pd.DataFrame(df_candidate)
    return df_candidate


# %% plot functions
def plot_kde_dist(pred, task_names, output_name):

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    sns.kdeplot(pred, label=task_names, color="blue", fill=True, linewidth=3)
    ax.set_title(f"{task_names} Distribution", fontsize=22)
    ax.set_xlabel(task_names, fontsize=22)
    ax.set_ylabel("Frequency", fontsize=2)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()

    if output_name is None:
        output_name = "kde_targets.png"
    plt.savefig(output_name, dpi=300)


def plot_kde_dist_multiple(pred, task_names, output_name):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    # Define a color palette
    colors = sns.color_palette("hsv", len(task_names))

    # Iterate over each property and plot its KDE
    for i, task_name in enumerate(task_names):
        sns.kdeplot(pred[:, i], label=task_name, color=colors[i], fill=True, linewidth=3)

    # ax.set_title(f"{' and '.join(task_names)} Distribution", fontsize=22)
    ax.set_xlabel("Values (eV)", fontsize=32)
    ax.set_ylabel("Frequency", fontsize=32)
    plt.legend(fontsize=32)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.tight_layout()

    if output_name is None:
        output_name = "kde_targets.png"
    plt.savefig(output_name, dpi=300)
    plt.show()


def plot_hist_dist(pred, task_names, output_name, num_bins=50):

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    bin_min = np.floor(np.min(pred))
    bin_max = np.ceil(np.max(pred))
    bin_width = np.ceil((bin_max - bin_min) / num_bins)
    bin_edges = np.arange(bin_min, bin_max + bin_width, bin_width)

    ax.hist(
        pred,
        bins=bin_edges,
        color="#1266A4",
        edgecolor="#1266A4",
        alpha=0.8,
        align="mid",
        rwidth=0.8,
    )

    ax.set_title(f"{task_names} Distribution", fontsize=22)
    ax.set_xlabel(task_names, fontsize=22)
    ax.set_ylabel("Number of occurence", fontsize=2)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()

    if output_name is None:
        output_name = "hist_targets.png"
    plt.savefig(output_name, dpi=300)


@contextmanager
def suppress_output():
    """Context manager to redirect stdout and stderr to /dev/null"""

    with open(devnull, "w") as nowhere:
        with redirect_stderr(nowhere) as err, redirect_stdout(nowhere) as out:
            yield (err, out)


# %%
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Analyze generative model by sampling and plotting distribution."
    )
    parser.add_argument(
        "--model_file",
        type=str,
        default="model/formed.prior",
        help="Path to the prior model file (default=model/formed.prior).",
    )
    parser.add_argument(
        "-p",
        "--prop_model",
        type=str,
        default="chemprop_model/form_alltarget/fold_0/model_0",
        help="""Path to the property prediction model (default=chemprop_model/form_alltarget/fold_0/model_0).
        """,
    )
    parser.add_argument(
        "--get_score",
        action="store_true",
        help="""Calculate the energy score of the sampled SMILES.
            The model should predict the following properties: S1, T1, exciton_size.""",
    )
    parser.add_argument(
        "--n_candidate",
        type=int,
        default=10,
        help="Number of candidate molecules to be selected (default=10).",
    )
    parser.add_argument(
        "--similarity_threshold",
        type=int,
        default=0.6,
        help="Similarity score threshold for the candidate molecules selectin.",
    )
    parser.add_argument(
        "--gaussian_input_id",
        type=int,
        default=0,
        help="Specify the id value to get Gaussian input for the energy score calculation..",
    )

    parser.add_argument(
        "--num_smiles",
        type=int,
        default=1024,
        help="Number of SMILES to be sampled (default=1023).",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="sampling.csv",
        help="Path to the output file (default=sampling.csv).",
    )
    parser.add_argument(
        "--names",
        dest="property_names",
        nargs="+",
        default=["S1", "T1", "exciton_size"],
        help="Specify the task names to be predicted (default=['S1', 'T1', 'exciton_size'])",
    )
    parser.add_argument(
        "--ref",
        dest="ref",
        type=str,
        default=None,
        help="Path to the reference .smi file (default=None).",
    )
    parser.add_argument(
        "-d",
        "--dir",
        dest="output_dir",
        default="analyze_res",
        help="Output directory for the analysis results (default=analyze_res).",
    )
    parser.add_argument(
        "--substructure_analysis",
        dest="substructure_analysis",
        action="store_true",
        help="Perform substructure analysis (default=False).",
    )

    args = parser.parse_args()

    model_file = args.model_file
    output_file = args.output_file
    property_names = args.property_names
    num_smiles = args.num_smiles
    pred_model_path = args.prop_model
    get_score = args.get_score
    similarity_threshold = args.similarity_threshold
    gaussian_input_id = args.gaussian_input_id
    n_candidate = args.n_candidate
    ref = args.ref
    output_dir = args.output_dir
    ss = args.substructure_analysis
    smiles_sampled = sample_smiles(model_file, output_file, num_smiles)

    property_names = ["S$_1$", "T$_1$", "d$_{e^- -h^+}^{S_1}$"]
    if not (os.path.exists(output_dir)):
        os.makedirs(output_dir, exist_ok=True)

    if ref is not None:
        print("Calculating basic molecular metrics...")
        smiles_ref = []
        with open(ref, "r") as file:
            for line in file:
                smiles_ref.append(line.strip())
        metrics = BasicMolecularMetrics(smiles_ref)
        rdkit_metrics = metrics.evaluate(smiles_sampled)
        print(f"Validity: {rdkit_metrics[0][0]:.2f}")
        print(f"Uniqueness: {rdkit_metrics[0][1]:.2f}")
        print(f"Novelty: {rdkit_metrics[0][2]:.2f}")

    if ss:
        print("Performing substructure analysis...")
        scaffolds = get_scaffold(smiles_sampled)
        print(f"Number of unique scaffolds: {len(scaffolds)}")
        with open(os.path.join(output_dir, "scaffolds.txt"), "w") as file:
            for scaffold in scaffolds:
                file.write(f"{scaffold}\n")

    sascores = np.zeros(len(smiles_sampled))
    mws = np.zeros(len(smiles_sampled))
    for i, smiles in enumerate(smiles_sampled):
        mol = Chem.MolFromSmiles(smiles)
        sascore = calculateSAScore(mol)
        sascores[i] = sascore
        mws[i] = Descriptors.MolWt(mol)

    ref = args.ref
    preds = prop_prediction(pred_model_path, smiles_sampled)
    preds = np.concatenate((preds, sascores.reshape(-1, 1), mws.reshape(-1, 1)), axis=1)
    property_names.extend(["SAScore", "Molecular Weight"])

    np.savetxt(os.path.join(output_dir, "preds.txt"), preds)

    for i in range(preds.shape[1]):
        plot_kde_dist(
            preds[:, i],
            property_names[i],
            os.path.join(output_dir, f"kde_plot_{property_names[i]}.png"),
        )
        plot_hist_dist(
            preds[:, i],
            property_names[i],
            os.path.join(output_dir, f"hist_plot_{property_names[i]}.png"),
        )

    df_dict = {"SMILES": smiles_sampled}
    for i, prop in enumerate(property_names):
        df_dict[prop] = preds[:, i]
    df_prop = pd.DataFrame(df_dict)
    df_prop.to_csv(os.path.join(output_dir, "prop_smiles.csv"), index=False)

    if len(property_names) > 1:
        plot_kde_dist_multiple(
            preds[:, :-2], property_names[:-2], os.path.join(output_dir, "kde_plot_all.png")
        )

    if get_score:
        print("Calculating energy score...\n")
        score = np.zeros(preds.shape[0])
        for i in range(preds.shape[0]):
            score[i] = energy_score(preds[i, 1], preds[i, 0])

        df_scores = pd.DataFrame({"SMILES": smiles_sampled, "Score": score})
        df_scores = df_scores.sort_values(by="Score", ascending=False)
        df_scores["Rank"] = np.arange(1, len(df_scores) + 1)

        df_scores = df_scores.reset_index(drop=True)

        print(df_scores)

        df_scores.to_csv(os.path.join(output_dir, "scored_smiles.csv"), index=False)

        df_candidate = get_candidate(df_scores, n_candidate, similarity_threshold)
        print(df_candidate)
        df_candidate.to_csv(os.path.join(output_dir, "candidate_smiles.csv"), index=False)

        if gaussian_input_id != 0:
            from smiles2geom_sf import smiles_to_3d, xyztoinput

            smiles_list = df_candidate["SMILES"].to_list()
            gaussian_inp_dir = os.path.join(output_dir, "gaussian_input")
            os.makedirs(gaussian_inp_dir, exist_ok=True)

            for i, smiles in tqdm(enumerate(smiles_list), total=len(smiles_list)):
                smiles = smiles.strip()
                filename = os.path.join(gaussian_inp_dir, f"{gaussian_input_id}_{i}.xyz")
                if smiles_to_3d(smiles, level=2, filename=filename):
                    xyztoinput(filename)
