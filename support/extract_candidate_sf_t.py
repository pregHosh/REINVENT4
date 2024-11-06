import argparse
import os

import numpy as np
import pandas as pd
from chimera import Chimera
from excit_obj_func import energy_score
from findsimilar import calculate_similarity, get_most_similar
from model_analysis import prop_prediction, sample_smiles
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.DataStructs import TanimotoSimilarity
from smiles2geom_sf import smiles_to_3d, xyztoinput_sf_vert
from tqdm import tqdm

from reinvent_plugins.components.SAScore.sascorer import calculateScore as calculateSAScore


def get_candidate(df_score: pd.DataFrame, n_select: int, theshold: float):

    smiles = df_score["SMILES"].to_list()
    score = df_score["Score"].to_list()

    df_candidate = {"SMILES": [], "Score": []}

    for i in range(n_select):
        if df_score["T1"].iloc[i] > 1.7:
            continue
        if df_score["S1"].iloc[i] < 1:
            continue
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


# %%
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Extract candidate molecules from the agent model for the TDDFT//DFT calculations."
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

    smiles_sampled = sample_smiles(model_file, output_file, num_smiles)
    coeff = [0.5, -0.1, 0.4]  # score, sascore, similarity

    if not (os.path.exists(output_dir)):
        os.makedirs(output_dir, exist_ok=True)

    sascores = np.zeros(len(smiles_sampled))
    mws = np.zeros(len(smiles_sampled))
    for i, smiles in enumerate(smiles_sampled):
        mol = Chem.MolFromSmiles(smiles)
        sascore = calculateSAScore(mol)
        sascores[i] = sascore
        mws[i] = Descriptors.MolWt(mol)

    # %% Filter #1
    # TODO get uq here as well
    preds = prop_prediction(pred_model_path, smiles_sampled)
    preds = np.concatenate((preds, sascores.reshape(-1, 1), mws.reshape(-1, 1)), axis=1)

    score = np.zeros(preds.shape[0])
    t1s = []
    s1s = []
    for i in range(preds.shape[0]):
        score[i] = energy_score(preds[i, 1], preds[i, 0])
        t1s.append(preds[i, 1])
        s1s.append(preds[i, 0])

    df_scores = pd.DataFrame({"SMILES": smiles_sampled, "Score": score})
    df_scores = df_scores.sort_values(by="Score", ascending=False)
    df_scores["Rank"] = np.arange(1, len(df_scores) + 1)
    df_scores["T1"] = t1s
    df_scores["S1"] = s1s

    df_scores = df_scores.reset_index(drop=True)
    df_scores.to_csv(os.path.join(output_dir, "score.csv"), index=False)

    # TODO to accout for uq for candidate selection
    df_candidate = get_candidate(df_scores, n_candidate, similarity_threshold)
    smiles_candidate = df_candidate["SMILES"].values

    # %% Filter #2
    smiles_ref = []
    with open(ref, "r") as file:
        for line in file:
            smiles_ref.append(line.strip())

    most_similars = []
    sim_scores = []
    sascores = []

    # TODO Ruben's validitiy check
    for smiles in tqdm(smiles_candidate, desc="Finding most similar", total=len(smiles_candidate)):
        most_similar, similarity = get_most_similar(smiles, smiles_ref, threhold=0.87)
        sascore = calculateSAScore(Chem.MolFromSmiles(smiles))
        most_similars.append(most_similar)
        sim_scores.append(similarity)
        sascores.append(sascore)
        print(f"Most similar: {most_similar}, Similarity: {similarity} for {smiles}")

    df_candidate["Most similar"] = most_similars
    df_candidate["Similarity"] = sim_scores
    df_candidate["sascore"] = sascores

    scores = df_candidate["Score"].to_numpy().reshape(-1, 1)
    sascores = df_candidate["sascore"].to_numpy().reshape(-1, 1) / 10
    similarities = df_candidate["Similarity"].to_numpy().reshape(-1, 1)
    concat_s = np.concatenate([scores, sascores, similarities], axis=1)

    # chimera = Chimera(
    #     tolerances=coeff, absolutes=[False, False, False], goals=["max", "min", "max"]
    # )
    # all_score = chimera.scalarize(concat_s)
    all_score = concat_s.dot(coeff)
    df_candidate["AllScore"] = all_score
    df_candidate = df_candidate.sort_values(by="AllScore", ascending=False)
    df_candidate.to_csv(os.path.join(output_dir, "candidate.csv"), index=False)

    smiles_candidate = df_candidate["SMILES"].values
    gaussian_inp_dir = os.path.join(output_dir, "gaussian_input")
    os.makedirs(gaussian_inp_dir, exist_ok=True)
    for i, smiles in tqdm(enumerate(smiles_candidate), total=len(smiles_candidate)):
        smiles = smiles.strip()
        filename = os.path.join(gaussian_inp_dir, f"{gaussian_input_id}_{i}.xyz")
        if smiles_to_3d(smiles, level=2, filename=filename):
            xyztoinput_sf_vert(filename)
