import argparse
import csv
import glob
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

logger = logging.getLogger("")
logger.setLevel(logging.INFO)
format = logging.Formatter("%(asctime)-10s %(message)s", "%H:%M:%S")

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(format)
logger.addHandler(handler)

format = logging.Formatter("%(asctime)-10s %(message)s", "%H:%M:%S")

# import theodore
# from ucGA.quantum_calculations.postprocessing import theodore_analysis
# from ucGA.quantum_calculations.xtb import XTB_Processor


def process_log_file(log_file_relaxed):
    T1, S1, T2, osc_strength_S1, S1ehdist = (np.nan,) * 5

    normal_termin, singlet_energies, triplet_energies, osc_strengths_singlets = (
        False,
        [],
        [],
        [],
    )

    Es = []
    with open(log_file_relaxed) as f:
        for line in f:
            if "Singlet-" in line:
                singlet_energies.append(float(line.split()[4]))
                osc_strengths_singlets.append(line.split()[8])

            if "Triplet-" in line:
                triplet_energies.append(float(line.split()[4]))

            if "Normal termination" in line:
                logger.info(f"Normal termination in {log_file_relaxed}")
                normal_termin = True

            if "SCF Done:" in line:
                Es.append(float(line.split()[4]))

    SCF_E = Es[-1]
    # print("Singlet", singlet_energies)
    # print("normal_termin", normal_termin)

    if normal_termin:
        try:
            S1 = singlet_energies[0]
            T1 = triplet_energies[0]
            T2 = triplet_energies[1]
            osc_strength_S1 = float(osc_strengths_singlets[0].split("=")[1])
        except IndexError:
            logger.error(f"Error in {log_file_relaxed}")
            raise Exception(f"Error in {log_file_relaxed}")
    else:
        logger.error(f"Error in {log_file_relaxed}")
        raise Exception(f"Error in {log_file_relaxed}")
    # try:
    #     # TODO CHANGE
    #     S1ehdist = theodore_analysis.theodore_workflow_S1_excdist(
    #         self.config.path_dens_ana_in, log_file_relaxed
    #     )
    # except theodore.error_handler.MsgError:
    #     print("Theodore error", flush=True)
    #     S1ehdist = None

    return T1, S1, T2, osc_strength_S1, S1ehdist, SCF_E


def process_log_file_d(log_filename, nstate=3):

    logger.info(f"Processing {log_filename} with {nstate} states")

    normal_termin = False
    Es = []
    S1s = []
    T1s = []
    with open(log_filename) as f:
        for line in f:
            if "SCF Done:" in line:
                Es.append(float(line.split()[4]))
            if "Singlet" in line:
                S1s.append(float(line.split()[4]))
            if "Triplet" in line:
                T1s.append(float(line.split()[4]))
            if "Normal termination" in line:
                normal_termin = True
    if normal_termin:
        SCF_E = Es[-1]
        if len(S1s) > 0:
            E1 = S1s[-nstate]
        elif len(T1s) > 0:
            E1 = T1s[-nstate]
        else:
            raise Exception(f"Cannot find S1 or T1 in {log_filename}")
        return E1, SCF_E
    else:
        raise Exception(f"Error in {log_filename}")


def extract_filename(filepath):
    filename_with_extension = os.path.basename(filepath)
    filename, _ = os.path.splitext(filename_with_extension)
    return filename


def create_log_dict(folder_path):
    log_dict = defaultdict(list)

    # TODO sort lisy to read vert first!!
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".log"):
            if "_s1" in file_name:
                base_name = file_name.replace("_s1.log", "")
            elif "_t1" in file_name:
                base_name = file_name.replace("_t1.log", "")
            else:
                base_name = file_name.replace(".log", "")

            full_path = os.path.join(folder_path, file_name)
            log_dict[base_name].append(full_path)

    return dict(log_dict)


def main():

    parser = argparse.ArgumentParser(
        description="Find the most similar molecules in the dataset to the sampled molecules."
    )
    parser.add_argument(
        "--dir",
        dest="dir",
        type=str,
        default="",
        help="Path to the .log files of the DFT//TDA calculations.",
    )
    parser.add_argument(
        "--nstate",
        dest="nstate",
        type=int,
        default=3,
        help="Number of states to be considered in the log files.",
    )

    args = parser.parse_args()
    wdir = args.dir
    nstate = args.nstate
    H_to_eV = 27.211386245988

    logs = glob.glob(f"{wdir}*.log")
    df = {
        "filename": [],
        "T1_vert": [],
        "S1_vert": [],
        "T1_d": [],
        "S1_d": [],
        "T2": [],
        "osc_strength_S1": [],
        "S1ehdist": [],
    }

    log_dict = create_log_dict(wdir)

    for logs in log_dict.values():
        filename, T1, S1, T2, osc_strength_S1, S1ehdist, e0, s1_d, t1_d = (np.nan,) * 9
        for log in logs:
            try:
                if "s1" not in log and "t1" not in log:
                    T1, S1, T2, osc_strength_S1, S1ehdist, e0 = process_log_file(log)
                    filename = extract_filename(log)

                if "s1" in log:
                    s1_1, e_s1 = process_log_file_d(log, nstate)
                    s1_d = s1_1 + (e_s1 - e0) * H_to_eV

                elif "t1" in log:
                    t1_1, e_t1 = process_log_file_d(log, nstate)
                    t1_d = t1_1 + (e_t1 - e0) * H_to_eV
            except Exception as e:
                print(e)
                continue
        df["filename"].append(filename)
        df["T1_vert"].append(T1)
        df["S1_vert"].append(S1)
        df["T1_d"].append(t1_d)
        df["S1_d"].append(s1_d)
        df["T2"].append(T2)
        df["osc_strength_S1"].append(osc_strength_S1)
        df["S1ehdist"].append(S1ehdist)

    df = pd.DataFrame(df)
    df = df.sort_values(by="filename", ascending=False)
    df.to_csv("energies.csv", index=False)


if __name__ == "__main__":
    main()
