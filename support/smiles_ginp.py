import argparse
import os
import shutil

from smiles2geom_sf import smiles_to_3d, xyztoinput_sf_s1, xyztoinput_sf_t1, xyztoinput_sf_vert
from tqdm import tqdm

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Extract candidate molecules from the agent model for the TDDFT//DFT calculations."
    )
    parser.add_argument(
        "-i",
        "--i",
        "--input_smi",
        dest="input_smi",
        type=str,
        default="test.smi",
        help="Path to the input SMILES file (default=test.smi).",
    )
    parser.add_argument(
        "--xyz_folder",
        type=str,
        default=None,
        help="Path to the folder containing XYZ files.",
    )
    parser.add_argument(
        "--gaussian_input_id",
        type=str,
        default="123124",
        help="Specify the id value to get Gaussian input for the energy score calculation.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="gaussian_input",
        help="Path to the Gaussian input files.",
    )
    args = parser.parse_args()
    path_smi = args.input_smi
    xyz_folder = args.xyz_folder
    gaussian_input_id = args.gaussian_input_id
    gaussian_inp_dir = args.output_file

    os.makedirs(gaussian_inp_dir, exist_ok=True)

    if xyz_folder:
        # Process XYZ files from folder
        xyz_files = [f for f in os.listdir(xyz_folder) if f.endswith(".xyz")]
        for i, xyz_file in tqdm(enumerate(xyz_files), total=len(xyz_files)):
            try:
                filepath = os.path.join(xyz_folder, xyz_file)
                filename_s1 = os.path.join(
                    gaussian_inp_dir, f"{os.path.splitext(xyz_file)[0]}_s1.xyz"
                )
                filename_t1 = os.path.join(
                    gaussian_inp_dir, f"{os.path.splitext(xyz_file)[0]}_t1.xyz"
                )

                # filepath_ = os.path.join(gaussian_inp_dir, f"{os.path.splitext(xyz_file)[0]}.xyz")
                shutil.copy(filepath, filename_s1)
                shutil.copy(filepath, filename_t1)
                xyztoinput_sf_vert(filepath)
                filepath_ = os.path.join(xyz_folder, f"{os.path.splitext(xyz_file)[0]}.com")
                shutil.move(filepath_, gaussian_inp_dir)

                xyztoinput_sf_s1(filename_s1)
                xyztoinput_sf_t1(filename_t1)
            except Exception as e:
                print(f"Error in {xyz_file}: {e}")
                continue

    else:
        # Process SMILES from input file
        smiles = []
        with open(path_smi, "r") as f:
            for line in f:
                smiles.append(line)

        for i, smiles in tqdm(enumerate(smiles), total=len(smiles)):
            try:
                smiles = smiles.strip()
                filename = os.path.join(gaussian_inp_dir, f"{gaussian_input_id}_{i}.xyz")
                if smiles_to_3d(smiles, level=2, filename=filename):

                    filename_s1 = os.path.join(gaussian_inp_dir, f"{gaussian_input_id}_{i}_s1.xyz")
                    filename_t1 = os.path.join(gaussian_inp_dir, f"{gaussian_input_id}_{i}_t1.xyz")

                    shutil.copy(filename, filename_s1)
                    shutil.copy(filename, filename_t1)
                    xyztoinput_sf_vert(filename)
                    xyztoinput_sf_s1(filename_s1)
                    xyztoinput_sf_t1(filename_t1)
                for file in os.listdir(gaussian_inp_dir):
                    if file.endswith(".xyz"):
                        os.remove(os.path.join(gaussian_inp_dir, file))
            except Exception as e:
                print(f"Error in {smiles}: {e}")
                continue
