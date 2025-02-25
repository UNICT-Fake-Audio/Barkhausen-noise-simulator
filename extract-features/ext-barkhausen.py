import pathos as pa
from tqdm import tqdm
from glob import glob
import os

from utils import create_dir_if_not_exists, get_correct_sub_directory, get_file_name

import sys
from pathlib import Path

# hack for importing files from submodule fake-audio-detector
sys.path.append(str(Path(__file__).parent / "fake-audio-detector"))
from shared.constants import SP_FEATS_NAMES, SPECTRUM_FEATURES
from shared.feature_extraction import get_all_features_from_sample

path = "../gen-data/data/"
SUB_DIRECTORIES = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

create_dir_if_not_exists("output")

feats_names = ["file"] + SP_FEATS_NAMES + ["bit_rate"] + SPECTRUM_FEATURES

for SUB_DIRECTORY in SUB_DIRECTORIES:

    with open(
        f"output/{SUB_DIRECTORY}_output_features.csv", "a+", encoding="utf8"
    ) as f:
        f.write(",".join(feats_names + ["param"]))
        f.write("\n")

    def extract_features(file_path: str) -> None:
        sample_features = get_all_features_from_sample(file_path)

        file_name = get_file_name(file_path)

        # hackfix for defining inside the pathos process another dynamic variable which is the sub directory
        sub_dir = get_correct_sub_directory(SUB_DIRECTORIES, file_path)

        sample_features = [file_name] + sample_features + [sub_dir]

        with open(f"output/{sub_dir}_output_features.csv", "a+", encoding="utf8") as f:
            f.write(",".join([str(v) for v in sample_features]))
            f.write("\n")

    DATASET_PATH = f"../gen-data/data/{SUB_DIRECTORY}/*.wav"

    full_files_path = glob(DATASET_PATH)

    # for file_path in tqdm(full_files_path):
    #     print(file_path)
    #     extract_features(file_path)

    ncpu = 4
    with pa.multiprocessing.ProcessingPool(ncpu) as p:
        res = list(
            tqdm(p.imap(extract_features, full_files_path), total=len(full_files_path))
        )
