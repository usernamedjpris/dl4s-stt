import pandas as pd
from subprocess import run
import argparse
import os
from shutil import copyfile
import soundfile
from tqdm import tqdm

def get_parser():
    
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--cv", default=None, type=str,
                        required=True, help="Path vers le dossier commonvoice")
    parser.add_argument("-t", "--train", default=None, type=str,
                        required=True, help="Path vers le split de train (tsv)")
    parser.add_argument("-v", "--valid", default=None, type=str,
                        required=True, help="Path vers le split de valid (tsv)")
    parser.add_argument("-f", "--fairseq", default=None, type=str,
                        required=True, help="Path vers le dossier fairseq")
    parser.add_argument("-o", "--log_dir", default=None, type=str,
                        required=True, help="Path vers le dossier des logs")
    return parser

def generate_manifest(split, cv, output):
    print("> Generate manifest for split ", split)
    df = pd.read_csv(os.path.join(cv, split), "\t")
    root_path = os.path.join(cv, "clips") # clips folder dans le dataset
    with open(output, "w") as split_f:
        print(root_path, file=split_f)
        for index, row in tqdm(df.iterrows()) :
            abs_path = os.path.join(root_path, row["path"])
            rel_path = row["path"]
            frames = soundfile.info(abs_path).frames
            print(f"{rel_path}\t{frames}", file=split_f)
    print("> Done!")

def main(args):
    run_id = args.train.split("_")[-1] # Nombre d'heure dans le split
    output_dir = os.path.join(args.log_dir, f'pretraining_{run_id}' )

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    generate_manifest(args.train, args.cv, os.path.join(output_dir, "train.tsv"))
    generate_manifest(args.valid, args.cv, os.path.join(output_dir, "valid.tsv"))

if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    main(args)