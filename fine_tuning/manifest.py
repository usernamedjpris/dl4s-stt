import pandas as pd
from subprocess import run
import argparse
import os
from shutil import copyfile
import soundfile

def get_parser():
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", default=None, type=str,
                        required=True, help="Path vers le split de train (tsv)")
    parser.add_argument("--valid", default=None, type=str,
                        required=True, help="Path vers le split de valid (tsv)")
    parser.add_argument("--fairseq", default=None, type=str,
                        required=True, help="Path vers le dossier fairseq")

    return parser

def generate_manifest(split, output):
    df = pd.read_csv(split, "\t")
    root_path = args.path.join(split.split("/")[0], "clips") # clips folder dans le dataset
    with open(output, "w") as split_f:
        print(root_path, file=split_f)
        for index, row in df.iterrows() :
            path = row["path"]
            frames = soundfile.info(path).frames
            print(f"{path}\t{frames}", file=split_f)
    
def main(args):
    run_id = args.train.split("_")[-1] # Nombre d'heure dans le split
    output_dir = os.path.join(args.log_dir, f'pretraining_{run_id}' )

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    generate_manifest(args.train, os.path.join(output_dir, "train.tsv"))
    generate_manifest(args.valid, os.path.join(output_dir, "valid.tsv"))

if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    main(args)