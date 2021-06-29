import pandas as pd
from subprocess import run
import argparse
import os
from shutil import copyfile
import soundfile

def generate_manifest(fairseq, output_dir, split, data_path): 
    run('python ' + 
        os.path.join(fairseq, "examples/wav2vec/wav2vec_manifest.py") + " " + data_path +
        " --dest " + data_path + 
        " --ext wav --valid-percent 0", shell=True)
    os.remove(os.path.join(data_path, "valid.tsv"))
    copyfile(os.path.join(data_path, "train.tsv"), os.path.join(output_dir, split))

def generate_manifest(split):
    df = pd.read_csv(split, "\t")
    root_path = args.path.join(split.split("/")[0], "clips") # clips folder dans le dataset
    with open(split, "w") as split_f:
        print(root_path, file=split_f)
        for index, row in df.iterrows() :
            path = row["path"]
            frames = soundfile.info(path).frames
            print(f"{path}\t{frames}", file=split_f)
    
def main(args):

    generate_manifest(args.train, args.output_dir)
    generate_manifest(args.valid, args.output_dir)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=None, type=str,
                        required=True, help="Path vers le split de train (tsv)")
    parser.add_argument("--valid", default=None, type=str,
                        required=True, help="Path vers le split de valid (tsv)")
    parser.add_argument("--fairseq", default=None, type=str,
                        required=True, help="Path vers le dossier fairseq")
    parser.add_argument("--fairseq", default=None, type=str,
                        required=True, help="Path vers le dossier fairseq")
    args = parser.parse_args()

main(args)