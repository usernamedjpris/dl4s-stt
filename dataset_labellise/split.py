import pandas as pd
import argparse
import glob
import os

def size_to_sec(size) :
    # Formule : bit depth * freq / bits / 8 * sec = size en bytes
    
    return size / (16 * 16000 / 1000000 / 8)

def calculate_duration(cv):

    duration = pd.DataFrame(columns=["split", "duration"])

    tsv = ["other.tsv", "old_test.tsv", "test.tsv", "train.tsv", "dev.tsv"]
    tsv = [os.path.join(cv, t) for t in tsv]
    # tsv = glob.glob(os.path.join(cv, "*.tsv"))
    all_split_size = 0.0
    for t in tsv:
        total_size = 0.0

        df = pd.read_csv(t, '\t')
        if "path" in df.columns:
            print(t)
            
            # df["path"] = df["path"].str.replace("mp3", "wav")

            for index, row in df.iterrows():
                filename = row["path"]
                total_size += os.path.getsize(os.path.join(cv, f"clips/{filename}"))
            total_size_mb = total_size / 1_000_000
            all_split_size += total_size_mb
            duration_split = size_to_sec(total_size_mb) / 3600

            duration = duration.append({"split":t, "duration":duration_split}, ignore_index=True)
            print("> Durée :", duration_split)
            duration.to_csv(os.path.join(cv, "duration.csv"), index=False)


    # écrire dans un fichier de meta data
    print('> Saving duration')
    duration.to_csv(os.path.join(cv, "duration.csv"), index=False)
    print("> Durée totale :", size_to_sec(all_split_size) / 3600)

def generate_split(cv, source_split, duration):

    durations = pd.read_csv(os.path.join(cv, "duration.csv"))
    total_duration = durations[durations["split"].str.contains(source_split)]["duration"]

    frac = duration / total_duration 
    source_split_df = pd.read_csv(os.path.join(cv, source_split), "\t")
    split = source_split_df.sample(frac)
    split_name = source_split.split('.')[0] + "_" + str(duration) + ".tsv"
    split.to_csv(os.path.join(cv, split_name), "\t", index=False)

    return split


def main(args):
    if not os.path.isfile(os.path.join(args.cv, "duration.csv")):
        calculate_duration(args.cv)
    
    generate_split(args.cv, "test.tsv", 5)

    # TODO Calculer la durée totale en heure par split
    # splits = []


if __name__ == "__main__":    
    parser = argparse.ArgumentParser()

    parser.add_argument("--cv", default=None, type=str,
                        required=True, help="Dossier commonvoice")    
    parser.add_argument("--bdd", default=None, type=str,
                        required=False, help="Chemin vers le dossier contenant les BDD (pickle)")    
    
    args = parser.parse_args()
    main(args)