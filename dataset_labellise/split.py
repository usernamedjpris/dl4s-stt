import pandas as pd
import argparse
import glob
import os

def size_to_sec(size) :
    # Formule : bit depth * freq / bits / 8 * sec = size en bytes
    
    return size / (16 * 16000 / 1000000 / 8)

def calculate_duration(cv):

    duration = pd.DataFrame(["split", "duration"])

    tsv = glob.glob(os.path.join(cv, "*.tsv"))
    all_split_size = 0.0
    for t in tsv:
        total_size = 0.0

        df = pd.read_csv(t, '\t')
        if "path" in df.columns:
            print(t)
            
            df["path"] = df["path"].str.replace("mp3", "wav")

            for index, row in df.iterrows():
                filename = row["path"]
                total_size += os.path.getsize(os.path.join(cv, f"clips/{filename}"))
            total_size_mb = total_size / 1_000_000
            all_split_size += total_size_mb
            duration_split = size_to_sec(total_size_mb) / 3600

            duration.append({"split":t, "duration":duration_split}, ignore_index=True)
            print("> Durée :", duration_split)

    # écrire dans un fichier de meta data
    print('> Saving duration')
    duration.to_csv(os.path.join(cv, "duration.csv"), index=False)
    print("> Durée totale :", size_to_sec(all_split_size) / 3600)

def generate_split(cv, source_split, duration):

    meta = pd.read_csv("meta.csv")
    total_duration = meta[meta["name"] == source_split]["total_duration"]

    frac = duration / total_duration 
    source_split_df = pd.read_csv(os.path.join(cv, source_split))
    split = source_split_df.sample(frac)
    split_name = source_split.split('.')[0] + "_" + str(duration)
    split.to_csv(os.path.join(cv, split_name + ".tsv"))

    return split


def main(args):
    calculate_duration(args.cv)
    pass


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