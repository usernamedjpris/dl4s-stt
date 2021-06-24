import pandas as pd
import argparse
import glob
import os

def size_to_sec(size) :
    """
    size: Mo
    """
    # Formule : bit depth * freq / bits / 8 * sec = size en bytes
    # valable pour mono 16kHz 16 bits 


    return size / (16 * 16000 / 1000000 / 8)

def get_duration_from_split(cv, split):

    total_size = 0
    for index, row in split.iterrows():
        filename = row["path"]
        total_size += os.path.getsize(os.path.join(cv, f"clips/{filename}"))
        total_size_mb = total_size / 1_000_000
        split_duration = size_to_sec(total_size_mb) / 3600

    return split_duration

def calculate_duration(cv):

    duration = pd.DataFrame(columns=["split", "duration"])

    tsv = ["other.tsv", "old_test.tsv", "test.tsv", "train.tsv", "dev.tsv"]
    tsv = [os.path.join(cv, t) for t in tsv]
    for t in tsv:
        df = pd.read_csv(t, '\t')
        if "path" in df.columns:
            print(t)
            split_duration = get_duration_from_split(cv, df)       
            duration = duration.append({"split":t.split('/')[-1], "duration":split_duration}, ignore_index=True)
            print("> Durée :", split_duration)
            duration.to_csv(os.path.join(cv, "duration.csv"), index=False)

    print('> Saving duration')
    duration.to_csv(os.path.join(cv, "duration.csv"), index=False)
    print("> Durée totale :", duration["duration"].sum() / 3600)

def generate_split(cv, source_split, duration):

    # Récupération de la durée du split source
    durations = pd.read_csv(os.path.join(cv, "duration.csv"))
    ind = durations[durations["split"] == source_split].index[0]
    total_duration = durations.iloc[ind]["duration"]

    # Sample de la durée voulue dans le split
    proportion = duration / total_duration 
    source_split_df = pd.read_csv(os.path.join(cv, source_split), "\t")
    split = source_split_df.sample(frac=proportion, random_state=0)

    # Enregistrement
    split_name = source_split.split('.')[0] + "_" + str(duration) + ".tsv"
    split.to_csv(os.path.join(cv, split_name), "\t", index=False)
    print("> Split créée :", split_name)

    split_duration = get_duration_from_split(cv, split)
    print("> Vérification durée :", split_duration / 3600)

    return split


def main(args):
    if not os.path.isfile(os.path.join(args.cv, "duration.csv")):
        print("> Calcul de la durée de chaque split")
        calculate_duration(args.cv)
    
    generate_split(args.cv, "test.tsv", 1)
    generate_split(args.cv, "test.tsv", 5)
    generate_split(args.cv, "test.tsv", 10)


if __name__ == "__main__":    
    parser = argparse.ArgumentParser()

    parser.add_argument("--cv", default=None, type=str,
                        required=True, help="Dossier commonvoice")    

    
    args = parser.parse_args()
    main(args)