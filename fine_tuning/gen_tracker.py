import pandas as pd
import glob 
from subprocess import run
import os
from collections import Counter


    
def main():
    # Import et nettoyage
    csv_path = 'data/dataset_FR_gen.csv'
    data = pd.read_csv(csv_path)
    data = data.drop(["category", "topic"], axis=1)
    data = data.dropna()

    # Sélection des contributeurs fiables
    data = data[data["file"].apply(lambda x : "CLEMENT" in x or "MONTA" in x or "GOULARD" in x)]
#     data = data[data["file"].apply(lambda x : "MONTA" in x)]


    # Supprime les noms de fichiers avec espace
    data = data.drop(data[data['file'].str.contains("ARN")].index)

    # On enlève les chevrons qui apparaissent dans les "<euh>"
    transcription = data["transcription"].apply(lambda x : x.replace("<", "").replace(">", ""))
    data["transcription"] = transcription

    # On supprime les fichiers non présents dans le tracker
    delete_untracked(data)

    # Save
    data.to_csv("data/dataset_FR_filtered.csv", index=False)
    
    gen_trans(data)
    gen_dict()
    
main()