import pandas as pd
import os
import glob
from subprocess import run
import numpy as np
import argparse
from tqdm import tqdm

def process_wp1(path, contrib_to_remove):
    """ Retourne un dataframe filtré à partir du tracker wp1 

    :param path: chemin vers le dossier contenant le tracker csv
    :type path: str
    :param contrib_to_remove: Noms des contributeurs à enlever 
    :type contrib_to_remove: array

    :rtype: DataFrame
    """

    ####
    ## Import et nettoyage
    ####

    wp1 = pd.read_csv(os.path.join(path, "dataset_FR_gen.csv"))

    # Calcul durée
    wp1["duration"] = wp1["end"] - wp1["start"]

    # Correction doublon en categorie
    wp1["category"] = wp1["category"].replace(["Cours"], "cours")

    # Filtre contributeurs
    contributeurs = set([x.split("_")[-2] for x in wp1["file"]])
    to_remove = ["audio"]
    to_remove = to_remove + contrib_to_remove
    for x in to_remove:
        contributeurs.remove(x)
    contributeurs = list(contributeurs)
    wp1['keep'] = wp1["file"].apply(lambda x : x.split('_')[-2] in contributeurs)
    wp1 = wp1[wp1["keep"] == True]

    # Harmonisation nom des colonnes avec CV
    wp1 = wp1[["file", "transcription", "duration", "category", "topic"]]
    wp1["path"] = wp1["file"].apply(lambda x : x.replace(" ", "_"))
    wp1["path"] = wp1["path"].apply(lambda x : os.path.join("wp1", x))
    wp1["sentence"] = wp1["transcription"]
    wp1 = wp1.drop(["file", 'transcription'], axis=1)

    return wp1

    ####
    ## Sampling
    ####

def test_sample_wp1(wp1, target_duration):
    """ Retoune un échantillon du WP1 équilibré sur les catégories

    :param wp1: Dataframe clean du wp1
    :type wp1: DataFrame
    :param target_duration: Nombre d'heures dans le dataframe final
    :type target_duration: int

    :rtype: DataFrame
    """
    nb_cat = 5

    # Calcul du nb d'échantillons à prendre pour avoir 
    # [target duration]h en [nb_cat] catégories
    nb_sample_min = (3600 * target_duration) / (nb_cat * wp1["duration"].mean())
    print(nb_sample_min)

    # Filtre sur les catégories
    threshold = nb_sample_min
    dummy_category = pd.get_dummies(wp1["category"]).sum()
    filtered = dummy_category.apply(lambda x : 0 if x < threshold else x)
    filtered = filtered.drop(filtered[filtered == 0].index)
    filtered["other"] = dummy_category.sum() - filtered.sum()

    categories = list(filtered.index)
    categories.remove("other")


    nb_sample = min(min(filtered), int(nb_sample_min))
    wp1_balanced_duration = wp1["duration"].mean() * nb_sample * nb_cat / 3600

    if wp1_balanced_duration < target_duration:
        print(f">> !! Il faut environ {int(nb_sample_min)} sample par catégorie pour avoir \
            {target_duration}h dans le dataset avec {nb_cat} catégories")
    print(f"Nombre de sample disponible par catégorie : {nb_sample}")
    print(f"Durée totale du dataset équilibré : {wp1_balanced_duration:.2f}h")

    # Sampling sur les catégories
    subset = []
    for cat in categories:
        samples = wp1[wp1["category"] == cat].sample(n=nb_sample, random_state=41)
        subset.append(samples)
    wp1_balanced = pd.concat(subset)

    return wp1_balanced

def train_sample_wp1(wp1, wp1_test_sample):
    # wp1_test_sample est un sous ensemble de wp1
    # Pour obtenir le complémentaire dans wp1 du split test 
    # On concatène et on drop les doublons
    df = pd.concat([wp1, wp1_test_sample]).drop_duplicates(keep=False)
    return df


def process_cv(path_train, path_test):
    cv_train = pd.read_csv(path_train, '\t')
    cv_train = cv_train[['path', 'sentence']]

    cv_test = pd.read_csv(path_test, '\t')
    cv_test = cv_test[['path', 'sentence']]

    return cv_train, cv_test

def generate_tsv(args):
    """
    Génére et sauvegarde un tsv en fusionnant le split CV et le WP1 filtré
    """

    # Import et split du dataset WP1
    wp1 = process_wp1(args.wp1, CONTRIB_TO_REMOVE)
    wp1_test_sample = test_sample_wp1(wp1, args.target_duration)
    wp1_train_sample = train_sample_wp1(wp1, wp1_test_sample)

    # Import des splits du dataset commonvoice
    train_path = os.path.join(args.cv, "train.tsv")
    test_path = os.path.join(args.cv, "test.tsv")
    cv_train, cv_test = process_cv(train_path, test_path)

    # Ajout des sources
    cv_train["origin"] = ['commonvoice'] * len(cv_train)
    cv_test["origin"] = ['commonvoice'] * len(cv_test)
    wp1_train_sample["origin"] = ['wp1'] * len(wp1_train_sample)
    wp1_test_sample["origin"] = ['wp1'] * len(wp1_test_sample)

    # Merge
    train_data = pd.concat([cv_train, wp1_train_sample])
    test_data = pd.concat([cv_test, wp1_test_sample])


    # Rename ancien tsv
    run(f'mv {train_path} {os.path.join(args.cv, "old_train.tsv")}', shell=True)
    run(f'mv {test_path} {os.path.join(args.cv, "old_test.tsv")}', shell=True)


    # Sauvegarde du tracker tsv
    train_data.to_csv(os.path.join(args.cv, "train.tsv"), sep="\t", index=False)
    test_data.to_csv(os.path.join(args.cv, "test.tsv"), sep="\t", index=False)

    return wp1_test_sample, wp1_train_sample

def generate_dataset(wp1, wp1_files_folder, cv_files_folder):
    """ Copie les fichiers du WP1 dans le dossier clips/ de commonvoice

    :param wp1: Dataframe clean du wp1
    :type wp1: DataFrame
    :param wp1_files_folder: Dossier contenant les fichiers audios du wp1
    :type wp1_files_folder: str
    :param cv_files_folder: Dossier contenant les fichiers audios de commonvoice
    :type cv_files_folder: str
    """

    files_absolute = glob.glob(os.path.join(wp1_files_folder,"*.wav"))

    print("> Renaming")
    to_rename = [f for f in files_absolute if " " in f]
    for f in to_rename:
        os.rename(f, f.replace(" ", "_"))

    # Update des paths
    files_absolute = glob.glob(os.path.join(wp1_files_folder,"*.wav"))
    files_relative = [f.split(os.path.sep)[-1] for f in files_absolute]

    path_list = wp1["path"].apply(lambda x : x.split("/")[-1]).to_list()
    print("> Copy WP1 files")
    for i in tqdm(range(len(files_relative))):
    # for i in range(len(files_relative)):
        if files_relative[i] in path_list :
            run(f'cp -f {files_absolute[i]} {os.path.join(cv_files_folder, "clips/wp1/")}', shell=True)
            # print(f'cp -f {files_absolute[i]} {os.path.join(cv_files_folder, "clips/wp1/")}')
            # input()


def convert_audios(cv):
    """ Conversion des fichiers audios du dossier clips au format
    wave 16 kHz 16 bits mono
    """

    # potentielle amélioration en loadant le tsv subfolders

    print("> Conversion ")
    subfolders = glob.glob(os.path.join(cv, "clips/*"))
    for subfolder in subfolders:
        print(">>", subfolder)
        subfolder_abs = os.path.join(os.path.join(cv, "clips"), subfolder)
        cmd = "for i in " + os.path.join(subfolder_abs, "*.mp3") + "; do ffmpeg -y -i \"$i\" -ar 16000 -ac 1 -acodec pcm_s16le \"${i%.*}.wav\"; done"
        run(cmd, shell=True)
        # print(cmd)

        print("> Suppression des fichiers source")
        # print("rm -f " + os.path.join(subfolder_abs, "*.mp3"))
        run("rm -f " + os.path.join(subfolder_abs, "*.mp3"), shell=True)

    # Mise à jour des noms des fichiers dans les trackers tsv
    tsv = glob.glob(os.path.join(cv, "*.tsv"))
    for t in tsv:
        df = pd.read_csv(t, '\t')
        df["path"] = df["path"].str.replace("mp3", "wav")
        df.to_csv(t, '\t')

def size_to_sec(size) :
    # Formule : bit depth * freq / bits / 8 * sec = size en bytes
    return size / (16 * 16000 / 1000000 / 8)

def calculate_duration(cv):

    # A vérifier pour les path

    tsv = glob.glob(os.path.join(cv, "*.tsv"))
    total_total_size = 0.0
    for t in tsv:
        total_size = 0.0

        df = pd.read_csv(t, '\t')
        df["path"] = df["path"].str.replace("mp3", "wav")
        for index, row in t.iterrows():
            filename = row["path"]
            total_size += os.path.getsize(os.path.join(cv, f"clips/{filename}"))
        total_size_mb = total_size / 1_000_000
        total_total_size += total_size_mb
        print(t)
        print("> Durée :", size_to_sec(total_size_mb) / 3600)

    print("> Durée totale :", size_to_sec(total_total_size) / 3600)


def main(args):

    # Génération des nouveaux trackers
    wp1_test_sample, wp1_train_sample = generate_tsv(args)

    # Conversion des audios au bon format
    # convert_audios(args.cv)

    # Merge "physique" des fichiers audios dans le dossier de commonvoice
    generate_dataset(pd.concat([wp1_test_sample,wp1_train_sample]), args.wp1, args.cv)

    # Calculer la durée en fonction du split
    # calculate_duration(args.cv)


if __name__ == "__main__":    

    parser = argparse.ArgumentParser()

    parser.add_argument("--wp1", default=None, type=str,
                        required=True, help="Path vers le dossier contenant les fichiers audio wp1 ainsi que le tracker csv")
    parser.add_argument("--cv", default=None, type=str,
                        required=True, help="Path vers le dossier commonvoice fr")
    parser.add_argument("--out", default=None, type=str,
                        required=False, help="Dossier pour enregistrer le nouveau .tsv")
    parser.add_argument("--target_duration", default=20, type=int,
                        required=False, help="Durée de chaque split")

    args = parser.parse_args()

    CONTRIB_TO_REMOVE = ["DELAHAYE"]

    main(args)

""" 
    Commonvoice tree
        [lang].tar.gz/
    ├── clips/
    │   ├── *.mp3 files
    |__ dev.tsv
    |__ invalidated.tsv
    |__ other.tsv
    |__ test.tsv
    |__ train.tsv
    |__ validated.tsv
    |__ reported.tsv (as of Corpus 5.0)

    WP1 tree
    ├── *.wav files
    |__ dataset_FR_gen.csv
"""


""" Liste des contributeurs

['ALLAGNAT',
 'MONTAGNANI',
 'DELAHAYE',
 'PAUZE',
 'CELY',
 'PERQUIER',
 'ZANI',
 'SAHAI',
 'BONNIEZ',
 'SIGWALD',
 'GOULARD',
 'SIGWALD ',
 'RAFFANEL',
 'BOUCHE',
 'LEDUC',
 'CLEMENT']

"""