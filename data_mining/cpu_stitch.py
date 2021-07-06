import glob
import argparse
from subprocess import run
import os
import pandas as pd

def size_to_sec(size) :
    # Formule : bit depth * freq / bits / 8 * sec = size en bytes
    
    return size / (16 * 16000 / 1000000 / 8)

def stitch(filename, segments, output_dir):
    """ Harmonisation des durées des segments pour un fichier audio via ffmpeg

    :param filename: Path vers le fichier source non traité
    :type filename: str
    :param segments: Path vers le csv contenant les infos de segmentation du fichier filename
    :type segments: str
    :param output_dir: Dossier d'enregistrement des segments audios
    :type output_dir: str
    """

    yt_id = filename.split('.')[0].split("/")[-1]


    trg_len = 30
    count = 0
    to_stitch = []
    cumul_len = 0
    for index, row in segments.iterrows():
        len_seg = row['len']
        if cumul_len + len_seg > trg_len:

            # Stitching
            filename_output = os.path.join(output_dir, f"{yt_id}_{count}.wav")
            aselect = "\'between(t,4,6.5)+between(t,17,26)+between(t,74,91)\'"
            aselect= "\'" + "+".join([f"between(t,{a},{b})" for a,b in zip(segments.iloc[to_stitch]["start"], segments.iloc[to_stitch]["stop"])]) + "\'"
            cmd = "ffmpeg -n -i "+ filename + " -af \"aselect=" + aselect + ", asetpts=N/SR/TB\" " + filename_output
            print(cmd)
            run(cmd, shell=True)

            # Reset
            to_stitch = []
            cumul_len = 0
            count +=1
        
        cumul_len += len_seg

        to_stitch.append(index)

def get_wav(args):

    # Fichiers wav "physiques"
    real_wav = glob.glob(os.path.join(args.clips, "*.wav"))

    # Fichiers wav "théoriques" d'après le tracker
    wav = pd.read_csv(args.part)['path'].apply(lambda x : os.path.join(args.clips, x)).to_list()

    # Fichiers déjà traités (segmentés)
    segmented = [f.split("/")[-1][:-4] for f in glob.glob(os.path.join(args.segment_dir, "*"))]

    # Filtrage pour avoir les wav restants à traiter
    wav = [w for w in real_wav if w in wav and w.split("/")[-1][:-4] not in segmented]

    return wav

def main(args):
    # Uniformisation : collage des segments en fichiers de 30 sec en moyenne
    # Traitement séquentiel fichier par fichier

    # ouvir un csv de split OK
    # get fichiers wav du dataset OK
    # filtrer ceux qui sont dans le split et le dataset OK
    # pour chaque wav qui passe le filtre Ok
    # trouver le segment correspondant dans segment_dir
    # stitch les segments dans le output_dir

    wav = get_wav(args)

    for filename in wav:

        # Import du résultat de la segmentation à partir du nom du fichier wav
        csv = filename.split("/")[-1][:-3] + 'csv'
        segments = pd.read_csv(os.path.join(args.segment_dir, csv), '\t')

        # Sélection uniquement des segments voisés trouvés par InaSpeech
        segments = segments[segments["labels"] == "speech"].reset_index()

        # Ajout de la durées de segments au dataframe
        segments["len"] = segments["stop"] - segments["start"]

        # Harmonisation
        stitch(filename, segments, args.segment_dir[:-1])

        # Suppression du fichier source
        # run(f'rm -f {filename}', shell=True)
    
    # Suppression des fichiers csv
    # run(f'rm -f {os.path.join(output_dir, "*.csv")}', shell=True)

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()

    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--clips", default=None, type=str,
                        required=True, help="Chemin de sauvegarde des fichiers audios")
    parser.add_argument("-p", "--part", default=None, type=str,
                        required=True, help="Fichier csv de la partie traité (1-4)")
    parser.add_argument("-s", "--segment_dir", default=None, type=str,
                        required=True, help="Chemin de sauvegarde des segments")  
    # parser.add_argument("-o", "--output_dir", default=None, type=str,
                        # required=True, help="Chemin de sauvegarde des segments")


    
    args = parser.parse_args()
    main(args)
