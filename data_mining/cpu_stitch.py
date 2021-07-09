import glob
import argparse
from subprocess import run
import os
import pandas as pd
from tqdm import tqdm
from multiprocessing import Process


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
    for index, row in tqdm(segments.iterrows()):
        len_seg = row['len']
        cumul_len += len_seg
        to_stitch.append(index)

        if cumul_len + len_seg > trg_len:

            # file name
            filename_output = os.path.join(output_dir, f"{yt_id}_{count}.wav")

            # Stitching
            aselect = "\'between(t,4,6.5)+between(t,17,26)+between(t,74,91)\'"
            aselect= "\'" + "+".join([f"between(t,{a},{b})" for a,b in zip(segments.iloc[to_stitch]["start"], segments.iloc[to_stitch]["stop"])]) + "\'"
            cmd = "ffmpeg -hide_banner -loglevel error -n -i "+ filename + " -af \"aselect=" + aselect + ", asetpts=N/SR/TB\" " + filename_output
            # print(cmd)
            run(cmd, shell=True)

            # Reset
            to_stitch = []
            cumul_len = 0
            count +=1
        

def process_stitch(args, segments_list):

    for s in tqdm(segments_list):
        segments = pd.read_csv(s, '\t')

        # Sélection uniquement des segments voisés trouvés par InaSpeech
        segments = segments[segments["labels"] == "speech"].reset_index()

        # Ajout de la durées de segments au dataframe
        segments["len"] = segments["stop"] - segments["start"]

        # Harmonisation
        # output_dir = "/".join(args.segment_dir.split("/")[:-1])
        output_dir = os.path.join(args.segment_dir, "clips")
        filename = os.path.join(args.clips, s.split("/")[-1].split('.')[0] + ".wav")
        stitch(filename, segments, output_dir)


def get_wav(args):

    # Fichiers wav "physiques"
    real_wav = glob.glob(os.path.join(args.clips, "*.wav"))

    # Fichiers wav "théoriques" d'après le tracker
    wav = pd.read_csv(args.part)['path'].apply(lambda x : os.path.join(args.clips, x)).to_list()

    # Fichiers déjà segmentés
    segmented = [f.split("/")[-1][:-4] for f in glob.glob(os.path.join(args.segment_dir, "*"))]

    # Fichiers déjàs stitchés
    output_dir = "/".join(args.segment_dir.split("/")[:-1])
    stitched_wav = glob.glob(os.path.join(output_dir, "clips/*"))
    stitched = [f.split("/")[-1].split("_")[0] for f in stitched_wav]

    # Filtrage pour avoir les wav restants à traiter (fichiers segmentés et pas stitchés)
    wav = [w for w in real_wav if w in wav and w.split("/")[-1][:-4] in segmented and w.split("/")[-1][:-4] not in stitched]

    return wav


def main_multi(args):

    # mettre à jour get wav

    segments = []
    for i in ["1/*", "2/*", "3/*", "4/*"]:
        segments += glob.glob(os.path.join(args.segment_dir, i))

    print(len(segments))
    n = len(segments)//args.process
    processes = []
    for i in range(args.process):
        split = segments[i*n:(i+1)*n]
        p = Process(target=process_stitch, args=(args, split))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()


# def main(args):

#     wav = get_wav(args)

#     for filename in wav:

#         # Import du résultat de la segmentation à partir du nom du fichier wav
#         csv = filename.split("/")[-1][:-3] + 'csv'
#         segments = pd.read_csv(os.path.join(args.segment_dir, csv), '\t')
#         print(segments)

#         # Sélection uniquement des segments voisés trouvés par InaSpeech
#         segments = segments[segments["labels"] == "speech"].reset_index()

#         # Ajout de la durées de segments au dataframe
#         segments["len"] = segments["stop"] - segments["start"]

#         # Harmonisation
#         output_dir = "/".join(args.segment_dir.split("/")[:-1])
#         stitch(filename, segments, os.path.join(output_dir, "clips"))

#         # Suppression du fichier source
#         # run(f'rm -f {filename}', shell=True)
    
#     # Suppression des fichiers csv
#     # run(f'rm -f {os.path.join(output_dir, "*.csv")}', shell=True)

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--clips", default=None, type=str,
                        required=True, help="Chemin de sauvegarde des fichiers audios non segmenté")
    # parser.add_argument("-p", "--part", default=None, type=str,
    #                     required=True, help="Fichier csv de la partie traité (1-4)")
    parser.add_argument("-s", "--segment_dir", default=None, type=str,
                        required=True, help="Chemin de sauvegarde des fichiers audios segmenté")
    parser.add_argument("-p", "--process", default=None, type=int,
                        required=True, help="Nombre de process")  
    # parser.add_argument("-o", "--output_dir", default=None, type=str,
                        # required=True, help="Chemin de sauvegarde des segments")


    
    args = parser.parse_args()
    main_multi(args)
