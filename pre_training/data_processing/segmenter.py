'''
    File name: segmenter.py
    Author: Mikael El Ouazzani
    Date created: 03/2021
    Date last modified: 07/06/2021
    Python Version: 3.7
'''

from vad import *
import wave
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import glob
import os
import argparse
from subprocess import run
from inaSpeechSegmenter import Segmenter, seg2csv


def size_to_sec(size) :
    # Formule : bit depth * freq / bits / 8 * sec = size en bytes
    
    return size / (16 * 16000 / 1000000 / 8)

def silence_removal_webrtc(filename, output_dir, aggressivity, target_duration):
    """ Suppression des silences d'un audio et enregistrement en segments de
    durées moyenne égale à target_duration secondes.

    :param filename: Path vers le fichier audio
    :type filename: str
    :param output_dir: Dossier d'enregistrement des audios ségmentés et harmonisés
    :type output_dir: str
    :param aggressivity: Agressivité de la ségmentation (3 stricte - 1 permissive)
    :type aggressivity: int
    :param target_duration: Durée moyenne voulue pour les segments voisés
    :type target_duration: int
    """
    audio, sample_rate = read_wave(filename)
    yt_id = filename.split('.')[0].split("/")[-1]

    # Initialisation 
    vad = webrtcvad.Vad(int(aggressivity))
    frames = frame_generator(30, audio, sample_rate)
    frames = list(frames)
    segments = vad_collector(sample_rate, 30, 300, vad, frames)
    to_stitch = b''
    length_accumulated = 0.0
    count = 0
    
    for i, segment in enumerate(segments):

        # Calcul de la durée du segment 
        # On utilise la fn len car les bits sont encodés dans une string
        length_sec = size_to_sec(len(segment) / 1000000)

        if count == 0 or count == 1: # début = jingle souvent -> on ignore
            count +=1

        else :
            if length_accumulated + length_sec >= target_duration and length_accumulated > 20:
                path = os.path.join(output_dir, f'{yt_id}_{count:04d}.wav')
                print(' Writing %s' % (path,))
                if len(to_stitch) == 0 : to_stitch = segment
                write_wave(path, to_stitch, sample_rate)
                length_accumulated = 0
                to_stitch = b''
                count +=1
            
            to_stitch += segment
            length_accumulated += length_sec

    # On ne traite pas le "reste" du tableau to_stitch car il correspond au dernier segments de l'audio
    # et on fait l'hypothèse que ce segments est souvent un jingle, qu'on peut donc ignorer comme le premier
    # segment

def segmentation_inaspeech(input_files, output_dir):
    """ Segmentation d'une liste de fichier par le modèle inaSpeechSegmenter (CNN)
    et enregistrement des résultats dans des fichiers csv du même nom que les 
    fichiers traités.

    :param input_files: Liste de nom de fichiers à traiter
    :type input_files: str
    :param output_dir: Dossier d'enregistrement des csv contenant l'info de segmentation
    :type output_dir: str
    """
    base = [os.path.splitext(os.path.basename(e))[0] for e in input_files]
    output_files = [os.path.join(output_dir, e + '.csv') for e in base]
    seg = Segmenter("smn", False, 'ffmpeg')
    seg.batch_process(input_files, output_files, verbose=True)

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
        if cumul_len + len_seg > trg_len and cumul_len > 0.6 * trg_len:

            # Stitching
            filename_output = os.path.join(output_dir, f"{yt_id}_{count}.wav")
            aselect = "\'between(t,4,6.5)+between(t,17,26)+between(t,74,91)\'"
            aselect= "\'" + "+".join([f"between(t,{a},{b})" for a,b in zip(segments.iloc[to_stitch]["start"], segments.iloc[to_stitch]["stop"])]) + "\'"
            cmd = "ffmpeg -i "+ filename + " -af \"aselect=" + aselect + ", asetpts=N/SR/TB\" " + filename_output
            print(cmd)
            run(cmd, shell=True)

            # Reset
            to_stitch = []
            cumul_len = 0
            count +=1
        
        cumul_len += len_seg

        to_stitch.append(index)

    # if to_stitch:
    #     ffmpeg_concat(to_stitch, count)

def main_inaspeech(input_dir, output_dir, tracker):
    """ Segmentation et harmonisation des audios listés dans la bdd
    et présents dans le dossier input_dir. Enregistrement dans output_dir
    Version : audios non fiables

    :param input_dir: Dossier contenant les audios
    :type input_dir: str
    :param output_dir: Dossier d'enregistrement des audios ségmentés et harmonisés
    :type output_dir: str
    :param videos: Path du DataFrame bdd des videos
    :type videos: str
    """

    # Segmentation et enregistrement des résulats dans un
    # csv par fichier traité
    # Traitement par batch automatique avec inaSpeechSegmenter
    wav = []
    for index, row in tracker.iterrows():
        yt_id = row["url"][-11:]
        filename = os.path.join(input_dir, yt_id + ".wav")
        wav.append(filename)
    segmentation_inaspeech(wav, output_dir)

    # Uniformisation : collage des segments en fichiers de 30 sec en moyenne
    # Traitement séquentiel fichier par fichier
    for filename in wav:

        # Import du résultat de la segmentation
        csv = filename.split("/")[-1][:-3] + 'csv'
        segments = pd.read_csv(os.path.join(output_dir, csv), '\t')

        # Sélection uniquement des segments voisés
        segments = segments[segments["labels"] == "speech"].reset_index()

        # Calcul de la durées de segments
        segments["len"] = segments["stop"] - segments["start"]

        # Harmonisation
        stitch(filename, segments, output_dir)

        # Suppression du fichier source
        run(f'rm -f {filename}', shell=True)
    
    # Suppression des fichiers csv
    run(f'rm -f {os.path.join(output_dir, "*.csv")}', shell=True)
    


def main_webrtcvad(input_dir, output_dir, videos):
    """ Segmentation et harmonisation des audios listés dans le tracker
    et présents dans le dossier input_dir. Enregistrement dans output_dir
    Version : Audios fiable

    :param input_dir: Dossier contenant les audios
    :type input_dir: str
    :param output_dir: Dossier d'enregistrement des audios ségmentés et harmonisés
    :type output_dir: str
    :param videos: Path du DataFrame bdd des videos
    :type videos: str
    """

    for index, row in videos.iterrows():
        yt_id = row["url"][-11:]
        filename = os.path.join(input_dir, yt_id + ".wav")

        # Segmentation et harmonisation
        silence_removal_webrtc(filename, output_dir, 3, 30)

        # Suppression du fichier source
        run(f'rm -f {filename}', shell=True)



def main(args):

    # Import bdd
    videos = pd.read_pickle(os.path.join(args.bdd, "videos.pkl"))

    # Tri des videos bien téléchargées et converties sans erreurs
    videos_ok = videos[(videos["downloaded"] == 1) & (videos["error"] == 0)]

    # Séparation du dataset en seux partitions
    videos_fiable = videos_ok[videos_ok["fiable"] == 1]
    videos_non_fiable = videos_ok[videos_ok["fiable"] == 0]
    print(videos_fiable)
    input()
    print(videos_non_fiable)
    input()
    # Traitement données fiables (rapide)
    main_webrtcvad(args.clips, args.out, videos_fiable)

    # Traitement données non fiables (lent + GPU)
    main_inaspeech(args.clips, args.out, videos_non_fiable)




if __name__ == "__main__":    
    parser = argparse.ArgumentParser()

    parser.add_argument("--clips", default=None, type=str,
                        required=True, help="Chemin de sauvegarde des fichiers audios")    
    parser.add_argument("--out", default=None, type=str,
                        required=True, help="Chemin de sauvegarde des segments")
    parser.add_argument("--bdd", default=None, type=str,
                        required=True, help="Chemin vers le dossier bdd") 

    
    args = parser.parse_args()
    main(args)
