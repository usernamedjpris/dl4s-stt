'''
    File name: segmenter.py
    Author: Mikael El Ouazzani
    Date created: 03/2021
    Date last modified: 07/06/2021
    Python Version: 3.7
'''

import pandas as pd
import glob
import os
import argparse
from subprocess import run
from inaSpeechSegmenter import Segmenter, seg2csv


def segmentation_inaspeech(input_files, output_dir, batch_size):
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
    seg = Segmenter("smn", False, 'ffmpeg', batch_size=batch_size)
    seg.batch_process(input_files, output_files, verbose=True)



def main(args):
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
    wav = glob.glob(os.path.join(args.clips, "*.wav"))
    # for index, row in tracker.iterrows():
    #     yt_id = row["url"][-11:]
    #     filename = os.path.join(input_dir, yt_id + ".wav")
    #     wav.append(filename)
        
    segmentation_inaspeech(wav, args.output_dir, args.batch_size)
    

if __name__ == "__main__":    

    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--clips", default=None, type=str,
                        required=True, help="Chemin de sauvegarde des fichiers audios")    
    parser.add_argument("-o", "--output_dir", default=None, type=str,
                        required=True, help="Chemin de sauvegarde des segments")
    parser.add_argument("-b", "--batch_size", default=None, type=int,
                        required=True, help="Chemin de sauvegarde des segments")
    
    args = parser.parse_args()
    main(args)
