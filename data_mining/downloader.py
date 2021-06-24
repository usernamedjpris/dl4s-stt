'''
    File name: downloader.py
    Author: Mikael El Ouazzani
    Date created: 03/2021
    Date last modified: 07/06/2021
    Python Version: 3.7
'''

import pandas as pd
import youtube_dl as ytdl
from subprocess import PIPE, run
import glob
import os
import time
import argparse

def out(command):
    """
    Run a bash command
    """
    result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True)
    return result

def print_summary(videos):
    """ 
    Affiche des informations sur le téléchargement en cours
    """
    
    downloaded_videos = len(videos[videos["downloaded"] == 1])
    total_videos = len(videos)

    downloaded_hours = int(videos[videos["downloaded"] == 1]["duration"].sum() / 3600)
    total_duration = int(videos["duration"].sum() / 3600)

    total_elapsed_time = total = videos[videos["downloaded"] == 1]["elapsed_time"].sum() / 60
    elapsed_hours = total_elapsed_time / 60
    target_hours = 10000
    # remaining_time_for_target_h = (elapsed_hours * target_hours) / downloaded_hours - elapsed_hours
    remaining_time_for_current_db = (elapsed_hours * (total_duration - downloaded_hours)) / downloaded_hours

    total_size = videos["size_Mo"].sum() / 1000
    downloaded_size = videos[videos["downloaded"] == 1]["size_Mo"].sum() / 1000

    print("Summary :")
    print(f"> Downloaded : {downloaded_size:.2f} Go / {total_size:.2f} Go  ({downloaded_size / total_size * 100:.2f} %)")
    print(f"> Downloaded videos : {downloaded_videos} / {total_videos}  ({downloaded_videos / total_videos * 100:.2f} %)")
    print(f"> Downloaded hours : {downloaded_hours} / {total_duration}  ({downloaded_hours / total_duration * 100:.2f} %)")
    print(f"> Total elapsed time : {total_elapsed_time // 60:.0f}h{total_elapsed_time % 60:.0f}min" )
    print(f"> ETA : {remaining_time_for_current_db:.0f}h{(remaining_time_for_current_db - int(remaining_time_for_current_db)) * 60:.0f}min")

def download_and_convert_video(url, path):
    """ Télécharge et convertit Youtube video en .wav (pcm_s16le 16kHz mono converter) via youtube-dl et ffmpeg

    :param url: url de la vidéo youtube
    :type url: str
    :param path: Destination de l'audio
    :type path: str
    """

    error = 0
    elapsed_time = 0

    # Fetch meta data
    with ytdl.YoutubeDL({'format': "bestaudio"}) as ydl:

        # Beaucoup d'erreurs potentielles car vidéos peuvent être supprimés par
        # les auteurs, passer en privé etc ...
        try :
            start_time = time.time()
            meta_data = ydl.extract_info(url, download = False)
        except Exception as e:
            error = 1
            print(e)
            bestaudio = None
            elapsed_time = 0
            pass

        else :
            # Récupération du format audio dispo sur la vidéo
            bestaudio = meta_data['ext']

            # Distinction des cas entre webm et m4a
            # Webm : on passe par le wrapper python de youtube-dl (plus simple)
            if bestaudio == "webm":
            

                # Params youtube-dl
                ydl_opts = {
                    'format': 'bestaudio/best',
                    'outtmpl' : f"{path}/%(id)s.%(ext)s",
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'wav',
                    }],
                    'postprocessor_args' : ["-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000"],

                }

                # Download webm, convert to wav and delete video file then rename
                with ytdl.YoutubeDL(ydl_opts) as ydl:
                    try :
                        ydl.download([url])
                    except Exception as e:
                        print(e)
                        error = 1
                        pass
                    else :
                        elapsed_time = time.time() - start_time
                        print("> [youtube-dl] : Download, conversion, removal : OK")

            elif bestaudio == "m4a":
                # Mêmes params mais en passant par l'outil youtube-dl en cli
                pass
                ydl_opts = {
                    'format': 'bestaudio',
                    'outtmpl' : f"{path}/%(id)s.%(ext)s",
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                    }],
                }

                # Download m4a
                with ytdl.YoutubeDL(ydl_opts) as ydl:
                    try :
                        ydl.download([url])
                    except Exception as e:

                        print(e)
                        error = 1
                        print("> ! Erreur de téléchargement")
                        pass
                    else :
                        print("> Download : OK")

                        # Récup fichier plus récent      
                        list_of_files = glob.glob(f"{path}/*m4a")   
                        filename = max(list_of_files, key=os.path.getctime).replace(' ', '\ ')

                        # Convertit les fichiers m4a en wav
                        print("> Converting from m4a to wav ...")

                        # méthode brutale
                        # cmd = "for i in " + path.replace(" ", "\ ") + "/*.m4a; do ffmpeg -y -i \"$i\" -ar 16000 -ac 1 -acodec pcm_s16le \"${i%.*}.wav\"; done"

                        # méthode plus propre
                        cmd = "export i=" + filename + " && ffmpeg -y -i \"$i\" -ar 16000 -ac 1 -acodec pcm_s16le \"${i%.*}.wav\""
                        command = out(cmd)

                        if command.returncode != 0:
                            print("> Conversion error !")
                            error = 2
                        else :
                            print("> Conversion : OK")
                            print(f"> Removing source file {filename}")
                            cmd = f'rm -f {filename}'
                            command = out(cmd)
                            
                            elapsed_time = time.time() - start_time
                    
        
    return error, elapsed_time, bestaudio


def download_videos_from_df(path, bdd):
    """ Lit la BDD de vidéos YT et lance le téléchargement + conversion 

    :param path: Path d'enregistrement des audios
    :type path: str
    :param bdd: Path vers dossier de la BDD (fichiers pkl)
    :type bdd: str
    """

    videos = pd.read_csv(os.path.join(bdd, "videos.csv"))
    to_download = videos[(videos["downloaded"] == 0) & (videos["error"] == 0)]

    k = 0 # Compteur
    while len(to_download) != 0:

        ## Summary 
        # if k%10 == 0 and len(videos[videos["downloaded"] == 1]) > 0:
        #     print_summary(videos)
        print("~~~")

        # Get a random video from database
        index = to_download.sample().index[0]

        # Download and extract wav
        error, elapsed_time, bestaudio = download_and_convert_video(videos.loc[index, "url"], path)
        print("> Error code :", error)
        print(f"> Elapsed time : {elapsed_time:.2f} s.")
        
        # Suivi des erreurs
        if error == 0: # RAS
            videos.loc[index, "converted"] = 1
            videos.loc[index, "downloaded"] = 1
        else :
            videos.loc[index, "error"] = 1
            if error == 2: # dl mais pas convert (m4s)
                videos.loc[index, "downloaded"] = 1

        videos.loc[index, "elapsed_time"] = elapsed_time
        videos.loc[index, "ext"] = bestaudio

        # Update des vidéos restantes à traiter
        to_download = videos[(videos["downloaded"] == 0) & (videos["error"] == 0)]

        # Mise à jour de la bdd
        videos.to_csv(os.path.join(bdd, "videos.csv"))
        print("------" * 10)    
            

def main(args):

    download_videos_from_df(args.out, args.bdd)

    
if __name__ == "__main__":    
    parser = argparse.ArgumentParser()

    parser.add_argument("--out", default=None, type=str,
                        required=True, help="Chemin de sauvegarde des fichiers audios")    
    parser.add_argument("--bdd", default=None, type=str,
                        required=True, help="Chemin vers le dossier contenant les BDD (csv)")    
    
    args = parser.parse_args()
    main(args)