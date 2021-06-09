'''
    File name: populate_bdd.py
    Author: Mikael El Ouazzani
    Date created: 03/2021
    Date last modified: 04/06/2021
    Python Version: 3.7
'''

import pandas as pd
import pytube as pt
import argparse
import os

def update_playlists_list(source, playlists, bdd):
    """ Lit un fichier source contenant des urls de playlists
    YouTube et fetch les metadata via l'API YT pour remplir
    la BDD de playlists

    :param source: chemin vers un .txt d'urls YT de playlists
    :type source: str
    :param playlists: BDD des metadata de playlists
    :type playlists: DataFrame
    :param bdd: Path vers dossier de la BDD
    :type bdd: str
    """

    # Read file
    f = open(source, "r")
    playlists_txt = f.read()
    playlists_txt_urls = playlists_txt.split("\n")

    for url in playlists_txt_urls:
        if url not in list(playlists.url) and url != "":

            # Fetch data from YouTube API
            print(url)
            playlist = pt.Playlist(url)
            
            # Créé une nouvelle entrée dans la BDD
            playlist_info = {"title" : playlist.title, 
                              "count":len(playlist.video_urls), 
                              "total_duration" : 0, 
                              "added_to_list" : 0,
                              "downloaded" : 0,
                              "stand_by" : 0,
                              "url" : url}
            
            print("> Adding", playlist_info["title"])
            playlists = playlists.append(playlist_info, ignore_index=True)
            playlists.to_pickle(os.path.join(bdd, "playlists.pkl"))

    # Save
    playlists.to_pickle(os.path.join(bdd, "playlists.pkl"))

def update_video_list(playlists, videos, bdd, min_duration):
    """ Lit la BDD de playlists et fetch les metadata de chaque vidéos
    de ces playlists pour remplit la BDD de vidéos. Save auto toutes les
    20 vidéos. !! Exceptions peuvent être à modifier en fonction de 
    l'évolution de l'API YouTube !!

    :param playlists: BDD des metadata de playlists
    :type playlists: DataFrame
    :param videos: BDD des metadata de videos
    :type videos: DataFrame
    :param bdd: Path vers dossier de la BDD
    :type bdd: str
    :param min_duration: Durée minimale acceptée pour un audio dans la BDD
    :type min_duration: int
    """
    # Count for auto save
    k = 0
    for index, row in playlists.iterrows():
        
        if row["added_to_list"] == 0 and row["stand_by"] == 0:

            print("> Playlist :", row["title"])

            # Fetch de la liste de urls de vidéos pour un playlist
            video_urls = pt.Playlist(row["url"]).video_urls

            for url in video_urls:

                # Vérification doublon
                if url not in list(videos.url):

                    try: 
                        video = pt.YouTube(url)
                    except pt.exceptions.VideoRegionBlocked :
                        print("#####" * 10)
                        print("> Restriction région")
                        print("#####" * 10)
                        continue
                    except pt.exceptions.VideoPrivate :
                        print("#####" * 10)
                        print("> Vidéo privée")
                        print("#####" * 10)
                        continue
                    except pt.exceptions.VideoUnavailable :
                        print("#####" * 10)
                        print("> Vidéo non disponible")
                        print("#####" * 10)
                        continue
                    except Exception as e:
                        print(e)
                        continue

                    try :
                        print("Durée :", video.length / 60)
                    except Exception as e:
                        print(e)
                        continue
                    
                    if video.length < min_duration * 60:
                        print("---!!---")
                        print("Durée :", video.length / 60)
                        print("Pas assez longue !")
                        print("---!!---")

                        continue
                    else : 
                        video_info = {"channel" : video.author, 
                                    "title" : video.title, 
                                    "duration" : video.length, 
                                    "url" : url, 
                                    "playlist_url" : row["url"], 
                                    "error" : 0,
                                    "downloaded" : 0,
                                    "converted" : 0,
                                    "m4a_webm" : 1,
                                    "elapsed_time" : 0,
                                    "fiable" : 1, # Par défaut on considère que la source est fiable
                                    "size_Mo" : 0.032 * video.length}
                                    # Taille calculé en utilisant la valeur de conversion
                                    # pour un audio .wav mono 16 kHz 16 bits
                                    # même si la conversion n'a pas encore été faite
                        
                        print("> Adding :", video_info)
                        videos = videos.append(video_info, ignore_index=True)
                        
                        print("------" * 10)
                        print(">> Saving videos")
                        videos.to_pickle(os.path.join(bdd, "videos.pkl"))
                        print("------" * 10)

            # Update bdd
            playlists.loc[index,"added_to_list"] = 1
            playlists.loc[index,"total_duration"] = int(sum(videos.loc[videos["playlist_url"] == row["url"]]['duration']) / 3600)

            # Save 
            print("------" * 10)
            print(">> Saving playlists")
            playlists.to_pickle(os.path.join(bdd, "playlists.pkl"))
            print("------" * 10)

    videos.to_pickle(os.path.join(bdd, "videos.pkl"))

    
def main(args):

    # Initialiation des bdd
    if os.path.exists(os.path.join(args.bdd, "playlists.pkl")):
        playlists = pd.read_pickle(os.path.join(args.bdd, "playlists.pkl"))
    else :
        playlists = pd.DataFrame(columns=['url'])
    
    if os.path.exists(os.path.join(args.bdd, "videos.pkl")):
        videos = pd.read_pickle(os.path.join(args.bdd, "videos.pkl"))
    else :
        videos = pd.DataFrame(columns=['url'])
  
    min_duration = 10
    update_playlists_list(args.source, playlists, args.bdd)
    playlists = pd.read_pickle(os.path.join(args.bdd, "playlists.pkl"))
    update_video_list(playlists, videos, args.bdd, min_duration)
    
if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=None, type=str,
                        required=True, help="Fichier texte contenant les urls des playlists ou chaînes YT")
    parser.add_argument("--bdd", default=None, type=str,
                        required=True, help="Chemin vers le dossier contenant les BDD (pickle)")    
    
    args = parser.parse_args()
    main(args)