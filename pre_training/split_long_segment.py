import argparse
from multiprocessing import Process
import os
import pandas as pd
from tqdm import tqdm
from subprocess import run
import numpy as np

def combine_durations(args):
    tab = []
    for i in range(16):
        df = pd.read_csv(os.path.join(args.clips, f"duration_{i}.csv"), "\t")
        for i in range(len(df)):
            if not str(df["duration"][i])[0].isnumeric():
                df = df.drop(i)
        tab.append(df)
    return pd.concat(tab)

def get_stat_on_durations(args, df):

    lost_data = df["duration"].apply(lambda x : x % args.split_duration if (x%args.split_duration) < args.lower_bound else 0).sum() / 3600
    print(f"Lost audio : {lost_data:.2f}h")
    

def get_segments_duration(duration):
    d = args.split_duration # Taille cible d'un segment audio

    m = duration % d
    n = duration // d if m < args.lower_bound else duration // d + 1
        
    start = [i * d for i in range(n)]
    end = [(x+1) * d for x in range(n) if (x+1) * d <= duration]

    if m >= args.lower_bound:
        end += [duration]

    return start, end

def split_audio(args, path, start, end):

    for i in range(len(start)):

        # file name
        filename_output = os.path.join(args.output_dir, path.split("/")[-1].split(".")[0] + f'_{i}.wav')

        # Cut
        aselect = "\'between(t,4,6.5)+between(t,17,26)+between(t,74,91)\'"
        aselect= "\'" + f"between(t,{start[i]},{end[i]})" + "\'"
        cmd = "ffmpeg -hide_banner -loglevel error -n -i "+ path + " -af \"aselect=" + aselect + ", asetpts=N/SR/TB\" " + filename_output
        # print(cmd)
        run(cmd, shell=True)

def split_from_filtered_df(args, df, i):

    for _, row in tqdm(df.iterrows(), desc=str(i)):
        path = row["path"]
        duration = row["duration"]
        start, end = get_segments_duration(duration)
        split_audio(args, path, start, end)

def main(args):
    concat_df = combine_durations(args)
    get_stat_on_durations(args, concat_df)
    filtered_df = concat_df[concat_df["duration"] > args.threshold]

    dfs = np.array_split(filtered_df, args.process)
    processes = []
    for i in range(args.process):
        df = dfs[i]
        p = Process(target=split_from_filtered_df, args=(args, df, i))
        processes.append(p)
        p.start()



if __name__ == "__main__":    
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--clips", default=None, type=str,
                        required=True, help="Chemin de sauvegarde des fichiers audios")
    parser.add_argument("-o", "--output_dir", default=None, type=str,
                        required=True, help="Dossier de sortie des audios")  

    parser.add_argument("-s", "--segment_dir", default=None, type=str,
                        required=False, help="Chemin de sauvegarde des segments")  
    parser.add_argument("-p", "--process", default=None, type=int,
                        required=False, help="nb de process")  
    parser.add_argument("-d", "--split_duration", default=30, type=int,
                        required=False, help="Durée des splits complets")  
    parser.add_argument("-t", "--threshold", default=45, type=int,
                        required=False, help="Seuil de durée")  
    parser.add_argument("-l", "--lower_bound", default=20, type=int,
                        required=False, help="Seuil de durée minimale autorisée")  


    args = parser.parse_args()
    main(args)
