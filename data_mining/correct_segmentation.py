import glob
import argparse
from multiprocessing import Process
import os
import pandas as pd
from tqdm import tqdm
from subprocess import run

def combine_durations(args):
    splits = glob.glob(os.join(args.clips, "duration*"))
    df = []
    for s in splits:
        df.append(pd.read_csv(s, "\t"))

    return pd.concat(df)

def get_stat_on_durations(df):
    pass

def split_audio(args, path, duration):
    pass

def main(args):
    concat_df = combine_durations(args)
    get_stat_on_durations(concat_df)
    filtered_df = concat_df[concat_df["duration"] > 50]
    for index, row in filtered_df.iterrows():
        path = row["path"]
        duration = row["duration"]
        split_audio(args, path, duration)


if __name__ == "__main__":    
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--clips", default=None, type=str,
                        required=False, help="Chemin de sauvegarde des fichiers audios")
    parser.add_argument("-s", "--segment_dir", default=None, type=str,
                        required=False, help="Chemin de sauvegarde des segments")  
    parser.add_argument("-p", "--process", default=None, type=int,
                        required=False, help="nb de process")  
    parser.add_argument("-o", "--output_dir", default=None, type=int,
                        required=False, help="Dossier de sortie des audios")  


    args = parser.parse_args()
    main(args)
