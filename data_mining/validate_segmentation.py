import glob
import argparse
import os
import pandas as pd
from tqdm import tqdm
import librosa
from datasets import Dataset
from subprocess import run

def size_to_sec(size) :
    # Formule : bit depth * freq / bits / 8 * sec = size en bytes
    
    return size / (16 * 16000 / 1000000 / 8)

def array_to_duration(batch):
    audio_array, _ = librosa.load(batch["path"])
    batch["duration"] = librosa.get_duration(audio_array)
    return batch

def get_real_duration(args):

    wav = pd.DataFrame(columns=["path", "duration"])
    wav["path"] =  glob.glob(os.path.join(args.clips, "*.wav"))
    wav["duration"] = [0] * len(wav)
    dataset = Dataset.from_pandas(wav)

    dataset = dataset.map(array_to_duration, batched=True, batch_size=32)

    return sum(dataset["duration"]), len(wav)

# def get_real_duration(args):
#     cmd = 'for x in ' + os.path.join(args.clips, "*.wav") + \
#     '; do ffprobe -i $x -show_entries format=duration -v quiet -of csv="p=0"; done > ' + \
#     os.path.join(args.clips, "duration.txt")

#     print(cmd)
#     run(cmd, shell=True)
    
#     with open(os.path.join(args.clips, "duration.txt"), 'r') as durations:
#         c = durations.read()
    
#     return sum([float(x) for x in c.split("\n") if len(x) > 0])

def get_real_duration(args):

    fnames = glob.glob(os.path.join(args.clips, "*.wav"))
    with open(os.path.join(args.clips, "duration.csv"), 'r+') as durations:
        print('path\tduration', file=durations)
        for fname in tqdm(fnames):
            cmd = 'ffprobe -i '+ fname +' -show_entries format=duration -v quiet -of csv="p=0"'
            output  = run(cmd, shell=True, capture_output=True)
            print(f"{fname}\t{output.stdout[:-2].decode('utf-8')}", file=durations) # -2 pour enlever "\n"

    return os.path.join(args.clips, "duration.csv")

def get_theoretical_duration(args):

    duration_sec = 0
    old_duration = 0
    for i in ["1/*", "2/*", "3/*", "4/*"]:
        segs = glob.glob(os.path.join(args.segment_dir, i))
        for s in tqdm(segs):
            segments = pd.read_csv(s, "\t")

            # Ajout de la durées de segments au dataframe
            segments["len"] = segments["stop"] - segments["start"]

            old_duration += segments["len"].sum()
            # Sélection uniquement des segments voisés trouvés par InaSpeech
            segments = segments[segments["labels"] == "speech"].reset_index()

            duration_sec += segments["len"].sum()

    return duration_sec, old_duration

def main(args):

    if args.segment_dir:
        theoretical_duration, old_duration = get_theoretical_duration(args)
        print("Theoretical duration :", theoretical_duration / 3600, "h")
        print("Before segmentation :", old_duration / 3600, "h")


    if args.clips:
        durations = pd.read_csv(get_real_duration(args), '\t')
        
        # print(duration/3600)
        # print("Real duration :", duration / 3600, "h for", nb_files, "files")

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--clips", default=None, type=str,
                        required=False, help="Chemin de sauvegarde des fichiers audios")
    parser.add_argument("-s", "--segment_dir", default=None, type=str,
                        required=False, help="Chemin de sauvegarde des segments")  


    args = parser.parse_args()
    main(args)
