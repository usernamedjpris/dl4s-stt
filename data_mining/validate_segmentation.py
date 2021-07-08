import glob
import argparse
import os
import pandas as pd
from tqdm import tqdm
import librosa
from datasets import Dataset

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


def get_theoretical_duration(args):

    duration_sec = 0
    for i in ["1/*", "2/*", "3/*", "4/*"]:
        segs = glob.glob(os.path.join(args.segment_dir, i))
        for s in tqdm(segs):
            segments = pd.read_csv(s, "\t")
            # Sélection uniquement des segments voisés trouvés par InaSpeech
            segments = segments[segments["labels"] == "speech"].reset_index()

            # Ajout de la durées de segments au dataframe
            segments["len"] = segments["stop"] - segments["start"]

            duration_sec += segments["len"].sum()

    return duration_sec

def main(args):

    if args.segment_dir:
        theoretical_duration = get_theoretical_duration(args)
        print("Theoretical duration :", theoretical_duration / 3600, "h")

    if args.clips:
        duration, nb_files = get_real_duration(args)

        print("Real duration :", duration / 3600, "h for", nb_files, "files")

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--clips", default=None, type=str,
                        required=False, help="Chemin de sauvegarde des fichiers audios")
    parser.add_argument("-s", "--segment_dir", default=None, type=str,
                        required=False, help="Chemin de sauvegarde des segments")  


    args = parser.parse_args()
    main(args)