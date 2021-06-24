import re
import os
import glob
import json
import soundfile as sf
import argparse
import torchaudio 
import librosa
import numpy as np

# Huggingface
from datasets import load_dataset
from transformers import Wav2Vec2Processor, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor,


def map_to_array(batch):
    speech, _ = sf.read(batch["file"])
    batch["speech"] = speech
    return batch

def import_data():
    
    format_csv_tracker(TRAIN_CSV_RAW, TRAIN_PATH, TRAIN_CSV)
    format_csv_tracker(VALID_CSV_RAW, VALID_PATH, VALID_CSV)

    data = load_dataset('csv', data_files={'train': TRAIN_CSV,'test': VALID_CSV})
    print(data)
    return data

def format_csv_tracker(source, source_path, output):
    df = pd.read_csv(source)
    df["file"] = df["file"].apply(lambda x : os.path.join(source_path, x))
    df["text"] = df["transcription"]
    # df["text"] = df["transcription"].apply(lambda x : x.upper())
    df = df.drop("transcription", axis=1)
    df.to_csv(output, index=False)

def extract_all_chars(batch):
    all_text = " ".join(batch["text"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}

def gen_vocab(data):
    vocabs = data.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, \
        remove_columns=data.column_names["train"])
    vocab_list = list(set(vocabs["train"]["vocab"][0]) | set(vocabs["test"]["vocab"][0]))
    vocab_dict = {v: k for k, v in enumerate(vocab_list)}
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    with open(f'results_hg/{MODEL}/{LABEL}/vocab.json', 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file)

def speech_file_to_array_fn(batch):
    chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'
    speech_array, sampling_rate = sf.read(batch["file"])
    batch["speech"] = speech_array
    batch["sampling_rate"] = sampling_rate
    # batch["target_text"] = batch["text"]
    batch["target_text"] = re.sub(chars_to_ignore_regex, '', batch["text"]).lower().replace("’", "'")

    return batch

def prepare_dataset(batch):
    batch["input_values"] = processor(batch["speech"], \
        sampling_rate=batch["sampling_rate"][0]).input_values
    
    with processor.as_target_processor():
        batch["labels"] = processor(batch["target_text"]).input_ids
    return batch



def data_preparation():

    data = import_data()
    
    global processor

    if glob.glob(f"results_hg/{MODEL}/{LABEL}/processor/*"):
        print(">> From pretrained processor ")
        processor = Wav2Vec2Processor.from_pretrained(f"results_hg/{MODEL}/{LABEL}/processor")
    else :
        print(">> Creating processor ")

        gen_vocab(data)
        tokenizer = Wav2Vec2CTCTokenizer(f"results_hg/{MODEL}/{LABEL}/vocab.json", unk_token="[UNK]", \
            pad_token="[PAD]", word_delimiter_token="|")
        feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, \
            sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
        processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
        processor.save_pretrained(f'results_hg/{MODEL}/{LABEL}/processor/')

    dataset = data.map(speech_file_to_array_fn, \
         remove_columns=data.column_names["train"], num_proc=4)

    dataset_prepared = dataset.map(prepare_dataset, \
        remove_columns=dataset.column_names["train"], batch_size=8, num_proc=4, batched=True)


    return processor, dataset_prepared

def remove_special_characters(batch):
    chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�]'

    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower() + " "
    return batch

def speech_file_to_array_fn_v2(batch):
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    batch["speech"] = speech_array[0].numpy()
    batch["sampling_rate"] = sampling_rate
    batch["target_text"] = batch["sentence"]
    return batch


def resample(batch):
    batch["speech"] = librosa.resample(np.asarray(batch["speech"]), 48_000, 16_000)
    batch["sampling_rate"] = 16_000
    return batch

def data_preparation_v2():
    common_voice_train = load_dataset("common_voice", "tr", split="train+validation")
    common_voice_test = load_dataset("common_voice", "tr", split="test")
    common_voice_train = common_voice_train.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])
    common_voice_test = common_voice_test.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])
    common_voice_train = common_voice_train.map(remove_special_characters)
    common_voice_test = common_voice_test.map(remove_special_characters)

    # Vocab
    vocab_train = common_voice_train.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=common_voice_train.column_names)
    vocab_test = common_voice_test.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=common_voice_test.column_names)
    vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))
    vocab_dict = {v: k for k, v in enumerate(vocab_list)}
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    with open(f'results_hg/{MODEL}/{LABEL}/vocab.json', 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file)

    global processor

    print(">> Creating processor ")

    tokenizer = Wav2Vec2CTCTokenizer(f"results_hg/{MODEL}/{LABEL}/vocab.json", unk_token="[UNK]", \
        pad_token="[PAD]", word_delimiter_token="|")
    
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, \
        sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
    
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    processor.save_pretrained(f'results_hg/{MODEL}/{LABEL}/processor/')

    common_voice_train = common_voice_train.map(speech_file_to_array_fn_v2, remove_columns=common_voice_train.column_names)
    common_voice_test = common_voice_test.map(speech_file_to_array_fn_v2, remove_columns=common_voice_test.column_names)
    
    # inutile maintenant car déjà au bon format
    # common_voice_train = common_voice_train.map(resample, num_proc=4)
    # common_voice_test = common_voice_test.map(resample, num_proc=4)

    common_voice_train = common_voice_train.map(prepare_dataset, 
                                            remove_columns=common_voice_train.column_names, 
                                            batch_size=8,
                                            num_proc=4, 
                                            batched=True)
    common_voice_test = common_voice_test.map(prepare_dataset, remove_columns=common_voice_test.column_names, batch_size=8, num_proc=4, batched=True)

    return processor, common_voice_train, common_voice_test

def main() :
    processor, dataset_prepared, data_collator = data_preparation()

    dataset_name = "dataset"
    dataset_prepared.save_to_disk(os.path.join(args.output_dir), dataset_name)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-o", "--output-dir", default=None, type=str,
                        required=True, help="Train data tracker csv")


    args = parser.parse_args()

    TRAIN_CSV_RAW = args.train
    VALID_CSV_RAW = args.valid  
    MODEL = args.model #Facultatif : sert à ranger les modèles dans les bons dossiers

    TRAIN_PATH = TRAIN_CSV_RAW.split("dataset")[0]
    VALID_PATH = VALID_CSV_RAW.split("dataset")[0]

    TRAIN_CSV = os.path.join(TRAIN_PATH, "train_hg.csv")
    VALID_CSV = os.path.join(VALID_PATH, "valid_hg.csv")

    LABEL = TRAIN_PATH.split('_')[-1][:-1]

    main(args)