import re
import os
import glob
import json
import soundfile as sf
import argparse
import torchaudio 
import numpy as np
import pandas as pd
# Huggingface
from datasets import Dataset
from transformers import Wav2Vec2Processor

def import_dataset(split):
    df = pd.read_csv(split, "\t")
    dataset = Dataset.from_pandas(df)
    return dataset

def extract_all_chars(batch):
    all_text = " ".join(batch["text"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}

def remove_special_characters(batch):
    chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�]'
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower() + " "
    return batch

def speech_file_to_array(batch):
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    batch["speech"] = speech_array[0].numpy()
    batch["sampling_rate"] = sampling_rate
    batch["target_text"] = batch["sentence"]
    return batch

def prepare_dataset(batch):
    batch["input_values"] = processor(batch["speech"], \
        sampling_rate=batch["sampling_rate"][0]).input_values
    with processor.as_target_processor():
        batch["labels"] = processor(batch["target_text"]).input_ids
    return batch

def gen_vocab(train, valid):
    vocab_train = train.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=train.column_names)
    vocab_valid = valid.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=valid.column_names)
    vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_valid["vocab"][0]))
    vocab_dict = {v: k for k, v in enumerate(vocab_list)}
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    vocab_path = "######"
    with open(vocab_path, 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file)

    return vocab_path

def load_processor(args, train, valid):

    # nécéssaire pour les fonctions utilisées via "map"
    global processor

    if not os.path.idsir(os.path.join(args.output_dir, "processor")):
        from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor

        print("> Generating vocab file")
        vocab_path = gen_vocab(train, valid)
        print(">", vocab_path)

        print(">> Creating processor from vocab file")

        tokenizer = Wav2Vec2CTCTokenizer(vocab_path, unk_token="[UNK]", \
            pad_token="[PAD]", word_delimiter_token="|")
        
        feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, \
            sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
        
        processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
        processor.save_pretrained(os.path.join(args.output_dir, "processor"))
    else :
        print(">> Load from pretrained processor", os.path.join(args.output_dir, "processor"))
        processor = Wav2Vec2Processor.from_pretrained(os.path.join(args.output_dir, "processor"))
    
    return processor

def clean_dataset(dataset):
    dataset = dataset.remove_columns(["origin", "duration", "category", "topic"])
    dataset = dataset.map(remove_special_characters)
    return dataset

def load_audio_from_dataset(dataset):
    dataset = dataset.map(speech_file_to_array, remove_columns=dataset.column_names)
    return dataset

def data_preparation(args):
    train = import_dataset(args.train)
    valid = import_dataset(args.valid)

    # Cleaning
    train = clean_dataset(train)
    valid = clean_dataset(valid)

    # Chargement des fichiers audios en vecteurs
    train = load_audio_from_dataset(train)
    valid = load_audio_from_dataset(valid)

    # Chargement/Création du processeur
    processor = load_processor(args, train ,valid)

    # Encodage des transcriptions
    train = train.map(prepare_dataset, 
                remove_columns=train.column_names, 
                batch_size=8,
                num_proc=4, 
                batched=True)
    valid = valid.map(prepare_dataset, 
                remove_columns=valid.column_names, 
                batch_size=8, 
                num_proc=4, 
                batched=True)

    return processor, train, valid
