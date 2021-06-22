import soundfile as sf
import torch
import argparse
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset, Dataset, load_metric
import pandas as pd
import random
import os
from jiwer import wer
import time

def map_to_array(batch):
    speech, _ = sf.read(batch["file"])
    batch["speech"] = speech
    return batch

def import_data(source, source_path, output):
    df = pd.read_csv(source)
    df["file"] = df["file"].apply(lambda x : os.path.join(source_path, x))
    # df["text"] = df["transcription"]
    df["text"] = df["transcription"].apply(lambda x : x.upper())
    df = df.drop("transcription", axis=1)
    df.to_csv(output, index=False)
    data = load_dataset('csv', data_files=output)
    data = data.map(map_to_array)
    return data

def map_to_result(batch):
    
    if torch.cuda.is_available():
    # if False:
        model.to("cuda")
        input_values = processor(
            batch["speech"], 
            sampling_rate=16000, 
            return_tensors="pt"
        ).input_values.to("cuda")
    else:
        input_values = processor(
            batch["speech"], 
            sampling_rate=16000, 
            return_tensors="pt"
        ).input_values

    with torch.no_grad():
        logits = model(input_values).logits

    pred_ids = torch.argmax(logits, dim=-1)
    # print(processor.batch_decode(pred_ids)[0])
    
    batch["pred_str"] = processor.batch_decode(pred_ids)[0].upper()
  
    return batch

def show_random_elements(dataset, out, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
        
    df = pd.DataFrame(dataset[picks])[['text', 'pred_str']]
    print(df)
    example_log = os.path.join(out, "example.txt")
    with open(example_log, "w") as ex_log:
        print(df, file=ex_log)

def main(out):
    data = import_data(TEST_CSV_RAW, TEST_PATH, TEST_CSV)
    start = time.time()
    results = data.map(map_to_result)
    end  = time.time()
    duration = end - start
    results = results["train"]
    wer_metric = load_metric("wer")

    print(f"Dataset : {TEST_CSV_RAW}" + 
        f"\nInference time : {duration:.2f} \n" + 
        "Test WER: {:.3f}".format(wer_metric.compute(predictions=results["pred_str"], references=results["text"])))
    print("\n")
    show_random_elements(results, out, num_examples=10)
    wer_log = os.path.join(out, "wer.txt")
    with open(wer_log, "w") as err_file:
        print(f"Dataset : {TEST_CSV_RAW}" + 
        f"\nInference time : {duration:.2f} \n" + 
        "Test WER: {:.3f}".format(wer_metric.compute(predictions=results["pred_str"], references=results["text"])), file=err_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default=None, type=str,
                        required=True, help="Train data tracker csv")
    parser.add_argument("--out", default=None, type=str,
                        required=True, help="Output dir for wer_log.txt")
    parser.add_argument("--processor", default=None, type=str,
                        required=False, help="Valid ata subset label")
    parser.add_argument("--model", default=None, type=str,
                        required=False, help="Model folder")
    parser.add_argument("--benchmark", default=None, type=str,
                        required=False, help="Benchmarking mode")                    
    args = parser.parse_args()
    processor_dir = args.processor
    model_dir = args.model
    out = args.out
    
    start = time.time()

    if not model_dir :
        print("benchmark modèle HG")

       
        
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53-french")
        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-xlsr-53-french")

        # ## add base et xlsr base
        # processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
        # model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-xlsr-53")

        # processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        # model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base")

        # processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        # model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    else :
        print("benchmark local model")
        processor = Wav2Vec2Processor.from_pretrained(processor_dir)
        model = Wav2Vec2ForCTC.from_pretrained(model_dir)

    TEST_CSV_RAW = args.test
    TEST_PATH = TEST_CSV_RAW.split("dataset")[0]
    TEST_CSV = os.path.join(TEST_PATH, "test_hg.csv")

    main(out)
    print(f"\n\nOverall time: {time.time() - start} s.")