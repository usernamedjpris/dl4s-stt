import argparse
import torch
import numpy as np
import os

# DataCollator
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, \
    TrainingArguments, Trainer
from datasets import load_metric

from prepare_finetuning_commonwp1 import data_preparation

# Test solution à cuda out of memory error
import gc

gc.collect()

torch.cuda.empty_cache()
@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch


def compute_metrics(pred):
    wer_metric = load_metric("wer")
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

def main(args):

    finetune_str = args.train.split(".")[0].split("_")[-1]
    # finetune_str = args.train.split("/")[-1].split(".")[0].split("_")[-1]

    if os.path.isdir(os.path.join(args.output_dir, finetune_str)):
        print("> Un dossier existe déjà pour ce dataset d'entraînement !")
        # input("> Appuyez sur <>ENTER<> pour lancer l'entraînement et écraser les checkpoints. \n")
    else :
        os.mkdir(os.path.join(args.output_dir, finetune_str))

    # Initialiation
    global processor
    processor, train, valid = data_preparation(args)
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    model_str = "facebook/wav2vec2-base" if args.model == "base" else "facebook/wav2vec2-large-xlsr-53"

    print("> Starting fine-tuning on model " + model_str )
    print(">> Training dataset :", args.train)
    print(">> Validation dataset :", args.valid)


    # input("####FIN DU TEST########")
    print(">> Loading model")
    model = Wav2Vec2ForCTC.from_pretrained(
        args.base, 
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.1,
        gradient_checkpointing=True, 
        ctc_loss_reduction="mean", 
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer)
        )

    # Freeze car fine tuning et pas pre-training
    model.freeze_feature_extractor()
    
    # Paramètres du trainer
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, finetune_str),
        group_by_length=True,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=2,
        evaluation_strategy="steps",
        num_train_epochs=50,
        fp16=True,
        save_strategy = "steps",
        save_steps=400,
        eval_steps=400,
        logging_strategy="steps",
        logging_steps=100,
        learning_rate=3e-4,
        warmup_steps=250,
        save_total_limit=2,
    )

    # Initialisation du trainer
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train,
        eval_dataset=valid,
        tokenizer=processor.feature_extractor,
    )

    checkpoint = args.checkpoint

    print(">> Start training model")

    # BUG : Memory error à l'import du checkpoint
    if checkpoint :
        print(">> Resume from checkpoint", checkpoint)
        trainer.train(resume_from_checkpoint=checkpoint)
    else :
        print(">> Initiate fine-tuning")
        trainer.train()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--cv", default=None, type=str,
                        required=True, help="Path vers le dossier commonvoice")
    parser.add_argument("-b", "--base", default=None, type=str,
                        required=True, help="Path vers le modèle pré entrainé")
    parser.add_argument("-t", "--train", default=None, type=str,
                        required=True, help="Train data tracker csv")
    parser.add_argument("-v", "--valid", default=None, type=str,
                        required=True, help="Valid data tracker csv")
    parser.add_argument("-o", "--output_dir", default=None, type=str,
                        required=True, help="Output dir")
    parser.add_argument("--batch", default=8, type=int,
                        required=True, help="Batch size par GPU")
                        
    parser.add_argument("-m", "--model", default="xlsr", type=str,
                        required=False, help="Pretrained model (base / xlsr)")
    parser.add_argument("-c", "--checkpoint", default=None, type=str,
                        required=False, help="Pretrained model (base / xlsr)")

    args = parser.parse_args()



# Ex usage du script : python hg_trainer --train data/WP1_15m/dataset_FR_15m.csv --valid data/WP1_valid_small/dataset_FR_valid_small.csv

    main(args)
