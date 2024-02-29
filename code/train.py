import torch
from torch.random import seed
import torch.optim as optim

import numpy as np
from fastNLP import Trainer, cache_results, Event, TorchWarmupCallback, TorchGradClipCallback, CheckpointCallback, prepare_dataloader

from preprocess import HotpotREPipe, HotpotQAPipe

from transformers import DebertaV2TokenizerFast, RobertaTokenizerFast, ElectraTokenizerFast
from metrics import SpanSentenceMetric, DocselectionMetric

from model.ElectraRetriever import ElectraRetriever
from model.RobertaRetriever import RobertaRetriever

from model.DebertaReader import DebertaReader
from model.RobertaReader import RobertaReader
import ipdb
import argparse

parser = argparse.ArgumentParser()
# Basic Setting
parser.add_argument("--task", type=str, choices=["RE", "QA"], default="RE", help="task: RE/QA")
parser.add_argument("--data-path", type=str, default="../HotpotQAData", help="input file path")
parser.add_argument("--re-model", default="Roberta", choices=["Roberta", "Electra"], type=str, help="backbone for RE task")
parser.add_argument("--qa-model", default="Roberta", choices=["Roberta", "Deberta"], type=str, help="backbone for QA task")
# Hyperparameters
parser.add_argument("--lr", default=5e-6, type=float, help="learning rate")
parser.add_argument("--warmupsteps", default=0.1, type=float, help="warmup steps")
parser.add_argument("--batch-size", default=1, type=int, help="batch size")
parser.add_argument("--accumulation-steps", default=16, type=int, help="accumulation steps (Total batch size = batch size * accumulation steps)")
parser.add_argument("--epoch", default=16, type=int, help="epoch size")
parser.add_argument("--seed", default=41, type=int, help="random seed")
parser.add_argument("--LDLA-decay-rate", default=0.01, type=float, help="the decay rate for LDLA")

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = args.seed
print("Set LDLA Decay Rate:", args.LDLA_decay_rate)


# set seed
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(seed)

# QA Special Token
Sentence_token = "</e>"
DOC_token = "</d>"


def main():
    if args.task == "RE":
        # ELECTRA
        if args.re_model == "Electra":
            re_tokenizer = ElectraTokenizerFast.from_pretrained("google/electra-large-discriminator")
            re_model = ElectraRetriever.from_pretrained("google/electra-large-discriminator")

            @cache_results(_cache_fp="cache/re_electra_train_cache.pkl", _refresh=False)
            def load_data():
                databundle = HotpotREPipe(tokenizer=re_tokenizer).process_from_file(paths=args.data_path)
                return databundle

        # Roberta
        elif args.re_model == "Roberta":
            # FacebookAI/roberta-large
            re_tokenizer = RobertaTokenizerFast.from_pretrained("roberta-large")
            re_model = RobertaRetriever.from_pretrained("roberta-large")

            @cache_results(_cache_fp="cache/re_roberta_train_cache.pkl", _refresh=False)
            def load_data():
                databundle = HotpotREPipe(tokenizer=re_tokenizer).process_from_file(paths=args.data_path)
                return databundle

        metrics = DocselectionMetric()
        databundle = load_data()
        dl = prepare_dataloader(databundle, backend="torch", batch_size=args.batch_size, num_workers=4)
        optimizer = optim.AdamW(re_model.parameters(), lr=args.lr)
        name = "Task-{}-Seed-{}-lr-{}-batch_size-{}-accumulation_steps-{}-{}-checkpoints".format(args.task, args.seed, args.lr, args.batch_size, args.accumulation_steps, args.re_model)
        print("###### {} ######".format(name))
        save_path = "../checkpoints/{}".format(name)

        callback = []
        callback.append(CheckpointCallback(topk=3, monitor="f1", folder=save_path, every_n_epochs=1))
        callback.append(TorchGradClipCallback(clip_value=1))
        callback.append(TorchWarmupCallback(args.warmupsteps, schedule="constant"))

        @Trainer.on(Event.on_train_epoch_begin(every=1))
        # Initialization
        def decay_smoothing_weight(trainer):
            if re_model.epoch == 0:
                re_model.smoothing_weight = 0.1
            else:
                re_model.smoothing_weight -= args.LDLA_decay_rate if re_model.smoothing_weight > 0 else 0
            re_model.epoch += 1

        trainer = Trainer(
            driver="auto",
            model=re_model,
            train_dataloader=dl["train"],
            evaluate_dataloaders=dl["dev"],
            device=device,
            metrics={"docselection": metrics},
            callbacks=callback,
            optimizers=optimizer,
            n_epochs=args.epoch,
            accumulation_steps=args.accumulation_steps,
            fp16=True,
        )
        trainer.run()

    if args.task == "QA":
        # Deberta
        if args.qa_model == "Deberta":
            qa_tokenizer = DebertaV2TokenizerFast.from_pretrained("microsoft/deberta-v2-xxlarge")
            qa_tokenizer.add_tokens([Sentence_token, DOC_token])
            qa_model = DebertaReader.from_pretrained("microsoft/deberta-v2-xxlarge")
            qa_model.resize_token_embeddings(len(qa_tokenizer))

            @cache_results(_cache_fp="cache/qa_deberta_train_cache.pkl", _refresh=False)
            def load_data():
                databundle = HotpotQAPipe(tokenizer=qa_tokenizer).process_from_file(paths=args.data_path)
                return databundle

        # Roberta
        elif args.qa_model == "Roberta":
            # FacebookAI/roberta-large
            qa_tokenizer = RobertaTokenizerFast.from_pretrained("roberta-large")
            qa_tokenizer.add_tokens([Sentence_token, DOC_token])
            qa_model = RobertaReader.from_pretrained("roberta-large")
            qa_model.resize_token_embeddings(len(qa_tokenizer))

            @cache_results(_cache_fp="cache/qa_roberta_train_cache.pkl", _refresh=False)
            def load_data():
                databundle = HotpotQAPipe(tokenizer=qa_tokenizer).process_from_file(paths=args.data_path)
                return databundle

        metrics = SpanSentenceMetric(tokenizer=qa_tokenizer)

        databundle = load_data()
        dl = prepare_dataloader(databundle, backend="torch", batch_size=args.batch_size, num_workers=4)
        optimizer = optim.AdamW(qa_model.parameters(), lr=args.lr)
        name = "Task-{}-Seed-{}-lr-{}-batch_size-{}-accumulation_steps-{}-{}-checkpoints".format(args.task, args.seed, args.lr, args.batch_size, args.accumulation_steps, args.qa_model)
        print("###### {} ######".format(name))
        save_path = "../checkpoints/{}".format(name)

        callback = []
        callback.append(CheckpointCallback(topk=3, monitor="joint_f1", folder=save_path, every_n_epochs=1))
        callback.append(TorchGradClipCallback(clip_value=1))
        callback.append(TorchWarmupCallback(args.warmupsteps, schedule="constant"))

        @Trainer.on(Event.on_train_epoch_begin(every=1))
        def decay_smoothing_weight(trainer):
            # Initialization
            if qa_model.epoch == 0:
                qa_model.qa_smoothing_weight = 0.1
            else:
                qa_model.qa_smoothing_weight -= args.LDLA_decay_rate if qa_model.qa_smoothing_weight > 0 else 0
            qa_model.epoch += 1

        trainer = Trainer(
            driver="auto",
            model=qa_model,
            train_dataloader=dl["train"],
            evaluate_dataloaders=dl["dev"],
            device=device,
            metrics={"SpanSentence": metrics},
            callbacks=callback,
            optimizers=optimizer,
            n_epochs=args.epoch,
            accumulation_steps=args.accumulation_steps,
            fp16=True,
        )
        trainer.run()


if __name__ == "__main__":
    main()
