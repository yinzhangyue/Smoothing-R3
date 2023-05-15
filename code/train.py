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
parser.add_argument("--task", type=str, choices=["RE", "QA"], default="RE", help="task: RE/QA")
parser.add_argument("--data-path", type=str, default="../HotpotQAData", help="input file path")
parser.add_argument("--lr", default=5e-6, type=float, help="learning rate")
parser.add_argument("--warmupsteps", default=0.1, type=float, help="warmup steps")
parser.add_argument("--batch-size", default=1, type=int, help="batch size")
parser.add_argument("--accumulation-steps", default=8, type=int, help="accumulation steps (Total batch size = batch size * accumulation steps)")
parser.add_argument("--epoch", default=16, type=int, help="epoch size")
parser.add_argument("--seed", default=41, type=int, help="random seed")
parser.add_argument("--TSLA_T1", default=4, type=int, help="the epoches in first stage for TSLA")
parser.add_argument("--re-model", default="Electra", choices=["Roberta", "Electra"], type=str, help="backbone for RE task")
parser.add_argument("--qa-model", default="Electra", choices=["Roberta", "Deberta"], type=str, help="backbone for QA task")
parser.add_argument("--re-smoothing-method", default="None", choices=["None", "Constant", "LDLA", "TSLA"], type=str, help="smoothing method for Related Document Retrieval task")
parser.add_argument("--evidence-smoothing-method", default="None", choices=["None", "Constant", "LDLA", "TSLA"], type=str, help="smoothing method for Supporting Evidence Prediction task")
parser.add_argument("--qa-smoothing-method", default="None", choices=["None", "LabelSmoothing", "WordOverlapping", "F1Smoothing"], type=str, help="smoothing method for Answer Span Extraction task")

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

seed = args.seed


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
        callback = []
        name = "Task-{}-Smoothing_Method-{}-Seed-{}-lr-{}-batch_size-{}-accumulation_steps-{}-{}-checkpoints".format(args.task, args.re_smoothing_method, args.seed, args.lr, args.batch_size, args.accumulation_steps, args.re_model)

        print("###### {} ######".format(name))
        save_path = "../checkpoints/{}".format(name)

        callback.append(CheckpointCallback(topk=3, monitor="f1", folder=save_path, every_n_epochs=1))
        callback.append(TorchGradClipCallback(clip_value=1))
        callback.append(TorchWarmupCallback(args.warmupsteps, schedule="constant"))

        @Trainer.on(Event.on_train_epoch_begin(every=1))
        def decay_smoothing_weight(trainer):
            if args.re_smoothing_method == "None":
                re_model.smoothing_weight = 0
            elif args.re_smoothing_method == "Constant":
                re_model.smoothing_weight = 0.1
            elif args.re_smoothing_method == "LDLA":
                if re_model.epoch == 0:  # Initialization
                    re_model.smoothing_weight = 0.1
                else:
                    re_model.smoothing_weight -= 0.01 if re_model.smoothing_weight > 0 else 0
            elif args.re_smoothing_method == "TSLA":
                if re_model.epoch < args.TSLA_T1:
                    re_model.smoothing_weight = 0.1
                else:
                    re_model.smoothing_weight = 0
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
            # overfit_batches=3,
            # progress_bar="raw",
            fp16=True,
        )
        trainer.run()

    if args.task == "QA":
        if args.qa_model == "Deberta":
            # Deberta
            qa_tokenizer = DebertaV2TokenizerFast.from_pretrained("microsoft/deberta-v2-xxlarge")
            qa_tokenizer.add_tokens([Sentence_token, DOC_token])
            qa_model = DebertaReader.from_pretrained("microsoft/deberta-v2-xxlarge")
            qa_model.resize_token_embeddings(len(qa_tokenizer))

            @cache_results(_cache_fp="cache/qa_deberta_train_cache.pkl", _refresh=False)
            def load_data():
                databundle = HotpotQAPipe(tokenizer=qa_tokenizer).process_from_file(paths=args.data_path)
                return databundle

        elif args.qa_model == "Roberta":
            # Roberta
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
        name = "Task-{}-Evidence_Smoothing_Method-{}-QA_Smoothing_Method-{}-Seed-{}-lr-{}-batch_size-{}-accumulation_steps-{}-{}-checkpoints".format(
            args.task, args.evidence_smoothing_method, args.qa_smoothing_method, args.seed, args.lr, args.batch_size, args.accumulation_steps, args.qa_model
        )
        print(name)
        save_path = "../checkpoints/"

        callback = []
        callback.append(CheckpointCallback(topk=3, monitor="joint_f1", folder=save_path, every_n_epochs=1))
        callback.append(TorchGradClipCallback(clip_value=1))
        callback.append(TorchWarmupCallback(args.warmupsteps, schedule="constant"))

        qa_model.qa_smoothing_method = args.qa_smoothing_method

        @Trainer.on(Event.on_train_epoch_begin(every=1))
        def decay_smoothing_weight(trainer):
            if args.evidence_smoothing_method == "None":
                qa_model.smoothing_weight = 0
            elif args.evidence_smoothing_method == "Constant":
                qa_model.smoothing_weight = 0.1
            elif args.evidence_smoothing_method == "LDLA":
                if qa_model.epoch == 0:  # Initialization
                    qa_model.smoothing_weight = 0.1
                else:
                    qa_model.smoothing_weight -= 0.01 if qa_model.smoothing_weight > 0 else 0
            elif args.evidence_smoothing_method == "TSLA":
                if qa_model.epoch < args.TSLA_T1:
                    qa_model.smoothing_weight = 0.1
                else:
                    qa_model.smoothing_weight = 0
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
