import torch
from torch.random import seed
import numpy as np
import ipdb
from transformers import RobertaTokenizerFast
from model.RobertaRetriever import RobertaRetriever
from model.RobertaReader import RobertaReader
import argparse
import jsonlines
from tqdm import tqdm
import string
import json
import re

parser = argparse.ArgumentParser()
parser.add_argument("--Hotpot_test_path", type=str, default="hotpot_dev_distractor_v1.json")
parser.add_argument("--pred_filename", type=str, default="pred.json")
parser.add_argument("--re_checkpoint", type=str)
parser.add_argument("--qa_checkpoint", type=str)
args = parser.parse_args()

device = "cuda"
seed = 42


####################
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(seed)


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


class HotpotRETestPipe:
    def __init__(self, tokenizer, file_path):
        self.max_length = 512
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.question_ids_list, self.question_length_list = [], []
        self.doc_length_list, self.document_ids_list, self.doc_num_list = [], [], []

    def load_data(self):
        fields = ["_id", "question", "context"]
        datadict = {}
        for field in fields:
            datadict[field] = []
        with jsonlines.open(self.file_path) as f:
            for data in f:
                for i in range(len(data)):
                    for field in fields:
                        datadict[field].append(data[i][field])
        self.datadict = datadict

    def question_tokenize(self, question):
        tokenized_question = self.tokenizer(question, truncation=True, max_length=self.max_length)
        question_ids = tokenized_question["input_ids"]

        return question_ids, len(question_ids)

    def document_tokenize(self, context):
        doc_length = []
        document_ids = []
        doc_num = len(context)
        for i in range(doc_num):
            tokenized_document = self.tokenizer(
                "".join([context[i][0]] + context[i][1]),
                truncation=True,
                max_length=self.max_length,
            )
            doc_length.append(len(tokenized_document["input_ids"][1:]))  # without [CLS]
            document_ids.append(tokenized_document["input_ids"][1:])  # without [CLS]

        return doc_length, document_ids, doc_num

    def process(self):
        self.load_data()
        for qes in self.datadict["question"]:
            question_ids, question_length = self.question_tokenize(qes)
            self.question_ids_list.append(question_ids)
            self.question_length_list.append(question_length)
        for doc in self.datadict["context"]:
            doc_length, document_ids, doc_num = self.document_tokenize(doc)
            self.doc_length_list.append(doc_length)
            self.document_ids_list.append(document_ids)
            self.doc_num_list.append(doc_num)
        return (
            self.datadict["_id"],
            self.question_ids_list,
            self.question_length_list,
            self.document_ids_list,
            self.doc_length_list,
            self.doc_num_list,
        )


class HotpotQATestPipe:
    def __init__(self, tokenizer, file_path, selected_pair_dict):
        self.max_length = 512
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.selected_pair_dict = selected_pair_dict
        self.title_dict = {}
        self.DOC1_SEP_num_dict = {}
        self.input_ids_list, self.attention_mask_list, self.SEP_index_list = [], [], []
        # SEP
        self.SEP = "</e>"
        self.SEP_id = tokenizer.convert_tokens_to_ids(self.SEP)
        # DOC
        self.DOC = "</d>"
        self.DOC_id = tokenizer.convert_tokens_to_ids(self.DOC)

    def load_data(self):
        fields = ["_id", "question", "context"]
        datadict = {}
        for field in fields:
            datadict[field] = []
        with jsonlines.open(self.file_path) as f:
            for data in f:
                for i in range(len(data)):
                    for field in fields:
                        datadict[field].append(data[i][field])
        self.datadict = datadict
        self.length = len(datadict["_id"])

    def _tokenize(self):
        for i in range(self.length):
            _id = self.datadict["_id"][i]
            question = self.datadict["question"][i]
            context_list = self.datadict["context"][i]
            doc1 = context_list[self.selected_pair_dict[_id][0][0]]
            doc2 = context_list[self.selected_pair_dict[_id][0][1]]
            self.title_dict[_id] = [doc1[0], doc2[0]]
            self.DOC1_SEP_num_dict[_id] = len(doc1[1])
            context1 = [self.DOC + " " + doc1[0]] + [self.SEP + " " + c for c in doc1[1]]
            context2 = [self.DOC + " " + doc2[0]] + [self.SEP + " " + c for c in doc2[1]]
            context = " ".join(context1 + context2)
            output = self.tokenizer(
                question,
                context,
                truncation=True,
                max_length=self.max_length,
                return_offsets_mapping=True,
            )
            self.input_ids_list.append(output["input_ids"])
            self.attention_mask_list.append(output["attention_mask"])

    def find_SEP(self, input_ids):
        SEP_index = []
        length_input_ids = len(input_ids)
        for i in range(length_input_ids):
            if input_ids[i] == self.SEP_id:
                SEP_index.append(i)
        return SEP_index

    def process(self):
        self.load_data()
        self._tokenize()
        for i in range(self.length):
            self.SEP_index_list.append(self.find_SEP(self.input_ids_list[i]))
        return (
            self.datadict["_id"],
            self.title_dict,
            self.DOC1_SEP_num_dict,
            self.input_ids_list,
            self.attention_mask_list,
            self.SEP_index_list,
        )


def padding(input_list):
    max_length = max([len(x) for x in input_list])
    for i in range(len(input_list)):
        input_list[i] = input_list[i] + [0] * (max_length - len(input_list[i]))
    return input_list


def test_re():
    re_tokenizer = RobertaTokenizerFast.from_pretrained("roberta-large")
    re_model = RobertaRetriever.from_pretrained("roberta-large")
    print("RE Checkpoint:", args.re_checkpoint, "Data:", args.Hotpot_test_path)
    re_model.load_state_dict(torch.load(args.re_checkpoint))
    re_model.to(device)
    repipe = HotpotRETestPipe(tokenizer=re_tokenizer, file_path=args.Hotpot_test_path)
    id_list, question_ids_list, question_length_list, document_ids_list, doc_length_list, doc_num_list = repipe.process()
    selected_pair_dict = {}
    length = len(id_list)
    for i in tqdm(range(length), desc="RE Testing:"):
        question_ids = torch.LongTensor([question_ids_list[i]]).to(device)
        question_length = torch.LongTensor([question_length_list[i]]).to(device)
        document_ids = torch.LongTensor([padding(document_ids_list[i])]).to(device)
        doc_length = torch.LongTensor([doc_length_list[i]]).to(device)
        doc_num = torch.LongTensor([doc_num_list[i]]).to(device)

        re_model.eval()
        with torch.no_grad():
            re_output = re_model(
                question_ids=question_ids,
                document_ids=document_ids,
                question_length=question_length,
                doc_length=doc_length,
                doc_num=doc_num,
            )
        selected_pair_dict[id_list[i]] = re_output["selected_pair"].cpu().tolist()
    torch.cuda.empty_cache()
    return selected_pair_dict


def test_qa(selected_pair_dict):
    sp_pred = {}
    answer_pred = {}
    SEP = "</e>"
    DOC = "</d>"
    qa_tokenizer = RobertaTokenizerFast.from_pretrained("roberta-large")
    qa_tokenizer.add_tokens([SEP, DOC])
    qa_model = RobertaReader.from_pretrained("roberta-large")
    qa_model.resize_token_embeddings(len(qa_tokenizer))
    print("QA Checkpoint:", args.qa_checkpoint, "Data:", args.Hotpot_test_path)
    qa_model.load_state_dict(torch.load(args.qa_checkpoint))
    qa_model.to(device)
    qapipe = HotpotQATestPipe(tokenizer=qa_tokenizer, file_path=args.Hotpot_test_path, selected_pair_dict=selected_pair_dict)
    id_list, selected_title_dict, DOC1_SEP_num_dict, input_ids_list, attention_mask_list, SEP_index_list = qapipe.process()
    length = len(id_list)
    for i in tqdm(range(length), desc="QA Testing:"):
        input_ids = torch.LongTensor([input_ids_list[i]]).to(device)
        attention_mask = torch.LongTensor([attention_mask_list[i]]).to(device)
        SEP_index = torch.LongTensor([SEP_index_list[i]]).to(device)
        qa_model.eval()
        with torch.no_grad():
            qa_output = qa_model(input_ids=input_ids, attention_mask=attention_mask, sentence_index=SEP_index)

        # Supporting Facts
        sentence_predictions = qa_output["sentence_predictions"].cpu().tolist()[0]
        title1 = selected_title_dict[id_list[i]][0]
        title2 = selected_title_dict[id_list[i]][1]
        sp = []
        for s in range(len(sentence_predictions)):
            if sentence_predictions[s] == 1:
                if s < DOC1_SEP_num_dict[id_list[i]]:
                    sp.append([title1, s])
                else:
                    sp.append([title2, s - DOC1_SEP_num_dict[id_list[i]]])
        sp_pred[id_list[i]] = sp

        # Answer
        anstype = torch.argmax(qa_output["type_logits"]).item()
        if anstype == 0:
            answer_pred[id_list[i]] = "no"
        elif anstype == 1:
            answer_pred[id_list[i]] = "yes"
        else:
            start = torch.argmax(qa_output["start_logits"]).item()
            end = torch.argmax(qa_output["end_logits"]).item()
            span_id = input_ids_list[i][start : end + 1]
            if span_id is None:
                answer_pred[id_list[i]] = ""
            else:
                answer_pred[id_list[i]] = normalize_answer(qa_tokenizer.decode(span_id))
    torch.cuda.empty_cache()
    return sp_pred, answer_pred


def main():
    selected_pair_dict = test_re()
    sp_pred, answer_pred = test_qa(selected_pair_dict)
    with open(args.pred_filename, "w", encoding="utf-8") as f:
        json.dump({"answer": answer_pred, "sp": sp_pred}, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
