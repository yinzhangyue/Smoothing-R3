from soft_function import label_smoothing, word_overlapping, F1_smoothing_quick
from fastNLP import DataSet
from fastNLP.io import Loader
from fastNLP.io import DataBundle
from fastNLP.io import Pipe
import jsonlines
import ipdb
import os
import warnings


class HotpotLoader(Loader):
    def __init__(self):
        super().__init__()

    def _load(self, fpath: str):
        fields = [
            "_id",
            "answer",
            "question",
            "supporting_facts",
            "context",
        ]
        datadict = {}
        for field in fields:
            datadict[field] = []
        with jsonlines.open(fpath) as f:
            for data in f:
                for field in fields:
                    for i in range(len(data)):
                        datadict[field].append(data[i][field])

        return DataSet(datadict)

    def download(self) -> str:
        pass


class HotpotREPipe(Pipe):
    def __init__(self, tokenizer=None):
        # Tokenizer
        self.tokenizer = tokenizer
        self.max_length = 512

    def process(self, data_bundle: DataBundle) -> DataBundle:
        # id
        data_bundle.copy_field("_id", "id")

        # question_ids question_length
        def question_tokenize(question):
            tokenized_question = self.tokenizer(
                question,
                truncation=True,
                max_length=self.max_length,
            )
            question_ids = tokenized_question["input_ids"]
            return {
                "question_ids": question_ids,
                "question_length": len(question_ids),
            }

        data_bundle.apply_field_more(question_tokenize, field_name="question", num_proc=4)

        # document_ids doc_length doc_num
        def document_tokenize(context):
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

            return {
                "doc_length": doc_length,
                "document_ids": document_ids,
                "doc_num": doc_num,
            }

        data_bundle.apply_field_more(document_tokenize, field_name="context", num_proc=4)

        # gold_doc_pair gold_answer_doc
        def create_gold_doc_pair_and_answer_doc(instance):
            gold_doc_pair = []
            gold_answer_doc = -1
            titles_list = [con[0] for con in instance["context"]]
            context_list = ["".join(con[1]) for con in instance["context"]]
            titles_label_list = set([sup[0] for sup in instance["supporting_facts"]])
            for titles_label in titles_label_list:
                gold_doc_pair.append(titles_list.index(titles_label))
            for gold_doc in gold_doc_pair:
                if instance["answer"] in context_list[gold_doc]:
                    gold_answer_doc = gold_doc
            return {"gold_doc_pair": gold_doc_pair, "gold_answer_doc": gold_answer_doc}

        data_bundle.apply_more(create_gold_doc_pair_and_answer_doc, num_proc=4)

        return data_bundle

    def process_from_file(self, paths=None) -> DataBundle:
        res = {}
        for path in os.listdir(paths):
            if "train" in path:
                res["train"] = self.process(HotpotLoader()._load(os.path.join(paths, path)))
            elif "dev" in path:
                res["dev"] = self.process(HotpotLoader()._load(os.path.join(paths, path)))
            elif "test" in path:
                res["test"] = self.process(HotpotLoader()._load(os.path.join(paths, path)))
        return DataBundle(datasets=res)


class HotpotQAPipe(Pipe):
    def __init__(self, tokenizer=None):
        # Tokenizer
        self.tokenizer = tokenizer
        # SEP
        self.SEP = "</e>"
        self.SEP_id = tokenizer.convert_tokens_to_ids(self.SEP)
        # DOC
        self.DOC = "</d>"
        self.DOC_id = tokenizer.convert_tokens_to_ids(self.DOC)
        self.max_length = 512

    def process(self, data_bundle: DataBundle) -> DataBundle:
        def _tokenize(instance):
            question = instance["question"]
            sup_titles_list = list(set([sup[0] for sup in instance["supporting_facts"]]))
            title_list = [con[0] for con in instance["context"]]
            sup_titles = []
            for title in title_list:
                if title in sup_titles_list:
                    sup_titles.append(title)
            if set(sup_titles) != set(sup_titles_list):
                warnings.warn("The required document does not appear in the document list!")

            assert len(sup_titles) == 2, "supporting_facts documents number is not 2"
            con_dict = {}
            for con in instance["context"]:
                if con[0] in sup_titles:
                    con_dict[con[0]] = [self.DOC + " " + con[0]] + [self.SEP + " " + c for c in con[1]]
            gold_context_list = con_dict[sup_titles[0]] + con_dict[sup_titles[1]]
            gold_context = " ".join(gold_context_list)
            # Padding side determines if we do (question|context) or (context|question).
            pad_on_right = self.tokenizer.padding_side == "right"
            tokenized_examples = self.tokenizer(
                question if pad_on_right else gold_context,
                gold_context if pad_on_right else question,
                truncation=True,
                max_length=self.max_length,
                return_offsets_mapping=True,
            )
            d = dict(tokenized_examples)
            d["gold_context_list"] = gold_context_list
            d["gold_context"] = gold_context
            d["sup_titles"] = sup_titles
            d["sequence_ids"] = tokenized_examples.sequence_ids(0)
            d["sentence_len"] = len(tokenized_examples["input_ids"])
            return d

        data_bundle.apply_more(_tokenize, num_proc=4)

        def find_start_end_positions(instance):
            input_ids = instance["input_ids"]
            question = instance["question"]
            answer = instance["answer"]
            sequence_ids = instance["sequence_ids"]
            start_char = instance["gold_context"].find(answer)
            offsets = instance["offset_mapping"]
            cls_index = input_ids.index(self.tokenizer.cls_token_id)

            if start_char == -1:
                return {"start_positions": cls_index, "end_positions": cls_index}
            else:
                # Start/end character index of the answer in the text.
                end_char = start_char + len(answer)

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != 1:
                    # print(sequence_ids[token_start_index])
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != 1:
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    return {"start_positions": cls_index, "end_positions": cls_index}
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    token_start_index = token_start_index - 1
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    token_end_index = token_end_index + 1

                return {"start_positions": token_start_index, "end_positions": token_end_index}

        data_bundle.apply_more(find_start_end_positions, num_proc=4)

        def cal_sentence_labels(instance):
            sup_titles = instance["sup_titles"]
            sup_dict = {sup_titles[0]: [], sup_titles[1]: []}
            sup_len_dict = {}

            for con in instance["context"]:
                sup_len_dict[con[0]] = len(con[1])
            # [['Allie Goertz', 0], ['Allie Goertz', 1], ['Allie Goertz', 2], ['Milhouse Van Houten', 0]]
            for sup in instance["supporting_facts"]:
                if sup[1] < sup_len_dict[sup[0]]:
                    sup_dict[sup[0]].append(sup[1])
                else:
                    warnings.warn(
                        "The number of marked sentences exceeds the number of sentences in the original document.ID:{} Doc:{}".format(
                            instance["_id"], sup[0]
                        )
                    )
            sentence_num = sup_len_dict[sup_titles[0]] + sup_len_dict[sup_titles[1]]
            DOC0_SEP_num = sup_len_dict[sup_titles[0]]
            sentence_labels = [0 for _ in range(sentence_num)]
            for idx in sup_dict[sup_titles[0]]:
                sentence_labels[idx] = 1
            for idx in sup_dict[sup_titles[1]]:
                sentence_labels[DOC0_SEP_num + idx] = 1
            return {"sentence_labels": sentence_labels, "sentence_num": sentence_num}

        data_bundle.apply_more(cal_sentence_labels, num_proc=4)

        def find_SEP(input_ids):
            SEP_index = []
            length_input_ids = len(input_ids)
            for i in range(length_input_ids):
                if input_ids[i] == self.SEP_id:
                    SEP_index.append(i)
            return {"sentence_index": SEP_index}

        data_bundle.apply_field_more(find_SEP, field_name="input_ids", num_proc=4)

        def parse_answer_type(answer):
            answer_type = 2
            if answer == "no":
                answer_type = 0
            elif answer == "yes":
                answer_type = 1
            return {"answer_type": answer_type}

        data_bundle.apply_field_more(parse_answer_type, field_name="answer", num_proc=4)

        def create_soft_labels(instance):
            token_start_index = instance["start_positions"]
            token_end_index = instance["end_positions"]
            seq_len = instance["sentence_len"]
            label_smoothing_start_label, label_smoothing_end_label = label_smoothing(
                seq_len, token_start_index, token_end_index
            )
            # word_overlapping_start_label, word_overlapping_end_label = word_overlapping(seq_len, token_start_index, token_end_index, 0.2)
            F1_smoothing_start_label, F1_smoothing_end_label = F1_smoothing_quick(seq_len, token_start_index, token_end_index)
            return {
                "label_smoothing_start_label": label_smoothing_start_label,
                "label_smoothing_end_label": label_smoothing_end_label,
                # "word_overlapping_start_label": word_overlapping_start_label,
                # "word_overlapping_end_label": word_overlapping_end_label,
                "F1_smoothing_start_label": F1_smoothing_start_label,
                "F1_smoothing_end_label": F1_smoothing_end_label,
            }

        data_bundle.apply_more(create_soft_labels, num_proc=4)

        return data_bundle

    def process_from_file(self, paths: str = None) -> DataBundle:
        res = {}
        for path in os.listdir(paths):
            if "train" in path:
                res["train"] = self.process(HotpotLoader()._load(os.path.join(paths, path)))
            elif "dev" in path:
                res["dev"] = self.process(HotpotLoader()._load(os.path.join(paths, path)))
            elif "test" in path:
                res["test"] = self.process(HotpotLoader()._load(os.path.join(paths, path)))
        return DataBundle(datasets=res)


class HotpotTestLoader(Loader):
    def __init__(self):
        super().__init__()

    def _load(self, fpath: str):
        fields = [
            "_id",
            "question",
            "context",
        ]
        datadict = {}
        for field in fields:
            datadict[field] = []
        with jsonlines.open(fpath) as f:
            for data in f:
                for field in fields:
                    for i in range(len(data)):
                        datadict[field].append(data[i][field])

        return DataSet(datadict)

    def download(self, dev_ratio=0.1, re_download=False) -> str:
        pass
