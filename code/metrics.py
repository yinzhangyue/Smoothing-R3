import torch
import copy
import ipdb
from fastNLP import Metric
from collections import Counter
import string
from sklearn.metrics import f1_score
import re
import json


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


class DocselectionMetric(Metric):
    def __init__(self):
        r"""
        :param pred: 参数映射表中 `pred` 的映射关系，None表示映射关系为 `pred` -> `pred`
        :param target: 参数映射表中 `target` 的映射关系，None表示映射关系为 `target` -> `target`
        :param seq_len: 参数映射表中 `seq_len` 的映射关系，None表示映射关系为 `seq_len` -> `seq_len`
        """

        super().__init__()
        self.register_element("number", 0)
        self.register_element("em", 0)
        self.register_element("f1", 0)
        self.register_element("prec", 0)
        self.register_element("recall", 0)

    def update(self, selected_pair, gold_doc_pair, gold_answer_doc, doc_num, id):
        r"""
        evaluate函数将针对一个批次的预测结果做评价指标的累计

        :param torch.Tensor pred: 预测的tensor, tensor的形状可以是torch.Size([B,]), torch.Size([B, n_classes]),
                torch.Size([B, max_len]), 或者torch.Size([B, max_len, n_classes])
        :param torch.Tensor target: 真实值的tensor, tensor的形状可以是Element's can be: torch.Size([B,]),
                torch.Size([B,]), torch.Size([B, max_len]), 或者torch.Size([B, max_len])
        :param torch.Tensor seq_len: 序列长度标记, 标记的形状可以是None, None, torch.Size([B]), 或者torch.Size([B]).
                如果mask也被传进来的话seq_len会被忽略.

        """
        Batch_size = selected_pair.size(0)
        self.number += Batch_size

        for b in range(Batch_size):
            prediction = selected_pair[b].long().detach().cpu()
            gold = gold_doc_pair[b].long().detach().cpu()
            doc_num_single = doc_num[b].long().detach().cpu()
            tp, fp, fn = 0, 0, 0
            pred = torch.zeros(doc_num_single).scatter_(0, prediction, 1)
            label = torch.zeros(doc_num_single).scatter_(0, gold, 1)
            for s in range(doc_num_single):
                if pred[s] == 1:
                    if label[s] == 1:
                        tp += 1
                    else:
                        fp += 1
                elif label[s] == 1:
                    fn += 1
            prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
            recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
            f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
            em = 1.0 if fp + fn == 0 else 0.0
            self.em += em
            self.f1 += f1
            self.prec += prec
            self.recall += recall

    def get_metric(self):
        r"""
        get_metric函数将根据evaluate函数累计的评价指标统计量来计算最终的评价结果.

        :param bool reset: 在调用完get_metric后是否清空评价指标统计量.
        :return dict evaluate_result: {"acc": float}
        """
        evaluate_result = {
            "f1": self.f1.get_scalar() / self.number.get_scalar(),
            "em": self.em.get_scalar() / self.number.get_scalar(),
            "prec": self.prec.get_scalar() / self.number.get_scalar(),
            "recall": self.recall.get_scalar() / self.number.get_scalar(),
        }
        return evaluate_result


class SpanSentenceMetric(Metric):
    r"""
    准确率Metric（其它的Metric参见 :mod:`fastNLP.core.metrics` ）
    """

    def __init__(self, tokenizer=None):
        r"""
        :param pred: 参数映射表中 `pred` 的映射关系，None表示映射关系为 `pred` -> `pred`
        :param target: 参数映射表中 `target` 的映射关系，None表示映射关系为 `target` -> `target`
        :param seq_len: 参数映射表中 `seq_len` 的映射关系，None表示映射关系为 `seq_len` -> `seq_len`
        """

        super().__init__()
        self.tokenizer = tokenizer
        self.register_element("number", 0)
        self.register_element("em", 0)
        self.register_element("f1", 0)
        self.register_element("prec", 0)
        self.register_element("recall", 0)

        self.register_element("sp_em", 0)
        self.register_element("sp_f1", 0)
        self.register_element("sp_prec", 0)
        self.register_element("sp_recall", 0)

        self.register_element("cl_acc", 0)

        self.register_element("joint_f1", 0)
        self.register_element("joint_em", 0)
        self.register_element("joint_prec", 0)
        self.register_element("joint_recall", 0)

    def update(
        self,
        type_logits,
        start_logits,
        end_logits,
        sentence_predictions,
        sentence_labels,
        input_ids,
        answer,
        sentence_num,
    ):
        r"""
        evaluate函数将针对一个批次的预测结果做评价指标的累计

        :param torch.Tensor pred: 预测的tensor, tensor的形状可以是torch.Size([B,]), torch.Size([B, n_classes]),
                torch.Size([B, max_len]), 或者torch.Size([B, max_len, n_classes])
        :param torch.Tensor target: 真实值的tensor, tensor的形状可以是Element's can be: torch.Size([B,]),
                torch.Size([B,]), torch.Size([B, max_len]), 或者torch.Size([B, max_len])
        :param torch.Tensor seq_len: 序列长度标记, 标记的形状可以是None, None, torch.Size([B]), 或者torch.Size([B]).
                如果mask也被传进来的话seq_len会被忽略.

        """
        Batch_size = type_logits.size(0)
        self.number += Batch_size
        for b in range(Batch_size):
            type_logit, start, end, prediction, gold = (
                type_logits[b],
                start_logits[b],
                end_logits[b],
                sentence_predictions[b],
                sentence_labels[b],
            )
            ground_truth = answer[b]
            b_input_ids = input_ids[b].tolist()

            answer_type_pred = torch.argmax(type_logit).item()
            if answer_type_pred == 0:
                answers = "no"
            elif answer_type_pred == 1:
                answers = "yes"
            else:
                start = torch.argmax(start).item()
                end = torch.argmax(end).item()
                span_id = b_input_ids[start : end + 1]
                if span_id is None:
                    answers = ""
                else:
                    answers = self.tokenizer.decode(span_id)

            normalized_prediction = normalize_answer(answers)
            normalized_ground_truth = normalize_answer(ground_truth)
            answer_type_true = 2
            if normalized_ground_truth == "no":
                answer_type_true = 0
            elif normalized_ground_truth == "yes":
                answer_type_true = 1
            self.cl_acc += int(answer_type_pred == answer_type_true)

            em = normalized_prediction == normalized_ground_truth

            ZERO_METRIC = (0.0, 0.0, 0.0)
            if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
                f1, precision, recall = ZERO_METRIC
            elif normalized_ground_truth in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
                f1, precision, recall = ZERO_METRIC
            else:
                prediction_tokens = normalized_prediction.split()
                ground_truth_tokens = normalized_ground_truth.split()
                common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
                num_same = sum(common.values())
                if num_same == 0:
                    f1, precision, recall = ZERO_METRIC
                else:
                    precision = 1.0 * num_same / len(prediction_tokens)
                    recall = 1.0 * num_same / len(ground_truth_tokens)
                    f1 = (2 * precision * recall) / (precision + recall)
            self.em += float(em)
            self.f1 += f1
            self.prec += precision
            self.recall += recall

            tp, fp, fn = ZERO_METRIC
            # sp = {0: [], 1: []}
            for s in range(len(prediction)):
                if prediction[s] == 1:
                    if gold[s] == 1:
                        tp += 1
                    else:
                        fp += 1
                    # if s < DOC0_SEP_num[b].item():
                    #     sp[0].append(s)
                    # else:
                    #     sp[1].append(s - DOC0_SEP_num[b].item())
                elif gold[s] == 1:
                    fn += 1

            sp_prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
            sp_recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
            sp_f1 = 2 * sp_prec * sp_recall / (sp_prec + sp_recall) if sp_prec + sp_recall > 0 else 0.0
            sp_em = 1.0 if fp + fn < 1 else 0.0
            self.sp_em += sp_em
            self.sp_f1 += sp_f1
            self.sp_prec += sp_prec
            self.sp_recall += sp_recall

            joint_prec = precision * sp_prec
            joint_recall = recall * sp_recall
            if joint_prec + joint_recall > 0:
                joint_f1 = 2 * joint_prec * joint_recall / (joint_prec + joint_recall)
            else:
                joint_f1 = 0.0
            joint_em = em * sp_em
            self.joint_em += joint_em
            self.joint_f1 += joint_f1
            self.joint_prec += joint_prec
            self.joint_recall += joint_recall

    def get_metric(self):
        r"""
        get_metric函数将根据evaluate函数累计的评价指标统计量来计算最终的评价结果.

        :param bool reset: 在调用完get_metric后是否清空评价指标统计量.
        :return dict evaluate_result: {"acc": float}
        """
        evaluate_result = {
            "f1": self.f1.get_scalar() / self.number.get_scalar(),
            "em": self.em.get_scalar() / self.number.get_scalar(),
            "prec": self.prec.get_scalar() / self.number.get_scalar(),
            "recall": self.recall.get_scalar() / self.number.get_scalar(),
            "sp_f1": self.sp_f1.get_scalar() / self.number.get_scalar(),
            "sp_em": self.sp_em.get_scalar() / self.number.get_scalar(),
            "sp_prec": self.sp_prec.get_scalar() / self.number.get_scalar(),
            "sp_recall": self.sp_recall.get_scalar() / self.number.get_scalar(),
            "joint_f1": self.joint_f1.get_scalar() / self.number.get_scalar(),
            "joint_em": self.joint_em.get_scalar() / self.number.get_scalar(),
            "joint_prec": self.joint_prec.get_scalar() / self.number.get_scalar(),
            "joint_recall": self.joint_recall.get_scalar() / self.number.get_scalar(),
            "cl_acc": self.cl_acc.get_scalar() / self.number.get_scalar(),
        }
        return evaluate_result
