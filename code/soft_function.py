from sklearn.metrics import f1_score
import numpy as np
import math
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 8))


def softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x


def label_smoothing(seq_len: int, token_start_index: int, token_end_index: int, weight: float = 0.1):
    smoothing_span = np.array([1 / seq_len] * seq_len)
    start_seq = np.array([0] * seq_len)
    start_seq[token_start_index] = 1
    start_seq = (1 - weight) * start_seq + weight * smoothing_span
    end_seq = np.array([0] * seq_len)
    end_seq[token_end_index] = 1
    end_seq = (1 - weight) * end_seq + weight * smoothing_span
    return start_seq, end_seq


def word_overlapping(seq_len: int, token_start_index: int, token_end_index: int, temperature: float = 0.3):
    gold_span = [0] * seq_len
    for i in range(token_start_index, token_end_index + 1):
        gold_span[i] = 1
    all_span = {}
    for s in range(seq_len):
        for e in range(s, seq_len):
            span = [0] * seq_len
            for i in range(s, e + 1):
                span[i] = 1
            f1 = f1_score(gold_span, span)
            # all_span[(s, e)] = f1 if f1 > 0 else -10000
            all_span[(s, e)] = f1 / temperature

    new_dist_s = []
    new_dist_e = []
    for s in range(seq_len):
        temp = 0
        for e in range(s, seq_len):
            temp += math.exp(all_span[(s, e)])
        new_dist_s.append(temp)
    for e in range(seq_len):
        temp = 0
        for s in range(0, e + 1):
            temp += math.exp(all_span[(s, e)])
        new_dist_e.append(temp)

    return np.array([i / sum(new_dist_s) for i in new_dist_s]), np.array([i / sum(new_dist_e) for i in new_dist_e])


def F1_smoothing(seq_len: int, token_start_index: int, token_end_index: int):
    gold_span = [0] * seq_len
    for i in range(token_start_index, token_end_index + 1):
        gold_span[i] = 1
    all_span = {}
    for s in range(seq_len):
        for e in range(s, seq_len):
            span = [0] * seq_len
            for i in range(s, e + 1):
                span[i] = 1

            f1 = f1_score(gold_span, span)
            all_span[(s, e)] = f1

    new_dist_s = [-100] * seq_len
    new_dist_e = [-100] * seq_len
    tag = 0
    for s in range(token_end_index):
        temp = 0
        for e in range(s, seq_len):
            temp += all_span[(s, e)]
        new_dist_s[tag] = temp
        tag += 1
    tag = 0
    for e in range(seq_len):
        temp = 0
        for s in range(0, e + 1):
            temp += all_span[(s, e)]
        new_dist_e[tag] = temp
        tag += 1

    return softmax(new_dist_s), softmax(new_dist_e)


def F1_smoothing_quick(seq_len: int, token_start_index: int, token_end_index: int):
    def fun_soft_start(s, token_start_index, token_end_index, seq_len):
        answer_length = token_end_index - token_start_index + 1
        if s < token_start_index:
            e = np.arange(token_start_index, token_end_index + 1)
            e = (e - token_start_index + 1) / (e - s + 1 + answer_length)
            e_0 = np.sum(e)
            e = np.arange(token_end_index + 1, seq_len)
            e = answer_length / (e - s + 1 + answer_length)
            e_1 = np.sum(e)
            return 2 * (e_0 + e_1)
        elif s >= token_start_index and s <= token_end_index:
            e = np.arange(s, token_end_index + 1)
            e = (e - s + 1) / (e - s + 1 + answer_length)
            e_0 = np.sum(e)
            e = np.arange(token_end_index + 1, seq_len)
            e = (token_end_index - s + 1) / (e - s + 1 + answer_length)
            e_1 = np.sum(e)
            return 2 * (e_0 + e_1)
        else:
            return 0

    def fun_soft_end(e, token_start_index, token_end_index, seq_len):
        answer_length = token_end_index - token_start_index + 1
        if e > token_end_index:
            s = np.arange(token_start_index, token_end_index + 1)
            s = (token_end_index - s + 1) / (e - s + 1 + answer_length)
            s_0 = np.sum(s)
            s = np.arange(0, token_start_index)
            s = answer_length / (e - s + 1 + answer_length)
            s_1 = np.sum(s)
            return 2 * (s_0 + s_1)
        elif e >= token_start_index and e <= token_end_index:
            s = np.arange(token_start_index, e + 1)
            s = (e - s + 1) / (e - s + 1 + answer_length)
            s_0 = np.sum(s)
            s = np.arange(0, token_start_index)
            s = (e - token_start_index + 1) / (e - s + 1 + answer_length)
            s_1 = np.sum(s)
            return 2 * (s_0 + s_1)
        else:
            return 0

    idx = range(seq_len)
    answer_length = token_end_index - token_start_index + 1
    soft_start_zero = [-100] * seq_len
    soft_start_label = [
        fun_soft_start(i, token_start_index=token_start_index, token_end_index=token_end_index, seq_len=seq_len) for i in idx
    ]
    if token_start_index - answer_length + 1 >= 0:
        soft_start_zero[token_start_index - answer_length + 1 : token_end_index + 1] = soft_start_label[
            token_start_index - answer_length + 1 : token_end_index + 1
        ]
    else:
        soft_start_zero[: token_end_index + 1] = soft_start_label[: token_end_index + 1]
    start_seq = softmax(soft_start_zero)

    soft_end_zero = [-100] * seq_len
    soft_end_label = [
        fun_soft_end(i, token_start_index=token_start_index, token_end_index=token_end_index, seq_len=seq_len) for i in idx
    ]
    if token_end_index + answer_length + 1 <= seq_len:
        soft_end_zero[token_start_index : token_end_index + answer_length + 1] = soft_end_label[
            token_start_index : token_end_index + answer_length + 1
        ]
    else:
        soft_end_zero[token_start_index:] = soft_end_label[token_start_index:]
    end_seq = softmax(soft_end_zero)
    # Hybird Distribution
    return start_seq, end_seq


if __name__ == "__main__":
    seq_len = 20
    token_start_index = 8
    token_end_index = 15
    # print(word_overlap_label(seq_len=10, token_start_index=2, token_end_index=5))
    soft_start_label, soft_end_label = F1_smoothing_quick(
        seq_len=seq_len, token_start_index=token_start_index, token_end_index=token_end_index
    )
    idx = range(seq_len)
    plt.bar(idx, soft_start_label)
    print(f"soft_start_label:{soft_start_label} sum:{np.sum(soft_start_label)}")
    plt.savefig("pict/F1_smoothing_quick-soft_start_label.png")
    plt.clf()
    plt.bar(idx, soft_end_label)
    print(f"soft_end_label:{soft_end_label} sum:{np.sum(soft_end_label)}")
    plt.savefig("pict/F1_smoothing_quick-soft_end_label.png")
