import json
import os
import os.path as osp

import pandas as pd
import torch


def tokenize(
    seq,
    tokenizer,
    add_special_tokens=True,
    max_length=10,
    dynamic_padding=True,
    truncation=True,
):
    """
    :param seq: sequence of sequences of text
    :param tokenizer: language model tokenizer
    :return: torch tensor padded up to length max_length of tokens
    """
    token_ids = tokenizer.batch_encode_plus(
        seq,
        add_special_tokens=add_special_tokens,
        max_length=max_length,
        padding="longest" if dynamic_padding else "max_length",
        truncation=truncation,
    )["input_ids"]
    tokens = ''
    return torch.tensor(token_ids, dtype=torch.long), tokens


class AverageMeter:
    """ Computes and stores the average and current value for training stats """

    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_mask(lengths, max_length):
    """ Computes a batch of padding masks given batched lengths """
    mask = 1 * (
        torch.arange(max_length).unsqueeze(1).to(lengths.device) < lengths
    ).transpose(0, 1)
    return mask


def compute_a2v(vocab_path, bert_tokenizer, amax_words):
    """ Precomputes the answer embeddings of the open-ended vocabulary """
    a2id = json.load(open(vocab_path, "r"))
    id2a = {v: k for k, v in a2id.items()}
    a2v, _ = tokenize(
        list(a2id.keys()),
        bert_tokenizer,
        add_special_tokens=True,
        max_length=amax_words,
        dynamic_padding=True,
        truncation=True,
    )
    if torch.cuda.is_available():
        a2v = a2v.cuda()
    return a2id, id2a, a2v


def get_qsn_type(qsn, ans_rsn):
    """wh-type of a question, used as the label of the type classifier."""
    dos = ['does', 'do', 'did']
    bes = ['was', 'were', 'is', 'are']
    w5h1 = ['what', 'who', 'which', 'why', 'how', 'where']
    qsn_sp = qsn.split()
    type = qsn_sp[0].lower()
    if type == 'what':
        if qsn_sp[1].lower() in dos:
            type = 'whata'
        elif qsn_sp[1].lower() in bes:
            type = 'whatb'
        else:
            type = 'whato'
    elif type == 'how':
        if qsn_sp[1].lower() == 'many':
            type = 'howm'
    elif type not in w5h1:
        type = 'other'
    if ans_rsn in ['pr', 'cr']:
        # for causalVid, we distinguish answer and reason
        type += 'r'
    return type


def group(csv_data, gt=True):
    """Group answers / questions by question type, used to sample the candidates."""
    ans_group, qsn_group = {}, {}
    for idx, row in csv_data.iterrows():
        qsn, ans = row['question'], row['answer']
        if gt:
            type = row['type']
            if type == 'TP': type = 'TN'
        else:
            type = 'null' if 'type' not in row else row['type']
            type = get_qsn_type(qsn, type)
        if type not in ans_group:
            ans_group[type] = {ans}
            qsn_group[type] = {qsn}
        else:
            ans_group[type].add(ans)
            qsn_group[type].add(qsn)
    return ans_group, qsn_group


def load_file(filename):
    file_type = osp.splitext(filename)[-1]
    if file_type == '.csv':
        data = pd.read_csv(filename)
    else:
        with open(filename, 'r') as fp:
            if file_type == '.json':
                data = json.load(fp)
            elif file_type == '.txt':
                data = fp.readlines()
                data = [datum.rstrip('\n') for datum in data]
    return data


def save_to(filename, data):
    dirname = osp.dirname(filename)
    if not osp.exists(dirname):
        os.makedirs(dirname)
    with open(filename, 'w') as fp:
        json.dump(data, fp)
