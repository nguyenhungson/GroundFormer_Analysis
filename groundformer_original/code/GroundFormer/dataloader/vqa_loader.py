import sys
sys.path.insert(0, '../')

import os.path as osp
import random as rd

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import default_collate

from util import tokenize, get_qsn_type, group, load_file

rd.seed(1)

# wh-type of the question, derived from the question text
QTYPE_TO_IDX = {'how': 0, 'whato': 1, 'why': 2, 'where': 3, 'whata': 4, 'howm': 5, 'whatb': 6,
                'who': 7, 'which': 8, 'other': 9}
# annotated question type of NExT-GQA
DSET_QTYPE_TO_IDX = {'CW': 0, 'CH': 1, 'TC': 2, 'TP': 3, 'TN': 4, 'DC': 5, 'DL': 6, 'DO': 7}


class VideoQADataset(Dataset):
    def __init__(
        self,
        csv_path,
        features_path,
        qmax_words=20,
        amax_words=5,
        tokenizer=None,
        a2id=None,
        max_feats=20,
        mc=0,
        feat_type='CLIP',
        vg_loss=0
    ):
        """
        :param csv_path: path to a csv containing columns video_id, question, answer
        :param features_path: directory holding the video features
        :param qmax_words: maximum number of words for a question
        :param amax_words: maximum number of words for an answer
        :param tokenizer: language model tokenizer
        :param a2id: answer to index mapping (open-ended setting only)
        :param max_feats: maximum frames to sample from a video
        :param mc: number of answer candidates
        :param vg_loss: >0 enables the question-candidate sampling used by the first pass
        """
        self.data = pd.read_csv(csv_path)
        self.dset = csv_path.split('/')[-2]

        self.video_feature_path = features_path
        self.feat_type = feat_type
        self.use_frame = True
        self.use_mot = False
        self.qmax_words = qmax_words
        self.amax_words = amax_words
        self.a2id = a2id
        self.tokenizer = tokenizer
        self.v_questions = {}
        self.max_feats = max_feats
        self.mc = mc
        self.vg = vg_loss
        self.mode = osp.basename(csv_path).split('.')[0]  # train, val or test

        # GPT-4 question paraphrases, used to enlarge the pool of same-video questions
        self.agu = False

        if self.mode not in ['val', 'test']:
            self.all_answers = set(self.data['answer'])
            self.all_questions = set(self.data['question'])
            self.ans_group, self.qsn_group = group(self.data, gt=False)
            anno_path = osp.dirname(csv_path)
            agu_file = osp.join(anno_path, 'train_gpt4_sub.json')
            self.qsn_agu = load_file(agu_file) if osp.exists(agu_file) else {}

        self._gather_by_v()

        self.question_to_type = {}
        for idx, row in self.data.iterrows():
            self.question_to_type[row['question']] = row['type']

        app_feat_file = osp.join(self.video_feature_path, f'{feat_type}/{feat_type}_I_{self.mode}.h5')
        print('Load {}...'.format(app_feat_file))

        self.frame_feats = {}
        with h5py.File(app_feat_file, 'r') as fp:
            vids = fp['vid']
            feat_key = f'{feat_type}_I' if feat_type != 'Swin' else 'swin_2d'
            feats = fp[feat_key]
            for id, (vid, feat) in enumerate(zip(vids, feats)):
                vid = vid.decode("utf-8")
                self.frame_feats[str(vid)] = feat

    def __len__(self):
        return len(self.data)

    def _gather_by_v(self):
        """Group the questions of each video, used to sample same-video question negatives."""
        for idx, row in self.data.iterrows():
            vid, qsn, qtype = str(row['video_id']), row['question'], row['type']
            if qtype[0] == 'D': continue  # omit descriptive question
            if vid not in self.v_questions:
                self.v_questions[vid] = [qsn]
            else:
                self.v_questions[vid].append(qsn)

        if self.mode == 'train':
            for vid_qid, data in self.qsn_agu.items():
                vid = vid_qid.split('_')[0]
                if vid in self.v_questions:
                    for aug_qsn in data['gen']:
                        self.v_questions[vid].append(aug_qsn.rstrip('?'))

    def get_vid_feats(self, vid_id):
        feat = self.frame_feats[vid_id]
        fnum = feat.shape[0]
        sp_fids = np.linspace(0, fnum - 1, self.max_feats, dtype=int)
        feat = feat[sp_fids]

        return feat

    def __getitem__(self, index):

        cur_sample = self.data.loc[index]
        vid_id = cur_sample["video_id"]
        vid_id = str(vid_id)
        qid = str(cur_sample['qid'])

        vid_qid = f'{vid_id}_{qid}'
        vid_frames = self.get_vid_feats(vid_id)
        vlen = self.max_feats

        question_txt = cur_sample['question']
        dataset_qtype = 'null' if 'type' not in cur_sample else cur_sample['type']
        if self.mc >= 0:
            qsn_tk_id = torch.tensor(
                self.tokenizer.encode(
                    question_txt,
                    add_special_tokens=True,
                    padding="longest",
                    max_length=self.qmax_words,
                    truncation=True,
                ),
                dtype=torch.long
            )
            q_len = torch.tensor([len(qsn_tk_id)], dtype=torch.long)
        else:
            qsn_tk_id = torch.tensor([0], dtype=torch.long)
            q_len = torch.tensor([0], dtype=torch.long)

        qtype, ans_token_ids, answer_len = 0, 0, 0

        qsns_id, qsns_token_ids, qsns_seq_len, qsns_id_sv, qsns_token_ids_sv, qsns_seq_len_sv = 0, 0, 0, 0, 0, 0
        qtype = 'null' if 'type' not in cur_sample else cur_sample['type']

        if self.mode == 'train' and self.agu:
            if vid_qid in self.qsn_agu:
                agus = self.qsn_agu[vid_qid]['gen']
                question_txt = rd.sample(agus, 1)[0].rstrip('?') if rd.random() < 0.3 else question_txt

        if self.vg and self.mode not in ['val', 'test']:
            ### question candidates for the first pass ###
            try:
                qtype = get_qsn_type(question_txt, qtype)
            except:
                print(vid_qid, question_txt)
            neg_num = 5
            if qtype not in self.qsn_group or len(self.qsn_group[qtype]) < neg_num - 1:
                valid_qsncans = self.all_questions
            else:
                valid_qsncans = self.qsn_group[qtype]

            if rd.random() < 0.0:
                same_v_qsn = set(self.v_questions[vid_id])
                same_v_other_qsn = list(same_v_qsn - set(question_txt))
                num_other = len(same_v_other_qsn)
                if num_other >= self.mc - 1:
                    qchoices = rd.sample(same_v_other_qsn, self.mc - 1)
                else:
                    cand_qsn = valid_qsncans - same_v_qsn
                    if len(cand_qsn) < self.mc - 1 - num_other:
                        cand_qsn = set(self.all_question) - same_v_qsn
                    qchoices = same_v_other_qsn + rd.sample(list(cand_qsn), self.mc - 1 - num_other)
            else:
                cand_qsns = self.all_questions - set(self.v_questions[vid_id])
                qchoices = rd.sample(list(cand_qsns), self.mc - 1)

            qchoices.append(question_txt)
            rd.shuffle(qchoices)
            qsns_id = qchoices.index(question_txt)
            qsns_token_ids, qsn_tokens = tokenize(
                    qchoices,
                    self.tokenizer,
                    add_special_tokens=True,
                    max_length=self.qmax_words,
                    dynamic_padding=False,
                    truncation=True
                )
            qsns_seq_len = torch.tensor([len(qsn) for qsn in qsns_token_ids], dtype=torch.long)

            ### Q from same video ###
            same_v_qsn_sv = set(self.v_questions[vid_id])
            same_v_other_qsn_sv = list(same_v_qsn_sv - set(question_txt))
            num_other_sv = len(same_v_other_qsn_sv)
            if num_other_sv >= self.mc - 1:
                qchoices_sv = rd.sample(same_v_other_qsn_sv, self.mc - 1)
            else:
                qchoices_sv = same_v_other_qsn_sv + [question_txt] * (self.mc - 1 - num_other_sv)

            qchoices_sv.append(question_txt)
            rd.shuffle(qchoices_sv)
            qsns_id_sv = qchoices_sv.index(question_txt)
            qsns_token_ids_sv, qsn_tokens_sv = tokenize(
                    qchoices_sv,
                    self.tokenizer,
                    add_special_tokens=True,
                    max_length=self.qmax_words,
                    dynamic_padding=False,
                    truncation=True
                )
            qsns_seq_len_sv = torch.tensor([len(qsn) for qsn in qsns_token_ids_sv], dtype=torch.long)

        question_id = vid_id + '_' + str(cur_sample["qid"])

        if self.mc:
            ans = cur_sample['answer']
            choices = [str(cur_sample["a" + str(i)]) for i in range(self.mc)]
            answer_id = choices.index(ans) if ans in choices else -1

            qtype = get_qsn_type(question_txt, qtype)
            if self.mode not in ['val', 'test'] and rd.random() < 0.3:
                # replace the distractors with answers of the same question type
                try:
                    qtype = get_qsn_type(question_txt, qtype)
                except:
                    print(vid_qid, question_txt)
                if qtype not in self.ans_group or len(self.ans_group[qtype]) < self.mc - 1:
                    valid_anscans = self.all_answers
                else:
                    valid_anscans = self.ans_group[qtype]

                cand_answers = valid_anscans - set(ans)
                choices = rd.sample(list(cand_answers), self.mc - 1)
                choices.append(ans)

                rd.shuffle(choices)
                answer_id = choices.index(ans)

            # the model input is the question concatenated with each candidate
            answer_txts = [question_txt + f' {self.tokenizer.sep_token} ' + opt for opt in choices]
            answer_only_token_ids, _ = tokenize(
                choices,
                self.tokenizer,
                add_special_tokens=True,
                max_length=self.amax_words - self.qmax_words,
                dynamic_padding=False,
                truncation=True
            )
            try:
                ans_token_ids, answer_tokens = tokenize(
                    answer_txts,
                    self.tokenizer,
                    add_special_tokens=True,
                    max_length=self.amax_words,
                    dynamic_padding=False,
                    truncation=True
                )
            except:
                print('Fail to tokenize: ' + answer_txts)
            qas_len = torch.tensor([len(ans) for ans in ans_token_ids], dtype=torch.long)
        else:
            answer_txts = cur_sample["answer"]
            # -1 if the answer is not in the vocabulary, counted as a wrong prediction
            answer_id = self.a2id.get(answer_txts, -1)

        qtype_label = QTYPE_TO_IDX.get(qtype, 0)
        qtype_label2 = DSET_QTYPE_TO_IDX.get(dataset_qtype, 0)

        if self.vg and self.mode not in ['val', 'test']:
            qtype_labels = []
            qtype_labels2 = []
            for qchoice in qchoices:
                qchoice_type = get_qsn_type(qchoice, 'null')
                qtype_labels.append(QTYPE_TO_IDX.get(qchoice_type, 0))
                qtype_labels2.append(DSET_QTYPE_TO_IDX.get(self.question_to_type.get(qchoice, 'null'), 0))
        else:
            qtype_labels = [qtype_label]
            qtype_labels2 = [qtype_label2]

        if self.vg and self.mode not in ['val', 'test']:
            qtype_labels_sv = []
            qtype_labels2_sv = []
            for qchoice_sv in qchoices_sv:
                qchoice_type_sv = get_qsn_type(qchoice_sv, 'null')
                qtype_labels_sv.append(QTYPE_TO_IDX.get(qchoice_type_sv, 0))
                qtype_labels2_sv.append(DSET_QTYPE_TO_IDX.get(self.question_to_type.get(qchoice_sv, 'null'), 0))
        else:
            qtype_labels_sv = [qtype_label]
            qtype_labels2_sv = [qtype_label2]

        return {
            "video_id": vid_id,
            "video_frames": vid_frames,
            "video_len": vlen,                  # max_feats
            "type": qtype,                      # question type
            "qtype_label": qtype_labels,        # [mc] or [1] wh-type labels
            "qtype_label2": qtype_labels2,      # [mc] or [1] dataset type labels

            "qas_len": qas_len,
            "qsns_seq_len": qsns_seq_len,
            "q_len": q_len,

            "qsns_tokens": qsns_token_ids,      # question candidate token ids
            "qsn_token": qsn_tk_id,             # the real question
            "qsns_id": qsns_id,                 # index of the real question among the candidates
            "question_id": question_id,         # video id + question id

            "qa_tokens": ans_token_ids,         # question + answer candidate tokens
            "ans_id": answer_id,
            "qa_txt": answer_txts,              # question + answer candidate text

            "ans_tokens": answer_only_token_ids,
            "ans_txt": choices,

            "question_txt": question_txt,       # question text
            "question_txts": qchoices if (self.vg and self.mode not in ['val', 'test']) else [question_txt],

            "qsns_seq_len_sv": qsns_seq_len_sv,
            "qsns_tokens_sv": qsns_token_ids_sv,
            "qsns_id_sv": qsns_id_sv,
            "question_txts_sv": qchoices_sv if (self.vg and self.mode not in ['val', 'test']) else [question_txt],
            "qtype_label_sv": qtype_labels_sv,
            "qtype_label2_sv": qtype_labels2_sv,
        }


def videoqa_collate_fn(batch):
    """
    :param batch: [dataset[i] for i in N]
    :return: tensorized batch with the question and the ans candidates padded to the max length of the batch
    """
    qmax_len = max(len(batch[i]["qsn_token"]) for i in range(len(batch)))

    for i in range(len(batch)):
        if len(batch[i]["qsn_token"]) < qmax_len:
            batch[i]["qsn_token"] = torch.cat(
                [
                    batch[i]["qsn_token"],
                    torch.zeros(qmax_len - len(batch[i]["qsn_token"]), dtype=torch.long),
                ],
                0,
            )

    if not isinstance(batch[0]["qa_tokens"], int):
        amax_len = max(x["qa_tokens"].size(1) for x in batch)
        for i in range(len(batch)):
            if batch[i]["qa_tokens"].size(1) < amax_len:
                batch[i]["qa_tokens"] = torch.cat(
                    [
                        batch[i]["qa_tokens"],
                        torch.zeros(
                            (
                                batch[i]["qa_tokens"].size(0),
                                amax_len - batch[i]["qa_tokens"].size(1),
                            ),
                            dtype=torch.long,
                        ),
                    ],
                    1,
                )

    return default_collate(batch)


def get_videoqa_loaders(args, features_path, a2id, tokenizer, test_mode, rank=0, world_size=1):

    if test_mode != 'train':
        test_dataset = VideoQADataset(
            csv_path=args.val_csv_path if test_mode == 'val' else args.test_csv_path,
            features_path=features_path,
            qmax_words=args.qmax_words,
            amax_words=args.amax_words,
            tokenizer=tokenizer,
            a2id=a2id,
            max_feats=args.max_feats,
            mc=args.mc,
            feat_type=args.feat_type,
            vg_loss=args.vg_loss
        )

        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size_val,
            num_workers=args.num_thread_reader,
            sampler=test_sampler,
            drop_last=False,
            collate_fn=videoqa_collate_fn,
            pin_memory=True,
        )
        train_loader, val_loader = None, None
    else:
        train_dataset = VideoQADataset(
            csv_path=args.train_csv_path,
            features_path=features_path,
            qmax_words=args.qmax_words,
            amax_words=args.amax_words,
            tokenizer=tokenizer,
            a2id=a2id,
            max_feats=args.max_feats,
            mc=args.mc,
            feat_type=args.feat_type,
            vg_loss=args.vg_loss
        )
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_thread_reader,
            sampler=train_sampler,
            drop_last=True,
            collate_fn=videoqa_collate_fn,
            pin_memory=True,
        )
        val_dataset = VideoQADataset(
            csv_path=args.val_csv_path,
            features_path=features_path,
            qmax_words=args.qmax_words,
            amax_words=args.amax_words,
            tokenizer=tokenizer,
            a2id=a2id,
            max_feats=args.max_feats,
            mc=args.mc,
            feat_type=args.feat_type,
        )
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size_val,
            num_workers=args.num_thread_reader,
            sampler=val_sampler,
            collate_fn=videoqa_collate_fn,
            pin_memory=True,
        )
        test_loader = None

    return (train_loader, val_loader, test_loader)
