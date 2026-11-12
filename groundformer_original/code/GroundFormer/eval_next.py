"""QA accuracy of a prediction file ({split}-res.json), overall and per question type.

The grounding metrics (Acc@GQA, mIoP, mIoU) are reported by eval_ground.py.

Usage:
  python eval_next.py --dir <save_dir> --split test
  python eval_next.py --dir <save_dir> --split test --data_dir ../../datasets/nextgqa
"""
import argparse
import os.path as osp

from util import load_file

map_name = {'CW': 'Why', 'CH': 'How', 'TN': 'Bef&Aft', 'TC': 'When',
            'DC': 'Cnt', 'DL': 'Loc', 'DO': 'Other',
            'C': 'Acc_C', 'T': 'Acc_T', 'D': 'Acc_D'}


def accuracy_metric(sample_list, result):
    group = {'CW': [], 'CH': [], 'TN': [], 'TC': [], 'DC': [], 'DL': [], 'DO': []}
    for id, row in sample_list.iterrows():
        qns_id = str(row['video_id']) + '_' + str(row['qid'])
        qtype = str(row['type'])
        if qtype == 'TP':  # merge Temporal Past into Before&After
            qtype = 'TN'
        if qtype in group:
            group[qtype].append(qns_id)

    group_acc = {k: 0 for k in group}
    group_cnt = {k: 0 for k in group}
    overall_acc = {'C': 0, 'T': 0, 'D': 0}
    overall_cnt = {'C': 0, 'T': 0, 'D': 0}
    all_acc, all_cnt = 0, 0

    for qtype, qns_ids in group.items():
        acc, cnt = 0, 0
        for qid in qns_ids:
            if qid not in result:
                continue
            cnt += 1
            if result[qid]['answer'] == result[qid]['prediction']:
                acc += 1
        group_acc[qtype], group_cnt[qtype] = acc, cnt
        overall_acc[qtype[0]] += acc
        overall_cnt[qtype[0]] += cnt
        all_acc += acc
        all_cnt += cnt

    for cat in overall_acc:
        group_acc[cat] = overall_acc[cat]
        group_cnt[cat] = overall_cnt[cat]

    cols = [q for q in group_acc if group_cnt[q] > 0]
    print('\t'.join(map_name.get(q, q) for q in cols))
    print('\t'.join('{:.2f}'.format(group_acc[q] * 100.0 / group_cnt[q]) for q in cols))
    print('Acc: {:.2f}'.format(all_acc * 100.0 / all_cnt) if all_cnt else 'Acc: n/a')


def main(res_dir, split='test', data_dir='../../datasets/nextgqa/'):
    sample_list = load_file(osp.join(data_dir, f'{split}.csv'))
    result = load_file(osp.join(res_dir, f'{split}-res.json'))
    print('Evaluating {} ({} predictions)'.format(res_dir, len(result)))

    print('=============== All questions ===============')
    accuracy_metric(sample_list, result)

    gsub_file = osp.join(data_dir, f'gsub_{split}.json')
    if osp.exists(gsub_file):
        gsubset = load_file(gsub_file)
        gids, vids = [], []
        for idx, row in sample_list.iterrows():
            vid, qid = str(row['video_id']), str(row['qid'])
            if vid in gsubset and qid in gsubset[vid]['location'].keys():
                vids.append(vid)
                gids.append(idx)
        print('=============== Grounding subset ===============')
        print("Video: {}  Questions: {}".format(len(set(vids)), len(gids)))
        accuracy_metric(sample_list.iloc[gids], result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True, help="folder holding {split}-res.json")
    parser.add_argument("--split", type=str, default='test', choices=['val', 'test'])
    parser.add_argument("--data_dir", type=str, default='../../datasets/nextgqa/')
    args = parser.parse_args()

    main(args.dir, args.split, args.data_dir)
