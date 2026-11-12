import argparse
import logging
import os
import os.path as osp

from util import load_file

# NExT-GQA question types.
# C = Causal, T = Temporal, D = Descriptive
# CW: Causal Why, CH: Causal How | TC: Temporal When, TP: Temporal Past, TN: Temporal Before&After
# DC: Descriptive Count, DL: Descriptive Location, DO: Descriptive Other
map_name = {'CW': 'Causal_Why', 'CH': 'Causal_How', 'TN': 'Temporal_BefAft', 'TC': 'Temporal_When', 'TP': 'Temporal_Past',
            'DC': 'Descriptive_Count', 'DL': 'Descriptive_Location', 'DO': 'Descriptive_Other',
            'C': 'Acc_Causal', 'T': 'Acc_Temporal', 'D': 'Acc_Descriptive'}

NEXTGQA_TYPES = ['CW', 'CH', 'TN', 'TC', 'TP', 'DC', 'DL', 'DO']

# STAR question types
STAR_TYPES = ['interaction', 'sequence', 'prediction', 'feasibility']
map_name_star = {t: t.capitalize() for t in STAR_TYPES}


def get_tIoU(loc, span):

    if span[0] == span[-1]:
        if loc[0] <= span[0] and span[0] <= loc[1]:
            return 0, 1
        else:
            return 0, 0

    span_u = (min(loc[0], span[0]), max(loc[-1], span[-1]))
    span_i = (max(loc[0], span[0]), min(loc[-1], span[-1]))
    dis_i = (span_i[1] - span_i[0])

    if span_u[-1] > span_u[0]:
        IoU = dis_i / (span_u[-1] - span_u[0])
    else:
        IoU = 0.0
    if span[-1] > span[0]:
        IoP = dis_i / (span[-1] - span[0])
    else:
        IoP = 0.0

    return IoU, IoP


def eval_ground(gt_ground, pred_ground, pred_qa=None, subset=None):
    """Overall Acc@QA / Acc@GQA / mIoP / mIoU."""
    mIoU, mIoP = 0, 0
    cnt, cqt = 0, 0
    cqa = 0  # Acc@QA: answer correct regardless of the grounding
    crt3, crt5 = 0, 0
    crtp3, crtp5 = 0, 0
    for vid, anno in gt_ground.items():
        for qid, locs in anno['location'].items():
            if not (f'{vid}_{qid}' in pred_ground):
                continue
            if subset != None and not (f'{vid}_{qid}' in subset):
                continue
            kid = f'{vid}_{qid}'
            if pred_qa and pred_qa[kid]['answer'] == pred_qa[kid]['prediction']:
                cqa += 1

            max_tIoU, max_tIoP = 0, 0
            for loc in locs:
                span = pred_ground[f'{vid}_{qid}']
                tIoU, tIoP = get_tIoU(loc, span)
                if tIoU > max_tIoU:
                    max_tIoU = tIoU
                if tIoP > max_tIoP:
                    max_tIoP = tIoP
            if max_tIoP >= 0.3:
                crtp3 += 1
                if max_tIoP >= 0.5:
                    crtp5 += 1
                    if pred_qa and pred_qa[kid]['answer'] == pred_qa[kid]['prediction']:
                        cqt += 1

            if max_tIoU >= 0.3:
                crt3 += 1
                if max_tIoU >= 0.5:
                    crt5 += 1

            cnt += 1
            mIoU += max_tIoU
            mIoP += max_tIoP

    mIoU = mIoU / cnt * 100
    mIoP = mIoP / cnt * 100
    acc_qa = cqa * 1.0 / cnt * 100 if pred_qa else 0.0
    acc_gqa = cqt * 1.0 / cnt * 100 if pred_qa else 0.0
    logging.info('Acc&QA Acc&GQA mIoP TIoP@0.3 TIoP@0.5 mIoU TIoU@0.3 TIoU@0.5')
    logging.info('{:.1f}\t{:.1f}\t{:.1f}\t{:.1f}\t{:.1f}\t{:.1f}\t{:.1f}\t{:.1f}'.format(
        acc_qa, acc_gqa, mIoP, crtp3 * 1.0 / cnt * 100, crtp5 * 1.0 / cnt * 100,
        mIoU, crt3 * 1.0 / cnt * 100, crt5 * 1.0 / cnt * 100))


def _eval_groups(gt_ground, pred_ground, pred_qa, group, subset=None):
    """Per question-type metrics for the given {type: [vid_qid, ...]} grouping."""
    stats = {}
    for qtype, qns_ids in group.items():
        cnt = acc_qa_cnt = acc_gqa_cnt = 0
        total_IoU = total_IoP = 0
        IoU3_cnt = IoU5_cnt = IoP3_cnt = IoP5_cnt = 0

        for qid in qns_ids:
            if not (qid in pred_ground):
                continue
            if subset != None and not (qid in subset):
                continue

            if pred_qa and pred_qa[qid]['answer'] == pred_qa[qid]['prediction']:
                acc_qa_cnt += 1

            max_tIoU, max_tIoP = 0, 0
            vid, qid_num = qid.split('_', 1)
            if vid not in gt_ground or qid_num not in gt_ground[vid]['location']:
                continue
            for loc in gt_ground[vid]['location'][qid_num]:
                span = pred_ground[qid]
                tIoU, tIoP = get_tIoU(loc, span)
                max_tIoU = max(max_tIoU, tIoU)
                max_tIoP = max(max_tIoP, tIoP)

            if max_tIoP >= 0.5 and pred_qa and pred_qa[qid]['answer'] == pred_qa[qid]['prediction']:
                acc_gqa_cnt += 1

            total_IoU += max_tIoU
            total_IoP += max_tIoP
            if max_tIoU >= 0.3: IoU3_cnt += 1
            if max_tIoU >= 0.5: IoU5_cnt += 1
            if max_tIoP >= 0.3: IoP3_cnt += 1
            if max_tIoP >= 0.5: IoP5_cnt += 1
            cnt += 1

        stats[qtype] = dict(cnt=cnt, acc_qa=acc_qa_cnt, acc_gqa=acc_gqa_cnt,
                            IoU=total_IoU, IoP=total_IoP,
                            IoU3=IoU3_cnt, IoU5=IoU5_cnt, IoP3=IoP3_cnt, IoP5=IoP5_cnt)
    return stats


def _log_group_table(title, order, names, stats):
    logging.info(f'\n=== {title} ===')
    logging.info('Type\tCnt\tAcc&QA\tAcc&GQA\tmIoP\tTIoP@0.3\tTIoP@0.5\tmIoU\tTIoU@0.3\tTIoU@0.5')
    for qtype in order:
        s = stats.get(qtype)
        cnt = s['cnt'] if s else 0
        if cnt == 0:
            logging.info(f'{names.get(qtype, qtype)}\t0\tN/A\tN/A\tN/A\tN/A\tN/A\tN/A\tN/A\tN/A')
            continue
        logging.info(f'{names.get(qtype, qtype)}\t{cnt}\t'
                     f'{s["acc_qa"] * 100.0 / cnt:.1f}\t{s["acc_gqa"] * 100.0 / cnt:.1f}\t'
                     f'{s["IoP"] * 100.0 / cnt:.1f}\t{s["IoP3"] * 100.0 / cnt:.1f}\t{s["IoP5"] * 100.0 / cnt:.1f}\t'
                     f'{s["IoU"] * 100.0 / cnt:.1f}\t{s["IoU3"] * 100.0 / cnt:.1f}\t{s["IoU5"] * 100.0 / cnt:.1f}')


def eval_ground_by_type(gt_ground, pred_ground, pred_qa, sample_list, subset=None):
    """NExT-GQA: per question-type and per category (Causal / Temporal / Descriptive)."""
    group = {t: [] for t in NEXTGQA_TYPES}
    for idx, row in sample_list.iterrows():
        vid, qid = str(row['video_id']), str(row['qid'])
        qtype = str(row['type'])
        if qtype in group and vid in gt_ground and qid in gt_ground[vid]['location']:
            group[qtype].append(f'{vid}_{qid}')

    stats = _eval_groups(gt_ground, pred_ground, pred_qa, group, subset)
    _log_group_table('Question Type-wise Performance', NEXTGQA_TYPES, map_name, stats)

    # aggregate the fine-grained types into the three categories
    overall = {c: dict(cnt=0, acc_qa=0, acc_gqa=0, IoU=0, IoP=0, IoU3=0, IoU5=0, IoP3=0, IoP5=0)
               for c in ['C', 'T', 'D']}
    for qtype, s in stats.items():
        if s['cnt'] == 0:
            continue
        cat = overall[qtype[0]]
        for k in cat:
            cat[k] += s[k]
    _log_group_table('Category-wise Performance', ['C', 'T', 'D'], map_name, overall)


def eval_ground_by_type_star(gt_ground, pred_ground, pred_qa, sample_list, subset=None):
    """STAR: interaction / sequence / prediction / feasibility."""
    group = {t: [] for t in STAR_TYPES}
    for idx, row in sample_list.iterrows():
        vid, qid = str(row['video_id']), str(row['qid'])
        raw_type = str(row.get('type', '')).strip().lower()
        if not raw_type or raw_type not in group:
            # the CSV type is T1/T2/..., the question type is the qid prefix:
            # "Interaction_T1_1" -> "interaction"
            parts = qid.split('_')
            if parts:
                raw_type = parts[0].lower()
        if raw_type in group and vid in gt_ground and qid in gt_ground[vid]['location']:
            group[raw_type].append(f'{vid}_{qid}')

    stats = _eval_groups(gt_ground, pred_ground, pred_qa, group, subset)
    _log_group_table('STAR Question Type-wise Performance', STAR_TYPES, map_name_star, stats)


def main(res_dir, filename, data_dir=None, dataset='nextgqa'):
    if data_dir is None:
        data_dir = '../../datasets/nextgqa/'
    dset = filename.split('_')[0]  # val or test

    sample_list = load_file(osp.join(data_dir, f'{dset}.csv'))
    gt_ground = load_file(osp.join(data_dir, f'gsub_{dset}.json'))
    pred_ground = load_file(osp.join(res_dir, filename))
    pred_qa = load_file(osp.join(res_dir, f'{dset}-res.json'))

    eval_by_type_fn = eval_ground_by_type_star if dataset == 'star' else eval_ground_by_type

    logging.info(f'Evaluating {filename}')
    eval_ground(gt_ground, pred_ground, pred_qa, subset=None)
    eval_by_type_fn(gt_ground, pred_ground, pred_qa, sample_list, subset=None)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True, help="folder holding the prediction files")
    parser.add_argument("--file", type=str, default='test_ground_ada.json', help="prediction file name")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="folder holding {split}.csv and gsub_{split}.json")
    parser.add_argument("--dataset", type=str, default='nextgqa', choices=['nextgqa', 'nextqa', 'star'])
    args = parser.parse_args()

    main(args.dir, args.file, data_dir=args.data_dir, dataset=args.dataset)
