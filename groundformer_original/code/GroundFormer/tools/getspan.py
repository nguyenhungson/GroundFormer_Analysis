import sys
sys.path.insert(0, '../')

import argparse
import os.path as osp

import numpy as np

from util import load_file, save_to

THD_RATIO = 0.15   # frames above mean * (1 + THD_RATIO) extend the segment
MAX_ELAPSE = 10    # seconds; a gap larger than this stops the expansion


def find_seg_ada(vatt, cur):
    """Expand a segment around the peak frame, jointly considering the attention
    score and the distance to the maximum.

    vatt: per-frame attention scores, cur: timestamp (sec) of every frame
    """
    if not isinstance(vatt, list):
        vatt = [vatt]
    n = len(vatt)
    vatt = np.asarray(vatt)
    vatt = (np.asarray(vatt) - np.min(vatt)) / (np.max(vatt) - np.min(vatt) + 1e-12)

    max_id = np.argmax(vatt)
    mean_s = np.mean(vatt)

    path = [max_id]
    stamp_s = cur[max_id]
    thd = mean_s + THD_RATIO * mean_s
    dis = MAX_ELAPSE

    i = 1
    while max_id - i > 0:
        stamp = cur[max_id - i]
        elapse = abs(stamp_s - stamp)
        score = vatt[max_id - i] / (1 + elapse / dis)
        if score >= thd:
            path.append(max_id - i)
        elif elapse > dis:
            break
        i += 1

    i = 1
    while max_id + i < n:
        stamp = cur[max_id + i]
        elapse = abs(stamp_s - stamp)
        score = vatt[max_id + i] / (1 + elapse / dis)
        if score >= thd:
            path.append(max_id + i)
        elif elapse > dis:
            break
        i += 1

    sid, eid = np.min(path), np.max(path)
    return [cur[sid], cur[eid]]


def generate_ground(pred_file, seg_file, ground_file, qa_file):
    preds = load_file(pred_file)
    segs = load_file(seg_file)
    qas = load_file(qa_file)

    res_ground = {}
    missing = 0
    for idx, row in qas.iterrows():
        vid, qid = str(row['video_id']), str(row['qid'])
        vid_qid = '_'.join([vid, qid])
        if vid_qid not in preds:
            missing += 1
            continue
        atts = preds[vid_qid]
        cur = np.asarray(segs[vid])
        res_ground[vid_qid] = find_seg_ada(atts, cur)

    if missing:
        print(f'[getspan] WARNING: {missing}/{len(qas)} questions have no prediction in {pred_file}')
    save_to(ground_file, res_ground)


def main(data_dir, dset, anno_dir=None):
    if anno_dir is None:
        anno_dir = '../../datasets/nextgqa/'

    seg_file = osp.join(anno_dir, f'frame2time_{dset}.json')
    qa_file = osp.join(anno_dir, f'{dset}.csv')
    pred_file = osp.join(data_dir, f'{dset}_ground_att.json')
    ground_file = osp.join(data_dir, f'{dset}_ground_ada.json')

    generate_ground(pred_file, seg_file, ground_file, qa_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True, help="folder holding {split}_ground_att.json")
    parser.add_argument("--split", type=str, default='test', choices=['val', 'test'])
    parser.add_argument("--anno_dir", type=str, default=None,
                        help="folder holding {split}.csv and frame2time_{split}.json")
    args = parser.parse_args()

    main(args.dir, args.split, anno_dir=args.anno_dir)
