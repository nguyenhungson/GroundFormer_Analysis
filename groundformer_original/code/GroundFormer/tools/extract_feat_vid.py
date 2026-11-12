"""Extract per-frame CLIP features and store them as {feat_type}_I_{split}.h5.

The h5 file holds two datasets, matching what dataloader/vqa_loader.py expects:
  vid          : (N,)            video ids
  {feat_type}_I: (N, 32, D)      frame features, D = 512 (CLIP ViT-B/32) or 768 (CLIP ViT-L/14)

Input is either a folder of decoded frames (NExT-QA / NExT-GQA convention: videos
decoded at 6fps into {frame_dir}/{vid}/) or the raw videos ({video_dir}/{vid}.mp4).

Usage:
  python tools/extract_feat_vid.py --model_type CLIPL --mode test \
      --data_dir ../../../data/nextgqa --frame_dir /path/to/frames
"""
import sys
import os
sys.path.insert(0, '../')

import argparse
import os.path as osp

import h5py
import numpy as np
import pandas as pd
import torch
from PIL import Image

from dataloader.prepare_video import video_sampling, sample_clips

FRAME_NUM = 32


def get_vlist(filename):
    data = pd.read_csv(filename)
    return list(set(list(data['video_id'])))


def video_sampling_from_file(video_path, frame_num=FRAME_NUM):
    """Sample frame_num centre frames straight from a video file (no decoded frames needed)."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if total_frames <= 0:
        raise ValueError(f"Video has no frames: {video_path}")

    # same sampling as video_sampling: frame_num clips, centre frame of each clip
    frame_indices = [int(x) for x in sample_clips(total_frames, frame_num, num_frames_per_clip=4)]

    images = []
    cap = cv2.VideoCapture(video_path)
    for fid in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images.append(Image.fromarray(frame_rgb))
        elif images:
            images.append(images[-1])  # seek failed, repeat the last valid frame
        else:
            raise ValueError(f"Cannot read frame {fid} from {video_path}")
    cap.release()
    return images


def extract_feat(feat_type, mode, data_dir, frame_dir=None, video_dir=None, csv_path=None, out_dir=None):
    data_dir = osp.abspath(data_dir)
    if video_dir is not None:
        video_dir = osp.abspath(video_dir)
        frame_dir = None
    elif frame_dir is not None:
        frame_dir = osp.abspath(frame_dir)
    else:
        frame_dir = osp.join(data_dir, 'frames')

    qset_file = osp.abspath(csv_path) if csv_path else osp.join(data_dir, f'{mode}.csv')
    out_dir = osp.abspath(out_dir) if out_dir else osp.join(data_dir, 'video_feature', feat_type)
    os.makedirs(out_dir, exist_ok=True)
    app_feat_file = osp.join(out_dir, f'{feat_type}_I_{mode}.h5')

    vlist = get_vlist(qset_file)
    vnum = len(vlist)
    print(f"  csv: {qset_file}")
    print(f"  source: {'video_dir=' + str(video_dir) if video_dir else 'frames=' + str(frame_dir)}")
    print(f"  output: {app_feat_file}")
    print(f"  videos: {vnum}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    import clip as clip_model
    model, preprocess = clip_model.load('ViT-L/14' if feat_type == 'CLIPL' else 'ViT-B/32', device=device)
    model.eval()

    feat_dset, vid_dset = None, None
    with h5py.File(app_feat_file, 'w') as fp, torch.no_grad():
        for i, vid in enumerate(vlist):
            if video_dir is not None:
                video_path = None
                for ext in ('mp4', 'avi', 'mkv'):
                    cand = osp.join(video_dir, f"{vid}.{ext}")
                    if osp.exists(cand):
                        video_path = cand
                        break
                if video_path is None:
                    raise FileNotFoundError(f"Video not found: {video_dir}/{vid}.mp4|.avi|.mkv")
                frames = video_sampling_from_file(video_path, frame_num=FRAME_NUM)
            else:
                frames, _ = video_sampling(osp.join(frame_dir, str(vid)), frame_num=FRAME_NUM, mode='uniC')

            frames = [preprocess(img).numpy() for img in frames]
            images = torch.from_numpy(np.asarray(frames)).to(device)
            feat = model.encode_image(images).detach().cpu().numpy()  # [FRAME_NUM, D]

            if feat_dset is None:
                feat_dset = fp.create_dataset(f'{feat_type}_I', (vnum, FRAME_NUM, feat.shape[-1]), dtype=np.float32)
                vid_dset = fp.create_dataset('vid', (vnum,), dtype=h5py.special_dtype(vlen=str))
            feat_dset[i] = feat
            vid_dset[i] = str(vid)

            if i % 100 == 0:
                print(i, feat.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract per-frame CLIP features")
    parser.add_argument('--model_type', type=str, default='CLIPL', choices=['CLIP', 'CLIPL'])
    parser.add_argument('--mode', type=str, default='val', help='split: train, val or test')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='dataset root, e.g. ../../../data/nextgqa')
    parser.add_argument('--frame_dir', type=str, default=None,
                        help='folder of decoded frames, {frame_dir}/{vid}/ (default: data_dir/frames)')
    parser.add_argument('--video_dir', type=str, default=None,
                        help='raw videos instead of frames, {video_dir}/{vid}.mp4')
    parser.add_argument('--csv_path', type=str, default=None, help='override path to {mode}.csv')
    parser.add_argument('--out_dir', type=str, default=None, help='override output folder for the .h5')
    args = parser.parse_args()

    extract_feat(
        args.model_type,
        args.mode,
        data_dir=args.data_dir,
        frame_dir=args.frame_dir,
        video_dir=args.video_dir,
        csv_path=args.csv_path,
        out_dir=args.out_dir,
    )
