"""Sample video frames and load them into memory (used by tools/extract_feat_vid.py)."""
import os
import os.path as osp

import numpy as np
from PIL import Image


def sample_clips(total_frames, num_clips, num_frames_per_clip):
    """Split the video into num_clips clips and return the centre frame of each clip."""
    clips = []
    frames = list(range(total_frames))
    for i in np.linspace(0, total_frames, num_clips + 2, dtype=np.int32)[1: num_clips + 1]:
        clip_start = int(i) - int(num_frames_per_clip / 2)
        clip_end = int(i) + int(num_frames_per_clip / 2)
        clip_start = 0 if clip_start < 0 else clip_start
        clip_end = total_frames if clip_end > total_frames else clip_end
        clip = frames[clip_start:clip_end]
        if clip_start == 0 and len(clip) < num_frames_per_clip:
            shortage = num_frames_per_clip - (clip_end - clip_start)
            added_fids = [frames[clip_start] for _ in range(shortage)]
            if len(added_fids) > 0:
                clip = added_fids + clip
        if clip_end == total_frames and len(clip) < num_frames_per_clip:
            shortage = num_frames_per_clip - (clip_end - clip_start)
            added_fids = [frames[clip_end - 1] for _ in range(shortage)]
            if len(added_fids) > 0:
                clip += added_fids
        cid = clip[len(clip) // 2]  # use the center frame
        clips.append(cid)
    return clips


def video_sampling(vframe_dir, frame_num=32, mode='uniC', duration=None):
    frames = sorted(os.listdir(vframe_dir))
    # videos are decoded at 6fps, here we further sparsely sample by 3 for nextqa
    frame_ids = frames[::3]
    if duration != None:
        frame_ids = []
        for frame in frames:
            fid = int(frame.split('.')[0])
            # sample inside the ground-truth temporal segment
            if fid >= duration[0] and fid <= duration[1]:
                frame_ids.append(frame)

    frame_count = len(frame_ids)
    if mode == 'uniF':
        sampled_ids = np.linspace(0, frame_count - 1, frame_num).astype('int')
    elif mode == 'uniC':
        sampled_ids = sample_clips(frame_count, frame_num, num_frames_per_clip=4)
    elif mode == 'rad':
        sampled_ids = np.random.choice(frame_count, frame_num, replace=False).astype('int')
        sampled_ids = sorted(sampled_ids)

    sampled_frames = np.asarray(frame_ids)[sampled_ids]
    images = []
    for vid in sampled_frames:
        frame_path = osp.join(vframe_dir, vid)
        images.append(Image.open(frame_path).convert("RGB"))

    return images, sampled_frames
