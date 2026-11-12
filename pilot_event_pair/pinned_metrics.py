"""Unmodified functions extracted from the reviewed commit; VAL scoring only."""
import numpy as np

THD_RATIO = 0.15
MAX_ELAPSE = 10

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
