"""Frozen GroundFormer span reranking. This module never receives annotations."""
from dataclasses import dataclass
import hashlib

import numpy as np


MIN_GAIN = 0.05  # QA logit units, fixed before any model result
MAX_SPANS = 4   # baseline plus up to three alternatives
EPS = 1e-7     # membership only; never round or clip scored span endpoints


@dataclass(frozen=True)
class Candidate:
    span: tuple[float, float]
    indices: tuple[int, ...]


def pool(timestamps, original):
    t = np.asarray(timestamps, dtype=np.float64)
    s, e = map(float, original)
    if (t.shape != (32,) or not np.isfinite(t).all() or np.any(np.diff(t) < 0)
            or not (t[0] - EPS <= s <= e <= t[-1] + EPS)):
        raise ValueError("INVALID_TIMESTAMPS_OR_BASELINE_SPAN")

    def candidate(start, end):
        membership = np.flatnonzero((t >= start - EPS) & (t <= end + EPS))
        return Candidate((start, end), tuple(membership.tolist()))

    baseline = candidate(s, e)
    if s == e or len(baseline.indices) in (0, 32):
        return [baseline]

    # Translate using every sampled start. Preserve duration, candidate mask size and
    # supported timeline. Include each distinct mask once, keeping the original first.
    options = []
    seen = {baseline.indices}
    width = e - s
    for start in np.unique(t):
        start = float(start)
        end = start + width
        if end > t[-1]:
            continue
        proposal = candidate(start, end)
        if len(proposal.indices) != len(baseline.indices) or proposal.indices in seen:
            continue
        seen.add(proposal.indices)
        options.append(proposal)
    if len(options) >= MAX_SPANS:
        selection = np.linspace(0, len(options) - 1, MAX_SPANS - 1, dtype=int)
        options = [options[i] for i in selection]
    return [baseline, *options]


def controls(candidates, curve, timestamps):
    attention = np.asarray(curve, dtype=np.float64)
    if attention.shape != (32,) or not np.isfinite(attention).all():
        raise ValueError("INVALID_CURVE")
    t = np.asarray(timestamps, dtype=np.float64)
    midpoint = (t[0] + t[-1]) / 2
    return {
        "attention": int(np.argmax([attention[list(c.indices)].sum() for c in candidates])),
        "center": int(np.argmin([abs(sum(c.span) / 2 - midpoint) for c in candidates])),
    }


def donor_for(video_id, video_ids, timestamps_by_video):
    def duration(vid):
        t = timestamps_by_video[vid]
        return t[-1] - t[0]

    eligible = [vid for vid in video_ids if vid != video_id]
    if not eligible:
        raise ValueError("NO_OTHER_VIDEO_FOR_DONOR")
    matched = sorted(eligible, key=lambda vid: (abs(duration(vid) - duration(video_id)), vid))[:12]
    digest = hashlib.sha256(f"20261101|{video_id}".encode()).digest()
    return matched[int.from_bytes(digest[:8], "little") % len(matched)]


def interventions(features, donor, candidates):
    features, donor = map(lambda x: np.asarray(x, dtype=np.float32), (features, donor))
    if (features.ndim != 2 or features.shape[0] != 32 or features.shape != donor.shape
            or not np.isfinite(features).all() or not np.isfinite(donor).all()):
        raise ValueError("INVALID_VIDEO_FEATURES")
    yield "full", features
    yield "sham", features.copy()
    if len(candidates) == 1:
        return
    replacements = {"mean": np.broadcast_to(features.mean(axis=0), features.shape), "donor": donor}
    for kind, replacement in replacements.items():
        yield f"{kind}/null", replacement
        for i, candidate in enumerate(candidates):
            delete = features.copy()
            keep = replacement.copy()
            delete[list(candidate.indices)] = replacement[list(candidate.indices)]
            keep[list(candidate.indices)] = features[list(candidate.indices)]
            yield f"{kind}/{i}/delete", delete
            yield f"{kind}/{i}/keep", keep


def margin(logits, answer):
    return float(logits[answer] - max(logits[i] for i in range(5) if i != answer))


def choose(scores, answer, count, *, enabled=True):
    if not 0 <= answer < 5 or count < 1:
        raise ValueError("INVALID_ANSWER_OR_POOL")
    if not enabled:
        return 0, [], "DISABLED"
    scores = {key: np.asarray(value, dtype=np.float64) for key, value in scores.items()}
    if any(value.shape != (5,) or not np.isfinite(value).all() for value in scores.values()):
        raise ValueError("INVALID_LOGITS")
    baseline = scores["full"]
    if baseline.tobytes() != scores["sham"].tobytes():
        raise RuntimeError("SHAM_LOGITS_NOT_BITWISE_EQUAL")
    if count == 1:
        return 0, [], "NO_MATCHED_ALTERNATIVE"
    audits = []
    qualified = []
    for i in range(count):
        summary = {"index": i, "eligible": False}
        gains = []
        for kind in ("mean", "donor"):
            delete = scores[f"{kind}/{i}/delete"]
            keep = scores[f"{kind}/{i}/keep"]
            original_delete = scores[f"{kind}/0/delete"]
            necessity = margin(baseline, answer) - margin(delete, answer)
            gain = margin(original_delete, answer) - margin(delete, answer)
            chosen_drop = float(baseline[answer] - delete[answer])
            keep_margin = margin(keep, answer)
            support = keep_margin - margin(scores[f"{kind}/null"], answer)
            summary[kind] = {
                "necessity": necessity, "vs_original": gain, "chosen_logit_drop": chosen_drop,
                "keep_margin": keep_margin, "support_vs_null": support,
                "delete_logit_delta": (delete - baseline).tolist(),
            }
            if min(necessity, gain, chosen_drop, keep_margin, support) > MIN_GAIN:
                gains.append(gain)
        if i and len(gains) == 2:
            summary["eligible"] = True
            qualified.append((min(gains), i))
        audits.append(summary)
    if not qualified:
        return 0, audits, "CONTROL_AGREEMENT_OR_MARGIN_NOT_MET"
    return max(qualified, key=lambda pair: (pair[0], -pair[1]))[1], audits, "SWITCH"
