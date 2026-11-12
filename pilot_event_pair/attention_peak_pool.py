"""CPU replay of a fixed-width, peak-based GroundFormer proposal pool."""
from dataclasses import dataclass
import csv
import json
from pathlib import Path
import zipfile

import numpy as np

from pinned_metrics import find_seg_ada, get_tIoU

MAX_ALTERNATIVES = 3
EPS = 1e-7
SEED = 20261101
DRAWS = 10_000
EXPECTED = {"N": 3358, "videos": 567, "QA": 2030, "GQA": 621, "IoP05": 954}
EXPECTED_MIOP = 29.5399819475179
EXPECTED_CONTROLS = {"GQA": 623, "IoP05": 953}


@dataclass(frozen=True)
class Candidate:
    start: float
    end: float
    indices: tuple


def matched_windows(timestamps, span):
    """Same geometry and mask rules as the previous reranker, before its cap."""
    times = np.asarray(timestamps, dtype=np.float64)
    start, end = map(float, span)
    if (times.shape != (32,) or not np.isfinite(times).all() or np.any(np.diff(times) < 0)
            or not times[0] - EPS <= start <= end <= times[-1] + EPS):
        raise ValueError("INVALID_TIMESTAMPS_OR_BASELINE_SPAN")

    def make(left, right):
        idx = np.flatnonzero((times >= left - EPS) & (times <= right + EPS))
        return Candidate(left, right, tuple(idx.tolist()))

    original = make(start, end)
    if start == end or len(original.indices) in (0, 32):
        return original, []
    width = end - start
    seen = {original.indices}
    windows = []
    for left in np.unique(times).tolist():
        right = left + width
        if right > times[-1]:
            continue
        candidate = make(left, right)
        if len(candidate.indices) == len(original.indices) and candidate.indices not in seen:
            seen.add(candidate.indices)
            windows.append(candidate)
    return original, windows


def local_peaks(curve):
    """One representative per flat maximum, including endpoint peaks."""
    values = np.asarray(curve, dtype=np.float64)
    if values.shape != (32,) or not np.isfinite(values).all():
        raise ValueError("INVALID_ATTENTION_CURVE")
    peaks = []
    start = 0
    while start < len(values):
        end = start
        while end + 1 < len(values) and values[end + 1] == values[start]:
            end += 1
        left = start == 0 or values[start] > values[start - 1]
        right = end == len(values) - 1 or values[end] > values[end + 1]
        if left and right and end - start + 1 < len(values):
            peaks.append((start + end) // 2)
        start = end + 1
    return sorted(peaks, key=lambda i: (-values[i], i))


def peak_proposals(times, original, windows, curve):
    """Choose nearest centered windows, with disjoint sampled frames across alternatives."""
    selected = []
    used_frames = set()
    for peak in local_peaks(curve):
        if peak in original.indices or peak in used_frames:
            continue
        choices = [w for w in windows if peak in w.indices and used_frames.isdisjoint(w.indices)]
        if choices:
            chosen = min(choices, key=lambda w: (abs((w.start + w.end) / 2 - times[peak]), w.start))
            selected.append((chosen, peak))
            used_frames.update(chosen.indices)
        if len(selected) == MAX_ALTERNATIVES:
            break
    return selected


def evenly_spaced(windows):
    if len(windows) <= MAX_ALTERNATIVES:
        return windows
    return [windows[i] for i in np.linspace(0, len(windows) - 1, MAX_ALTERNATIVES, dtype=int)]


def attention_pick(candidates, curve):
    # The original is first. Exact score ties keep the earliest item.
    return max(candidates, key=lambda w: float(curve[list(w.indices)].sum()))


def overlaps(candidate, gt):
    pairs = [get_tIoU(interval, (candidate.start, candidate.end)) for interval in gt]
    return [float(max([0.0, *(pair[k] for pair in pairs)])) for k in (0, 1)]


def paired_ci(values, video_index, draws, denominators):
    sums = np.bincount(video_index, weights=values)
    rates = 100 * sums[draws].sum(axis=1) / denominators
    return np.quantile(rates, [.025, .975]).tolist()


def evaluate(rows, cache):
    ids = cache["ids"].tolist()
    if len(rows) != EXPECTED["N"] or len(set(ids)) != len(rows) or ids != [r["sample_id"] for r in rows]:
        raise ValueError("VAL_ID_OR_ORDER_MISMATCH")
    curves = cache["final_curves"]
    predictions = cache["prediction"]
    if curves.shape != (len(rows), 32) or predictions.shape != (len(rows),):
        raise ValueError("VAL_CACHE_SHAPE_MISMATCH")
    records = []
    arms = ("baseline", "even_attention", "peak_attention")
    iop = {arm: [] for arm in arms}
    iou = {arm: [] for arm in arms}
    qa = []
    for row, curve, prediction in zip(rows, curves, predictions):
        times = row["timestamps"]
        if find_seg_ada(curve.tolist(), times) != row["span"]:
            raise ValueError(f"BASELINE_DECODER_MISMATCH:{row['sample_id']}")
        original, windows = matched_windows(times, row["span"])
        score_curve = np.asarray(curve, dtype=np.float64)
        even_pool = [original, *evenly_spaced(windows)]
        peak_options = peak_proposals(times, original, windows, score_curve)
        peak_pool = [original, *(w for w, _ in peak_options)]
        selected = {"baseline": original, "even_attention": attention_pick(even_pool, score_curve),
                    "peak_attention": attention_pick(peak_pool, score_curve)}
        correct = int(prediction) == row["answer"]
        qa.append(correct)
        record = {"sample_id": row["sample_id"], "video_id": row["video_id"], "QA": int(correct),
                  "predicted_answer_index": int(prediction), "gt_intervals": row["GT_intervals"],
                  "even_alternatives": len(even_pool) - 1, "peak_alternatives": len(peak_options),
                  "peak_frame_indices": [i for _, i in peak_options],
                  "even_spans": [[w.start, w.end] for w in even_pool],
                  "peak_spans": [[w.start, w.end] for w in peak_pool]}
        for name, window in selected.items():
            u, p = overlaps(window, row["GT_intervals"])
            iop[name].append(p)
            iou[name].append(u)
            record[f"{name}_span"] = [window.start, window.end]
            record[f"{name}_iop"] = p
            record[f"{name}_iou"] = u
        for label, pool in (("even", even_pool), ("peak", peak_pool)):
            record[f"{label}_pool_hit"] = int(any(overlaps(w, row["GT_intervals"])[1] >= .5 for w in pool))
        records.append(record)
    qa = np.asarray(qa, dtype=bool)
    iop = {k: np.asarray(v, dtype=np.float64) for k, v in iop.items()}
    iou = {k: np.asarray(v, dtype=np.float64) for k, v in iou.items()}
    hit = {k: v >= .5 for k, v in iop.items()}
    gqa = {k: v & qa for k, v in hit.items()}
    videos, vi = np.unique([r["video_id"] for r in records], return_inverse=True)
    got = {"N": len(records), "videos": len(videos), "QA": int(qa.sum()),
           "GQA": int(gqa["baseline"].sum()), "IoP05": int(hit["baseline"].sum())}
    if got != EXPECTED or abs(100 * iop["baseline"].mean() - EXPECTED_MIOP) > 1e-8:
        raise ValueError(f"BASELINE_METRIC_MISMATCH:{got}")
    control = {"GQA": int(gqa["even_attention"].sum()), "IoP05": int(hit["even_attention"].sum())}
    if control != EXPECTED_CONTROLS:
        raise ValueError(f"EVEN_POOL_ATTENTION_PARITY_MISMATCH:{control}")
    draws = np.random.default_rng(SEED).integers(0, len(videos), size=(DRAWS, len(videos)))
    denominators = np.bincount(vi)[draws].sum(axis=1)
    summary = {"baseline_parity": got, "even_pool_parity": control, "arms": {}, "pool_diagnostics": {}}
    for name in arms:
        b = hit["baseline"]
        summary["arms"][name] = {
            "IoP05": int(hit[name].sum()), "mIoP_pct": 100 * iop[name].mean(),
            "GQA": int(gqa[name].sum()), "mIoU_pct": 100 * iou[name].mean(),
            "point_spans": sum(r[f"{name}_span"][0] == r[f"{name}_span"][1] for r in records),
            "changed_spans": sum(r[f"{name}_span"] != r["baseline_span"] for r in records),
            "IoP_rescued": int((~b & hit[name]).sum()), "IoP_lost": int((b & ~hit[name]).sum()),
        }
    for label in ("even", "peak"):
        covered = np.array([r[f"{label}_pool_hit"] for r in records], dtype=bool)
        failed_before = ~hit["baseline"]
        coverage = covered & failed_before
        rescued = hit[f"{label}_attention"] & failed_before
        summary["pool_diagnostics"][label] = {
            "baseline_failed": int(failed_before.sum()),
            "covered_among_failed": int(coverage.sum()),
            "rescued_among_covered": int((rescued & coverage).sum()),
            "missed_despite_coverage": int((~rescued & coverage).sum()),
            "no_alternative": sum(r[f"{label}_alternatives"] == 0 for r in records),
        }
    summary["peak_comparisons"] = {}
    for other in ("baseline", "even_attention"):
        block = {}
        for label, arrays in (("IoP05", hit), ("mIoP", iop), ("GQA", gqa)):
            delta = arrays["peak_attention"].astype(np.float64) - arrays[other].astype(np.float64)
            block[label] = {"delta_pp": 100 * delta.mean(),
                            "CI95_pp": paired_ci(delta, vi, draws, denominators)}
        summary["peak_comparisons"][other] = block
    passed = hit["peak_attention"].sum() > hit["baseline"].sum() and iop["peak_attention"].mean() >= iop["baseline"].mean()
    summary["decision"] = "PASSES_EXPLORATORY_IOP_GATE" if passed else "KEEP_BASELINE"
    return summary, records


def write_result(out, summary, records, lock):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "RESULTS.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (out / "LOCKED.json").write_text(json.dumps(lock, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (out / "PER_QUESTION.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0])
        writer.writeheader()
        for record in records:
            writer.writerow({k: json.dumps(v) if isinstance(v, list) else v for k, v in record.items()})
    report = [
        "# GroundFormer · Thử đặt đoạn quanh đỉnh attention",
        "", "3.358 câu VAL · 567 video · QA cố định · Kết quả thăm dò", "",
        "| Cách chọn | IoP@0,5 | IoP trung bình | Cứu / mất IoP | GQA |",
        "|---|---:|---:|---:|---:|",
    ]
    names = {"baseline": "GroundFormer gốc", "even_attention": "Đoạn rải đều + tổng attention",
             "peak_attention": "Đoạn theo đỉnh + tổng attention"}
    for name, v in summary["arms"].items():
        report.append(f"| {names[name]} | {v['IoP05']}/3358 | {v['mIoP_pct']:.3f}% | "
                      f"{v['IoP_rescued']} / {v['IoP_lost']} | {v['GQA']}/3358 |")
    report += ["", "| Danh sách | Có đoạn đạt trong 2.404 câu gốc trượt | Được cứu | Có đoạn đạt nhưng bỏ lỡ |",
               "|---|---:|---:|---:|"]
    for label, v in summary["pool_diagnostics"].items():
        report.append(f"| {label} | {v['covered_among_failed']}/2404 | {v['rescued_among_covered']} | "
                      f"{v['missed_despite_coverage']} |")
    report += ["", "CI95% ghép cặp theo video, 10.000 lần lấy mẫu. Đơn vị điểm phần trăm.", ""]
    for other, v in summary["peak_comparisons"].items():
        for label, s in v.items():
            low, high = s["CI95_pp"]
            report.append(f"- Đoạn theo đỉnh so với {names[other]}, {label} {s['delta_pp']:+.3f} "
                          f"(CI95% từ {low:+.3f} đến {high:+.3f}).")
    report += ["", f"**Quyết định thăm dò** `{summary['decision']}`.", "",
               "Danh sách có đoạn đạt là phép đo bằng GT. Bộ tạo đoạn và bộ chọn chỉ đọc đường attention, "
               "timestamp và đoạn gốc. Kết quả trên VAL đã dùng lại."]
    report = "\n".join(report) + "\n"
    (out / "KET_QUA.md").write_text(report, encoding="utf-8")
    archive = out.with_suffix(".zip")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
        for file in sorted(out.iterdir()):
            bundle.write(file, file.name)
    return report, archive
