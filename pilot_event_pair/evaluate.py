"""VAL-only comparison for the frozen span reranker."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import time

import numpy as np

import gpu_session as native
from pinned_metrics import find_seg_ada, get_tIoU
from segment_reranker import choose, controls, donor_for, interventions, pool


ROOT = Path(__file__).resolve().parent
SEED = 20261101
BATCH = 64
BLOCK = BATCH
MINUTES = 60
PARITY_TOLERANCE = 1e-6
SMOKE_SENTINEL = "3530231811_8"
PAIRS = ("baseline", "attention", "center", "uniform")
METRICS = ("GQA", "mIoU", "IoU03", "IoU05", "mIoP", "IoP05", "point")


def bind(path, value):
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise RuntimeError(f"RESUME_CONTRACT_DIFFERS:{path.name}")
    else:
        native.write_json(path, value)


def row_hash(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True, allow_nan=False).encode()).hexdigest()


def reference():
    cfg = json.loads((ROOT / "CONFIG.json").read_bytes())
    raw = native.read_bound(ROOT / "BASELINE_REFERENCE.npz", cfg["baseline_reference_sha256"])
    with np.load(io.BytesIO(raw), allow_pickle=False) as z:
        ref = {k: z[k] for k in ("ids", "prediction", "final_curves")}
    rows_raw = native.read_bound(ROOT / "VAL_REFERENCE_ROWS.json", cfg["val_reference_rows_sha256"])
    rows = json.loads(rows_raw)
    if [r["sample_id"] for r in rows] != ref["ids"].tolist() or len(rows) != 3358:
        raise RuntimeError("REFERENCE_IDS_MISMATCH")
    return cfg, ref, rows


def score(row, span, prediction):
    overlaps = [get_tIoU(gt, span) for gt in row["GT_intervals"]]
    iou = max([0.0, *(float(pair[0]) for pair in overlaps)])
    iop = max([0.0, *(float(pair[1]) for pair in overlaps)])
    qa = prediction == row["answer"]
    return np.array([qa and iop >= .5, iou, iou >= .3, iou >= .5, iop, iop >= .5,
                     span[0] == span[1]], dtype=np.float64)


def plans(ref, rows):
    by_video = {}
    for row in rows:
        vid = row["video_id"]
        if by_video.setdefault(vid, row["timestamps"]) != row["timestamps"]:
            raise RuntimeError("VIDEO_TIMESTAMPS_INCONSISTENT")
    donors = {vid: donor_for(vid, by_video, by_video) for vid in by_video}
    result = []
    for i, row in enumerate(rows):
        candidates = pool(row["timestamps"], row["span"])
        curve = ref["final_curves"][i]
        result.append({
            "sid": row["sample_id"], "video": row["video_id"], "donor": donors[row["video_id"]],
            "pool": candidates, "controls": controls(candidates, curve, row["timestamps"]),
        })
    return result


def preflight(ref, rows, plan):
    predictions = ref["prediction"]
    baseline = np.array([score(r, p["pool"][0].span, int(a))
                         for r, p, a in zip(rows, plan, predictions)])
    qa = sum(int(a) == r["answer"] for a, r in zip(predictions, rows))
    peak = sum(any(s <= r["timestamps"][int(curve.argmax())] <= e
                   for s, e in r["GT_intervals"])
               for r, curve in zip(rows, ref["final_curves"]))
    if (qa, int(baseline[:, 0].sum()), peak) != (2030, 621, 1052):
        raise RuntimeError("CACHED_BASELINE_METRIC_PARITY")
    if abs(float(baseline[:, 1].mean()) * 100 - 16.9114057578) > 1e-6:
        raise RuntimeError("CACHED_BASELINE_MIOU_PARITY")
    choices = []
    for r, p, a in zip(rows, plan, predictions):
        choices.append(np.array([score(r, c.span, int(a)) for c in p["pool"]]))
    oracle = sum(float(x[:, 0].max()) for x in choices)
    uniform = np.mean([x[:, 0].mean() for x in choices])
    return {
        "status": "PASS_CACHE_AND_POOL" if oracle > 621 else "STOP_NO_GQA_HEADROOM",
        "N": len(rows), "videos": len({r["video_id"] for r in rows}), "QA": qa, "GQA": 621,
        "peak": peak, "mIoU_pct": float(baseline[:, 1].mean() * 100),
        "with_alternative": sum(len(p["pool"]) > 1 for p in plan),
        "pool_size_counts": {str(k): sum(len(p["pool"]) == k for p in plan) for k in range(1, 5)},
        "oracle_GQA_count": int(oracle), "oracle_extra_questions": int(oracle) - 621,
        "same_pool_uniform_expected_GQA_pct": float(uniform * 100),
        "oracle_is_annotation_only": True,
    }


def native_batch_indices(index, count):
    start = index // BATCH * BATCH
    return range(start, min(start + BATCH, count))


def checked_forward(batch, plan, forward, deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("RERANKER_TIME_LIMIT")
    return forward(batch, plan)


def native_parity(index, full, curve, rows, ref):
    row = rows[index]
    answer = int(full.argmax())
    expected_answer = int(ref["prediction"][index])
    ref_error = float(np.max(np.abs(curve - ref["final_curves"][index])))
    replay_span = list(find_seg_ada(curve.tolist(), row["timestamps"]))
    if (not np.isfinite(ref_error) or ref_error > PARITY_TOLERANCE
            or answer != expected_answer or replay_span != list(row["span"])):
        raise RuntimeError(
            f"NATIVE_PARITY_FAILED:{row['sample_id']}:curve_error={ref_error:.8g}"
            f":tol={PARITY_TOLERANCE:.8g}:answer={answer}/{expected_answer}"
            f":span={replay_span}/{row['span']}:native_batch_start={index // BATCH * BATCH}"
        )
    return ref_error


def live_block(indices, rows, plan, ref, forward, features, deadline):
    # Baseline and each intervention preserve the cache's exact question order and
    # batch size (including the unpadded final batch). Only frame values change.
    requested = set(indices)
    batches = {}
    outputs = {i: {} for i in indices}
    errors = {}
    for i in indices:
        native_indices = native_batch_indices(i, len(rows))
        batches[native_indices.start] = native_indices
    for native_indices in batches.values():
        batch = [(i, "full", features[plan[i]["video"]]) for i in native_indices]
        logits, curves = checked_forward(batch, plan, forward, deadline)
        for j, i in enumerate(native_indices):
            errors[i] = native_parity(i, logits[j], curves[j], rows, ref)
            if i in requested:
                outputs[i]["full"] = (logits[j], curves[j])
    print(f"Native baseline PASS: {len(errors)} câu, max curve error {max(errors.values()):.8g}", flush=True)
    for native_indices in batches.values():
        variants = {}
        for i in native_indices:
            if i in requested:
                p = plan[i]
                variants[i] = dict(interventions(features[p["video"]], features[p["donor"]], p["pool"]))
        names = dict.fromkeys(name for sample in variants.values() for name in sample if name != "full")
        for name in names:
            batch = [(i, name, variants[i][name]) if i in variants and name in variants[i]
                     else (i, "full", features[plan[i]["video"]]) for i in native_indices]
            logits, curves = checked_forward(batch, plan, forward, deadline)
            for j, (i, applied, _) in enumerate(batch):
                if applied == name:
                    outputs[i][name] = (logits[j], curves[j] if name == "sham" else None)
    records = []
    for i in indices:
        p, seen = plan[i], outputs[i]
        full, curve = seen["full"]
        answer = int(full.argmax())
        if curve.tobytes() != seen["sham"][1].tobytes():
            raise RuntimeError(f"SHAM_CURVE_PARITY_FAILED:{p['sid']}")
        all_logits = {name: pair[0].tolist() for name, pair in seen.items()}
        selected, audits, reason = choose(all_logits, answer, len(p["pool"]))
        records.append({
            "sid": p["sid"], "video": p["video"], "donor": p["donor"], "answer": answer,
            "spans": [list(c.span) for c in p["pool"]], "frame_indices": [list(c.indices) for c in p["pool"]],
            "picked": selected, "reason": reason, "controls": p["controls"], "all_logits": all_logits,
            "effects": audits, "reference_curve_max_error": errors[i],
        })
    return records


def summarize(records, rows):
    if [x["sid"] for x in records] != [r["sample_id"] for r in rows]:
        raise RuntimeError("MISSING_OR_MISORDERED_FULL_VAL")
    n = len(records)
    outcomes = {arm: [] for arm in ("reranker", *PAIRS)}
    for item, row in zip(records, rows):
        metrics = np.array([score(row, s, item["answer"]) for s in item["spans"]])
        indices = {"reranker": item["picked"], "baseline": 0, **item["controls"]}
        for arm, i in indices.items():
            outcomes[arm].append(metrics[i])
        outcomes["uniform"].append(metrics.mean(axis=0))
    matrices = {arm: np.array(vals) for arm, vals in outcomes.items()}
    if int(matrices["baseline"][:, 0].sum()) != 621 or sum(
            x["answer"] == r["answer"] for x, r in zip(records, rows)) != 2030:
        raise RuntimeError("FULL_VAL_BASELINE_MISMATCH")
    videos, video_index = np.unique([x["video"] for x in records], return_inverse=True)
    counts = np.bincount(video_index)
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(videos), size=(10000, len(videos)))
    denom = counts[draws].sum(axis=1)
    means = {arm: dict(zip(METRICS, (100 * x.mean(axis=0)).tolist())) for arm, x in matrices.items()}
    comparisons = {}
    for arm in PAIRS:
        diff = 100 * (matrices["reranker"] - matrices[arm])
        accum = np.zeros((len(videos), len(METRICS)))
        np.add.at(accum, video_index, diff)
        statistics = {}
        for j, metric in enumerate(METRICS):
            dist = accum[:, j][draws].sum(axis=1) / denom
            alpha = .05 / len(PAIRS) if metric == "GQA" else .05
            statistics[metric] = {"delta_pp": float(diff[:, j].mean()),
                                  "CI": np.quantile(dist, [alpha/2, 1-alpha/2]).tolist(),
                                  "confidence": 1 - alpha}
        comparisons[arm] = statistics
    flip = matrices["reranker"][:, 0] - matrices["baseline"][:, 0]
    gqa_gate = all(x["GQA"]["CI"][0] > 0 for x in comparisons.values())
    miou = comparisons["baseline"]["mIoU"]
    miou_gate = miou["delta_pp"] >= 0 and miou["CI"][0] >= -.5
    result = {
        "status": "PASS_EXPLORATORY_VAL" if gqa_gate and miou_gate else "NOT_CONFIRMED_ON_VAL",
        "N": n, "videos": len(videos), "QA": 2030, "metrics_pct": means,
        "vs_controls": comparisons, "gqa_gate": gqa_gate, "miou_gate": miou_gate,
        "gqa_rescued": int((flip > 0).sum()), "gqa_lost": int((flip < 0).sum()),
        "switched": sum(x["picked"] != 0 for x in records),
        "no_matched_alternative": sum(len(x["spans"]) == 1 for x in records),
        "training_steps": 0, "TEST_opens": 0,
    }
    return result, matrices


def run(args):
    started = time.monotonic()
    deadline = started + MINUTES * 60
    if args.mode != "preflight" and args.authorization != "GPU-EVAL-OK":
        raise PermissionError("GPU_EVAL_REQUIRES_EXPLICIT_GPU-EVAL-OK")
    cfg, ref, rows = reference()
    plan = plans(ref, rows)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    lock = {
        "split": "VAL", "source": cfg["commit"], "checkpoint": cfg["checkpoint_sha256"],
        "reference": cfg["baseline_reference_sha256"], "reference_rows": cfg["val_reference_rows_sha256"],
        "bundle_hashes": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in ("segment_reranker.py", "evaluate.py", "gpu_session.py",
                                       "core.py", "pinned_metrics.py", "CONFIG.json", "LOCKED.json")},
        "seed": SEED, "batch": BATCH, "block": BLOCK, "cap_minutes_per_invocation": MINUTES,
        "batch_layout": "native consecutive VAL order for each arm; final batch unpadded",
        "parity_tolerance": PARITY_TOLERANCE,
        "smoke_cohort": "first 31 original VAL rows plus previous failure " + SMOKE_SENTINEL,
        "primary": "GQA vs four controls; Bonferroni .05/4, paired video bootstrap 10000",
        "mIoU_gate": "delta >= 0 and lower CI95 >= -0.5pp versus baseline",
        "decision": "exploratory reused VAL; no TRAIN; no TEST; no tuning on VAL",
        "oracle_gate": "stop if even GT selection on the locked same-width pool cannot gain",
    }
    bind(out / "LOCKED.json", lock)
    fingerprint = hashlib.sha256((out / "LOCKED.json").read_bytes()).hexdigest()
    (out / "LOCKED.json.sha256").write_text(fingerprint + "\n", encoding="ascii")
    cached = preflight(ref, rows, plan)
    bind(out / "CPU_PREFLIGHT.json", cached)
    print(json.dumps(cached, ensure_ascii=False), flush=True)
    if args.mode == "preflight" or cached["status"] != "PASS_CACHE_AND_POOL":
        return
    if not all((args.repo, args.h5, args.checkpoint)):
        raise ValueError("REPO_H5_AND_CHECKPOINT_REQUIRED")
    if args.mode == "full" and not (out / "SMOKE.json").is_file():
        raise RuntimeError("RUN_SMOKE_FIRST")
    writer = out / "ACTIVE_WRITER"
    with writer.open("x", encoding="utf-8"):
        pass
    try:
        run_live(args, cfg, ref, rows, plan, out, fingerprint, started, deadline)
    finally:
        writer.unlink()


def run_live(args, cfg, ref, rows, plan, out, fingerprint, started, deadline):
    _, _, csv_raw, h5_raw, checkpoint, _, row_map = native.verify_inputs(args.repo, args.h5, args.checkpoint)
    sys.path.insert(0, str(args.repo / "code/GroundFormer"))
    torch, model, features, tokens, pad, _, software = native.model_runtime(
        cfg, checkpoint, csv_raw, h5_raw, row_map)
    del checkpoint, h5_raw
    if any(p.requires_grad or p.dtype != torch.float32 for p in model.parameters()):
        raise RuntimeError("FROZEN_FP32_REQUIRED")
    bind(out / "SOFTWARE.json", software)

    def forward(block, plan):
        qa = torch.cat([tokens[plan[i]["sid"]][0] for i, _, _ in block]).cuda()
        f = torch.from_numpy(np.stack([a for _, _, a in block])).cuda()
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
            encoded = model.video(f, torch.ones(f.shape[:2], device="cuda"))
            out = model.answer(qa, None, encoded, None, (qa != pad).float(),
                               torch.zeros(len(block), dtype=torch.long, device="cuda"))
        return out[0].reshape(len(block), 5).cpu().numpy(), out[-1]["fatt"].reshape(len(block), 32).cpu().numpy()

    if args.mode == "smoke":
        sentinel_index = next(i for i, p in enumerate(plan) if p["sid"] == SMOKE_SENTINEL)
        smoke_ids = sorted(set(range(31)) | {sentinel_index})
        if len(smoke_ids) < 32:
            smoke_ids = list(range(32))
        tick = time.monotonic()
        smoke = live_block(smoke_ids, rows, plan, ref, forward, features, deadline)
        elapsed = time.monotonic() - tick
        native_batches = len({i // BATCH for i in smoke_ids})
        native.write_json(out / "SMOKE.json", {"status": "PASS_PARITY_AND_SHAM", "fingerprint": fingerprint,
                         "N": len(smoke), "native_batches": native_batches,
                         "previous_failure_id": SMOKE_SENTINEL,
                         "max_curve_error": max(x["reference_curve_max_error"] for x in smoke),
                         "wall_seconds": elapsed,
                         "rough_full_forward_minutes": elapsed * ((len(rows) + BATCH - 1) // BATCH)
                         / native_batches / 60})
        print(f"Smoke PASS, {len(smoke)} câu, {elapsed:.1f}s", flush=True)
        return
    smoke_receipt = json.loads((out / "SMOKE.json").read_text(encoding="utf-8"))
    if (smoke_receipt["fingerprint"] != fingerprint
            or smoke_receipt["status"] != "PASS_PARITY_AND_SHAM"):
        raise RuntimeError("SMOKE_LOCK_DIFFERS")
    if (out / "RESULTS.json").is_file():
        print("VAL đã hoàn tất. Đọc RESULTS.json hoặc PER_QUESTION.csv.", flush=True)
        return
    results = []
    for start in range(0, len(rows), BLOCK):
        indices = list(range(start, min(start + BLOCK, len(rows))))
        path = out / f"shard_{start:04d}.json"
        if path.exists():
            shard = json.loads(path.read_bytes())
            if (shard["fingerprint"] != fingerprint
                    or shard["rows_sha256"] != row_hash(shard["rows"])
                    or [x["sid"] for x in shard["rows"]] != [plan[i]["sid"] for i in indices]):
                raise RuntimeError("RESUME_SHARD_MISMATCH")
            block = shard["rows"]
        else:
            block = live_block(indices, rows, plan, ref, forward, features, deadline)
            native.write_json(path, {"fingerprint": fingerprint, "rows_sha256": row_hash(block), "rows": block})
        results.extend(block)
        print(f"VAL {len(results)}/{len(rows)}", flush=True)
    result, matrices = summarize(results, rows)
    result["wall_seconds_this_invocation"] = time.monotonic() - started
    with (out / "PER_QUESTION.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["sample_id", "video_id", "answer", "QA", "picked", "reason",
                         "baseline_span", "reranker_span", "pool_size", *[
                             f"{arm}_{metric}" for arm in matrices for metric in METRICS]])
        for i, (x, r) in enumerate(zip(results, rows)):
            writer.writerow([x["sid"], x["video"], x["answer"], int(x["answer"] == r["answer"]),
                             x["picked"], x["reason"], x["spans"][0], x["spans"][x["picked"]],
                             len(x["spans"]), *[value for arr in matrices.values() for value in arr[i]]])
    native.write_json(out / "RESULTS.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "smoke", "full"), required=True)
    for option in ("out", "repo", "h5", "checkpoint"):
        parser.add_argument(f"--{option}", type=Path, required=option == "out")
    parser.add_argument("--authorization", default="")
    run(parser.parse_args())
