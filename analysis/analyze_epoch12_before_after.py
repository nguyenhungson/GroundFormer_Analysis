"""Join saved Original epoch-12 predictions to NExT-GQA question metadata."""

import argparse
import csv
import gzip
import hashlib
import io
import json
import re
import shutil
import zipfile
from pathlib import Path
from statistics import fmean


COMMIT = "96b7e92040d8ef65d240ae3c2d135134ae9c7225"
EPOCH12_SHA = "5a2f49eb92b2d9631d9787da34589a6c6a1c9ffc814d727c74cb7ad61ef5e776"
EXPECTED = {"val": (3358, 567, 2030, 621), "test": (5553, 990, 3381, 1096)}
MARKERS = re.compile(r"\b(before|after)\b", re.IGNORECASE)


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(split, group, rows):
    n = len(rows)
    qa = sum(int(row["QA"]) for row in rows)
    gqa = sum(int(row["GQA"]) for row in rows)
    iop = sum(int(row["IoP@0.5"]) for row in rows)
    return {
        "split": split, "group": group, "n": n,
        "videos": len({row["video_id"] for row in rows}),
        "qa": qa, "gqa": gqa, "iop50": iop,
        "qa_correct_iop_fail": qa - gqa,
        "qa_wrong_iop_pass": iop - gqa,
        "qa_wrong_iop_fail": n - qa - iop + gqa,
        "peak_hit": sum(int(row["peak_hit"]) for row in rows),
        "qa_correct_iop_fail_peak_outside": sum(
            row["QA"] == "1" and row["IoP@0.5"] == "0" and row["peak_hit"] == "0"
            for row in rows
        ),
        "iop_zero": sum(float(row["mIoP"]) == 0 for row in rows),
        "mIoP_pct": 100 * fmean(float(row["mIoP"]) for row in rows) if n else None,
        "mIoU_pct": 100 * fmean(float(row["mIoU"]) for row in rows) if n else None,
        "tn": sum(row["type"] == "TN" for row in rows),
    }


def extract_split(archive, data, split, inputs):
    meta_path = data / f"{split}.csv"
    gt_path = data / f"gsub_{split}.json"
    inputs[meta_path.name] = digest(meta_path)
    inputs[gt_path.name] = digest(gt_path)
    with meta_path.open(newline="", encoding="utf-8-sig") as handle:
        metadata = list(csv.DictReader(handle))
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    prefix = f"EPOCH_OBSERVATIONS/epoch_12/{split}"
    saved = gzip.decompress(archive.read(f"{prefix}/PER_QUESTION.csv.gz"))
    rows = []
    for row in csv.DictReader(io.StringIO(saved.decode("utf-8-sig"))):
        meta = metadata[int(row["source_row"])]
        question_id = f"{meta['video_id']}_{meta['qid']}"
        prediction, answer = int(row["prediction"]), int(row["answer"])
        if (row["public_question_id"], row["video_id"], row["type"]) != (
            question_id, meta["video_id"], meta["type"]
        ) or not (0 <= prediction < 5 and 0 <= answer < 5):
            raise ValueError(f"Metadata mismatch for {split} {question_id}")
        if meta[f"a{answer}"] != meta["answer"] or int(row["QA"]) != int(prediction == answer):
            raise ValueError(f"Answer mismatch for {split} {question_id}")
        if int(row["GQA"]) != int(row["QA"]) * int(row["IoP@0.5"]):
            raise ValueError(f"GQA mismatch for {split} {question_id}")
        markers = {match.lower() for match in MARKERS.findall(meta["question"])}
        marker = {frozenset(): "none", frozenset({"before"}): "before_only",
                  frozenset({"after"}): "after_only", frozenset({"before", "after"}): "both"}
        intervals = gt[row["video_id"]]["location"][meta["qid"]]
        rows.append({
            "split": split, "marker": marker[frozenset(markers)],
            "question": meta["question"], "gold_answer": meta["answer"],
            "predicted_answer": meta[f"a{prediction}"],
            **{f"a{i}": meta[f"a{i}"] for i in range(5)},
            "gt_intervals_s": json.dumps(intervals),
            **row,
        })
    if len(rows) != len(metadata) or len({row["source_row"] for row in rows}) != len(rows):
        raise ValueError(f"Incomplete one-to-one join for {split}")
    total = summarize(split, "all", rows)
    metric = json.loads(archive.read(f"{prefix}/METRICS.json"))
    for field, key in (("QA", "qa"), ("GQA", "gqa"), ("IoP@0.5", "iop50"), ("peak_hit", "peak_hit")):
        if abs(100 * total[key] / total["n"] - metric[field]) > 1e-9:
            raise ValueError(f"{split} aggregate mismatch: {field}")
    for key in ("mIoP", "mIoU"):
        if abs(total[f"{key}_pct"] - metric[key]) > 1e-9:
            raise ValueError(f"{split} aggregate mismatch: {key}")
    if (total["n"], total["videos"], total["qa"], total["gqa"]) != EXPECTED[split]:
        raise ValueError(f"Unexpected split totals: {total}")
    return rows, metric


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pilot", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    inputs = {args.archive.name: digest(args.archive)}
    all_rows = []
    summary = []
    metrics = {}
    with zipfile.ZipFile(args.archive) as archive:
        for split in EXPECTED:
            rows, metrics[split] = extract_split(archive, args.metadata_dir, split, inputs)
            all_rows.extend(rows)
            groups = {"all": rows, "tn": [], "non_tn": [], "before_or_after": []}
            for marker in ("before_only", "after_only", "both", "none"):
                groups[marker] = []
                groups[f"tn/{marker}"] = []
            for row in rows:
                marker = row["marker"]
                groups[marker].append(row)
                groups["tn" if row["type"] == "TN" else "non_tn"].append(row)
                if row["type"] == "TN":
                    groups[f"tn/{marker}"].append(row)
                if marker != "none":
                    groups["before_or_after"].append(row)
            summary.extend(summarize(split, name, group) for name, group in groups.items())
    write_csv(args.out / "EPOCH12_PER_QUESTION.csv", all_rows)
    write_csv(args.out / "EPOCH12_BEFORE_AFTER_SUMMARY.csv", summary)
    if args.pilot:
        for name in ("LOCKED.json", "RESULTS.json", "VERIFIED.json", "PER_QUESTION.csv"):
            shutil.copyfile(args.pilot / name, args.out / f"PILOT_{name}")
            inputs[f"pilot/{name}"] = digest(args.pilot / name)
    provenance = {
        "upstream_commit": COMMIT, "epoch": 12, "seed": 666, "checkpoint_sha256": EPOCH12_SHA,
        "checkpoint_bytes": 843915519, "inputs_sha256": inputs,
        "marker_pattern": MARKERS.pattern, "summary": summary, "archived_metrics": metrics,
    }
    (args.out / "EPOCH12_PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for row in summary:
        print(*(row[key] for key in ("split", "group", "n", "qa", "gqa", "iop50", "peak_hit",
                                    "qa_correct_iop_fail_peak_outside", "iop_zero", "tn")))


if __name__ == "__main__":
    main()
