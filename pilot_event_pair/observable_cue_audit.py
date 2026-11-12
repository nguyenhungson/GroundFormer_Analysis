"""Frozen-pool CPU audit. No model forward, training, or temporal labels in a scorer."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from attention_peak_pool import attention_pick, matched_windows, overlaps, peak_proposals
from evaluate import reference


ROOT = Path(__file__).resolve().parent
PROJECT = Path.home() / 'Downloads' / 'GroundFormer'
MEDIA = PROJECT / 'outputs/GroundFormer_Full_Video_Coverage_20261012/runs/GF_FULL_VAL_MEDIA_V1/media'
META = PROJECT / 'CMA_Error_Atlas_20261103/INPUTS/val.csv'
OUT = ROOT / 'Observable_Cue_Audit_20261109'
SEED = 'observable-cue-audit-20261109'
COUNT = 24
PROTOCOL = {
    'seed': SEED, 'n': COUNT, 'strata': 'none; sort by SHA256(seed, video id, question id), one pair per video with local media',
    'cohort': '367 QA-correct misses with baseline retained by attention and a passing same-geometry peak proposal',
    'pair': 'baseline versus highest-attention IoP-passing alternative; truth used for case selection only',
    'display': 'up to six evenly spaced sampled timestamps per window, seek in archived video; no timestamps or GT on sheets',
    'annotations': 'A/B only if a concrete cue from QA is visible there exclusively; BOTH, NEITHER or UNCLEAR otherwise',
    'uncertain': 'offscreen causes, unseen transition, poor detail, ambiguity; never force a choice',
    'reviewer_count': 1, 'scope': 'exploratory; one reviewer and media-available subset, VAL reused',
    'screen_for_second_review': 'at least 8/24 correct unique cues and at least twice as many as wrong unique cues',
    'gate_scope': 'positive screen merits a second independent review; does not validate a model or authorize a long training run',
}


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def write(name, value):
    target = OUT / name
    with target.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def extract(row):
    video = row['source']
    if hashlib.sha256(video.read_bytes()).hexdigest() != row['sha256']:
        raise ValueError(f'Video hash mismatch: {video.name}')
    bands = []
    for times in row['times']:
        positions = np.linspace(0, len(times)-1, min(6, len(times)), dtype=int)
        band = []
        for i in positions:
            args = ['ffmpeg', '-v', 'error', '-ss', str(times[i]), '-i', str(video), '-map', '0:v:0',
                    '-frames:v', '1', '-vf', 'scale=216:124:force_original_aspect_ratio=decrease',
                    '-threads', '1', '-c:v', 'png', '-f', 'image2pipe', 'pipe:1']
            decoded = subprocess.run(args, capture_output=True, check=False, timeout=25)
            if decoded.returncode:
                raise ValueError(decoded.stderr.decode(errors='replace'))
            data = decoded.stdout
            if not data:
                band.append(Image.new('RGB', (216,124), '#999999'))
                continue
            with Image.open(io.BytesIO(data)) as im:
                band.append(ImageOps.pad(im.convert('RGB'), (216,124), color='#151515'))
        bands.append(band)
    return bands


def prepare():
    OUT.mkdir(exist_ok=False)
    write('LOCKED.json', PROTOCOL)
    cfg, ref, rows = reference()
    meta_raw = META.read_bytes().replace(b'\r\n', b'\n')
    if hashlib.sha256(meta_raw).hexdigest() != cfg['val_csv_sha256']:
        raise ValueError('VAL csv hash mismatch')
    meta = {f"{r['video_id']}_{r['qid']}":r for r in csv.DictReader(io.StringIO(meta_raw.decode('utf-8-sig')))}
    hits = []
    for i, row in enumerate(rows):
        if row['answer'] != int(ref['prediction'][i]):
            continue
        ts, gt = row['timestamps'], row['GT_intervals']
        base, windows = matched_windows(ts, row['span'])
        if overlaps(base, gt)[1] >= .5:
            continue
        curve = ref['final_curves'][i]
        pool = [base, *(w for w, _ in peak_proposals(ts, base, windows, curve))]
        good = [w for w in pool[1:] if overlaps(w, gt)[1] >= .5]
        if good and attention_pick(pool, curve) == base:
            good_window = attention_pick(good, curve)
            assert math.isclose(base.end-base.start, good_window.end-good_window.start, abs_tol=1e-8)
            assert len(base.indices) == len(good_window.indices)
            hits.append((row, base, good_window))
    assert len(hits) == 367
    seen = set()
    for row, _, _ in hits:
        if (MEDIA/f"{row['video_id']}.mp4").is_file():
            seen.add(row['video_id'])
    available = sum(row['video_id'] in seen for row, _, _ in hits)
    distinct = len({r['video_id'] for r, _, _ in hits})
    selected = []
    taken = set()
    for row, base, good in sorted(hits, key=lambda x:digest(f"{SEED}|{x[0]['video_id']}|{x[0]['sample_id']}")):
        vid = row['video_id']
        if vid not in seen or vid in taken:
            continue
        taken.add(vid)
        number = len(selected)+1
        sid = row['sample_id']
        right = int(digest(f'{SEED}|blind|{sid}')[-1], 16) % 2
        windows = [base,good] if right else [good,base]
        video_metadata = json.loads((MEDIA.parent/'videos'/f'{vid}.json').read_bytes())
        assert video_metadata['video_id'] == vid and video_metadata['status'] == 'DECODED_AND_AUDITED'
        m = meta[sid]
        assert m['answer'] == m[f"a{row['answer']}"]
        selected.append({'case': f'C{number:02}', 'sid':sid, 'video':vid, 'question':m['question'], 'answer':m['answer'],
                         'type': m['type'], 'source':MEDIA/f'{vid}.mp4', 'sha256':video_metadata['video_sha256'],
                         'spans':[[w.start,w.end] for w in windows], 'frame_count':len(base.indices),
                         'times':[[row['timestamps'][i] for i in w.indices] for w in windows],
                         'GT':gt, 'winner':'AB'[right]})
        if number == COUNT:
            break
    assert len(selected) == COUNT
    write('UNBLIND_KEY.json', [{**r,'source':str(r['source'])} for r in selected])
    write('SAMPLING.json', {'cohort_questions':len(hits), 'cohort_videos':distinct,
                            'media_available_questions':available, 'media_available_videos':len(seen), 'sample':COUNT})
    write('BLIND_QUESTIONS.json', [{k:r[k] for k in ('case','question','answer')} for r in selected])
    render()


def render():
    selected = json.loads((OUT/'UNBLIND_KEY.json').read_bytes())
    for row in selected:
        row['source'] = Path(row['source'])
    font = ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf', 20)
    with ThreadPoolExecutor(max_workers=4) as executor:
        pictures = list(executor.map(extract, selected))
    for start in range(0,COUNT,4):
        sheet = Image.new('RGB',(1358, 4*344),'white')
        draw = ImageDraw.Draw(sheet)
        for j, (row, bands) in enumerate(zip(selected[start:start+4],pictures[start:start+4])):
            y = j*344
            lines = textwrap.wrap(f"{row['case']}    Q: {row['question']}    A: {row['answer']}", width=113)
            draw.multiline_text((15,y+6),'\n'.join(lines),font=font,fill='#1d2731')
            for band, label, pos in zip(bands, 'AB', (y+63,y+194)):
                draw.text((12,pos+48),label,font=font,fill='#1d2731')
                for i, im in enumerate(band):
                    sheet.paste(im,(43+218*i,pos))
            draw.line([(12,y+333),(1342,y+333)],fill='#ccd1d5',width=2)
        sheet.save(OUT/f'BLIND_{1+start//4}.jpg',quality=91)
    print(f'{COUNT} blind pairs ready')


def finalize():
    key = json.loads((OUT/'UNBLIND_KEY.json').read_bytes())
    annot = json.loads((OUT/'BLIND_ANNOTATIONS.json').read_bytes())
    assert len(annot)==COUNT and [r['case'] for r in key]==[a['case'] for a in annot]
    result = []
    for row, a in zip(key, annot):
        vote = a['vote']
        assert vote in ('A','B','BOTH','NEITHER','UNCLEAR') and a['cue'] and a['note']
        status = ('CORRECT_UNIQUE' if vote==row['winner'] else 'WRONG_UNIQUE') if vote in 'AB' else vote
        result.append({**{k:row[k] for k in ('case','sid','question','answer','spans','frame_count','winner')},**a,'result':status})
    counts = Counter(r['result'] for r in result)
    c, w = counts['CORRECT_UNIQUE'], counts['WRONG_UNIQUE']
    report = {'counts':counts, 'second_review_screen':c>=8 and c>=2*w,
              'reviewers':1, 'training_steps':0, 'gpu_minutes':0,
              'lock_sha256': hashlib.sha256((OUT/'LOCKED.json').read_bytes()).hexdigest(),
              'annotation_sha256':hashlib.sha256((OUT/'BLIND_ANNOTATIONS.json').read_bytes()).hexdigest()}
    write('RESULTS.json', report)
    with (OUT/'PER_PAIR.csv').open('x', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f,fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':
    {'prepare':prepare,'render':render,'finalize':finalize}[sys.argv[1]]()
