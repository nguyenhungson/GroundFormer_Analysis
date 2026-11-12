"""QA-only CPU prototype for reranking GroundFormer spans with masked visual words."""
from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import time

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from attention_peak_pool import attention_pick, matched_windows, overlaps, peak_proposals
from evaluate import reference
from observable_cue_audit import META, OUT, PROJECT


ROOT = Path(__file__).resolve().parent
TRAIN = ROOT.parent / 'e02_module_notebook_v1/official_full/datasets/nextgqa/train.csv'
H5_ROOT = PROJECT / 'outputs/GroundFormer_Teacher_10ep_20261013/CPU_INPUT_CACHE/CPU_RUN_ALL_INPUT_PREFLIGHT/data/features/CLIPL'
SAMPLE_SEED = 20261109
SEEDS = (666, 667, 668)
STOP = set('a an the to on in of at for with was were is are be do does did doing by they he she it his her this that from as and or if then than before after what why where when how many much video same again because get got take make look thing not no yes up down out over there here have had has all some one two three four five'.split())
GROUPS = '''remove removes removed removing
stand stands stood standing
sit sits sat sitting
fall falls fell falling
jump jumps jumped jumping
walk walks walked walking
run runs ran running
dance dances danced dancing
slide slides slid sliding
throw throws threw throwing
catch catches caught catching
kick kicks kicked kicking
clap claps clapped clapping
touch touches touched touching
hold holds held holding
lift lifts lifted lifting
drop drops dropped dropping
put puts putting
pick picks picked picking
open opens opened opening
close closes closed closing
turn turns turned turning
push pushes pushed pushing
pull pulls pulled pulling
ride rides rode riding
swing swings swung swinging
drink drinks drank drinking
eat eats ate eating
pour pours poured pouring
cut cuts cutting
wipe wipes wiped wiping
wash washes washed washing
wear wears wore wearing
point points pointed pointing
shake shakes shook shaking
wave waves waved waving
hit hits hitting
smile smiles smiled smiling
carry carries carried carrying
reach reaches reached reaching
roll rolls rolled rolling
climb climbs climbed climbing
lie lies lay lying
swim swims swam swimming
clean cleans cleaned cleaning
drive drives drove driving
play plays played playing
use uses used using'''
VERBS = {word: forms[0] for forms in map(str.split,GROUPS.splitlines()) for word in forms}
OBJECTS = set('phone controller helmet shoe shoes skate skates bicycle bike ball table chair bottle cup water bowl plate dog cat bird snake baby shirt dress umbrella car guitar rope screen box backpack bag hat glasses shovel snowboard skateboard toy book spoon knife pumpkin red blue green yellow white black brown orange purple pink maroon'.split())
WINDOWS = [(s,s+k-1) for k in (4,8,12,16,24) for s in range(0,33-k,4)]
LOCK = {
    'val': '2000 rows by SHA256 of fixed seed and sample ID across full VAL, then stable hash order, no eligibility resampling',
    'sample_seed': SAMPLE_SEED, 'seeds': SEEDS, 'steps_per_arm_per_seed': '2 full epochs on identical eligible train rows',
    'train': 'Official TRAIN questions and answers only; exclude both rows of the ambiguous duplicate key as in the prior audit; same train videos, examples, params and targets-per-example in both arms',
    'arm_cue': 'one canonical action or object/color from answer, otherwise question; repeated target occurrences masked',
    'arm_generic': 'different QA token closest by TRAIN log frequency; one canonical target, same vocab and all occurrences masked',
    'fallback': 'baseline span if no eligible pair of masks, no alternative or tied max',
    'pool_val': 'frozen peak proposal pool, baseline plus up to three alternatives; exact same geometry/num frames and QA fixed',
    'pool_train': 'GT-free windows across 32 frames, lengths 4/8/12/16/24, stride 4',
    'model': 'frozen CLIP, local window average plus endpoint difference; 64-wide learned projected visual and masked context; token softmax',
    'loss': 'per-video negative log mean of p(target|masked QA, window); no temporal label, not identifiable by construction',
    'primary': 'IoP05 on all 2000; compare cue to original, same-pool attention, generic mask; 98.333% video-bootstrap CI per contrast',
    'gate': 'all three lower bounds >0 in seed-averaged paired comparison and positive point differences on all three seeds; descriptive pilot on reused VAL',
    'sham': 'cue model with identical zero clip summaries; must keep baseline on every tie',
    'bootstrap': {'draws':10000,'seed':20261109,'unit':'video'},
}


def hash_text(text):
    return int(hashlib.sha256(text.encode()).hexdigest()[:16],16)


def read_csv(raw):
    return list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))


def words(text):
    return [VERBS.get(w,w) for w in re.findall(r'[a-z]+',text.lower())]


def cue(q, answer):
    for source in (answer,q):
        for choices in (set(VERBS.values()),OBJECTS):
            match = next((w for w in words(source) if w in choices),None)
            if match:
                return match
    return None


def context(q,a,target):
    values = [1+hash_text(f'{role}|{word if word!=target else "[mask]"}')%4095
              for role,text in [('question',q),('answer',a)] for word in words(text)]
    return (values[:48]+[0]*48)[:48]


def pack(q,a,vocab,freq):
    specific = cue(q,a)
    if specific not in vocab:
        return None
    options = set(words(q)+words(a)).intersection(vocab) - STOP - {specific}
    if not options:
        return None
    other = min(options,key=lambda w:(abs(math.log(freq[w]/freq[specific])),hash_text(f'{q}|{a}|{w}')))
    return (context(q,a,specific),vocab[specific]),(context(q,a,other),vocab[other])


class Scorer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.text = nn.Embedding(4096,64,padding_idx=0)
        self.visual = nn.Linear(1536,64)
        self.combine = nn.Sequential(nn.Linear(192,64),nn.GELU(),nn.Linear(64,vocab_size))

    def forward(self, clips, context_ids):
        text = self.text(context_ids).sum(dim=1) / (context_ids!=0).sum(dim=1,keepdim=True).clamp_min(1)
        text = text[:,None,:].expand(-1,clips.shape[1],-1)
        vision = self.visual(clips)
        return self.combine(torch.cat([vision,text,vision*text],dim=-1))


def h5(path, expected=None):
    sha = hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
    if expected is not None and sha!=expected:
        raise ValueError(f'H5 hash mismatch: {sha}')
    with h5py.File(path) as f:
        ids = [x.decode() for x in f['vid'][:]]
        x = torch.from_numpy(f['CLIPL_I'][:])
    assert len(set(ids))==len(ids) and x.shape==(len(ids),32,768)
    return {v:i for i,v in enumerate(ids)}, x, sha


def summaries(x, starts, ends):
    assert torch.all(ends>=starts)
    start = starts[None,:].expand(x.shape[0],-1) if starts.ndim==1 else starts
    end = ends[None,:].expand(x.shape[0],-1) if ends.ndim==1 else ends
    batch = torch.arange(x.shape[0])[:,None]
    prefix = F.pad(x.cumsum(dim=1), (0,0,1,0))
    mean = (prefix[batch,end+1]-prefix[batch,start]) / (end-start+1)[...,None]
    return torch.cat([mean,x[batch,end]-x[batch,start]],dim=-1)


def evaluate_arm(model, visual, cx, cy, valid):
    assert visual.shape[1]==valid.shape[1]==4
    with torch.inference_mode():
        scores = []
        for inds in torch.arange(len(cx)).split(128):
            logits = model(visual[inds],cx[inds])
            scores.append(logits.log_softmax(-1).gather(-1,cy[inds,None,None].expand(-1,4,1)).squeeze(-1))
        scores = torch.cat(scores).masked_fill(~valid, -torch.inf)
        # Near ties use the baseline. A clip-free arm therefore has no switches.
        best, ids = scores.max(dim=1)
        ties = ((scores-best[:,None]).abs()<=1e-6).sum(dim=1)>1
        return torch.where(ties,0,ids).numpy()


def run():
    torch.set_num_threads(5)
    work = OUT/'CPU_2000_V2'
    work.mkdir(exist_ok=False)
    lock_path = work/'LOCKED.json'
    lock_path.write_text(json.dumps(LOCK,indent=2),encoding='utf-8')
    t0 = time.monotonic()
    cfg, ref, rows = reference()
    raw = META.read_bytes().replace(b'\r\n',b'\n')
    assert hashlib.sha256(raw).hexdigest()==cfg['val_csv_sha256']
    meta = {f"{r['video_id']}_{r['qid']}":r for r in read_csv(raw)}
    train_raw = TRAIN.read_bytes()
    train_rows = read_csv(train_raw)
    key_counts = Counter((r['video_id'],r['qid']) for r in train_rows)
    train_unique = {(r['video_id'],r['qid']):r for r in train_rows if key_counts[r['video_id'],r['qid']]==1}
    assert len(train_unique)==34130 and not {r['video_id'] for r in train_unique.values()}.intersection(r['video_id'] for r in rows)
    freq = Counter(w for r in train_unique.values() for w in words(r['question'])+words(r['answer']) if w not in STOP)
    voc = {w:i for i,w in enumerate(sorted(freq,key=lambda w:(-freq[w],w))[:768])}
    train_vid, train_features, train_sha = h5(H5_ROOT/'CLIPL_I_train.h5')
    val_vid, val_features, val_sha = h5(H5_ROOT/'CLIPL_I_val.h5',cfg['val_h5_sha256'])
    assert len(train_vid)==3870 and len(val_vid)==570
    t_rows = []
    for r in train_unique.values():
        target = pack(r['question'],r['answer'],voc,freq)
        if target is not None:
            t_rows.append((train_vid[r['video_id']],target))
    tr = torch.tensor([v for v,_ in t_rows])
    tx = [torch.tensor([p[i][0] for _,p in t_rows]) for i in range(2)]
    ty = [torch.tensor([p[i][1] for _,p in t_rows]) for i in range(2)]
    v_x = [[],[]]; v_y = [[],[]]; v_mask = []; v_videos = []; v_bounds = []; records = []
    chosen = sorted(range(len(rows)),key=lambda i:hash_text(f'{SAMPLE_SEED}|{rows[i]["sample_id"]}'))[:2000]
    for i in chosen:
        r = rows[i]
        sid,vid = r['sample_id'],r['video_id']
        m = meta[sid]
        p = int(ref['prediction'][i])
        target = pack(m['question'],m[f'a{p}'],voc,freq)
        found = target is not None
        orig, eligible = matched_windows(r['timestamps'],r['span'])
        curve = ref['final_curves'][i]
        candidates = [orig,*(w for w,_ in peak_proposals(r['timestamps'],orig,eligible,curve))]
        assert all(len(w.indices)==len(orig.indices) and math.isclose(w.end-w.start,orig.end-orig.start,abs_tol=1e-8) for w in candidates)
        attention = candidates.index(attention_pick(candidates,curve))
        bound = [(w.indices[0],w.indices[-1]) for w in candidates]
        iop = [overlaps(w,r['GT_intervals']) for w in candidates]
        qa = p==r['answer']
        if not found:
            target = (([0]*48,0),([0]*48,0))
        for j in range(2):
            v_x[j].append(target[j][0]); v_y[j].append(target[j][1])
        v_mask.append([k<len(bound) and (found or k==0) for k in range(4)])
        v_bounds.append((bound+[(0,0)]*4)[:4]); v_videos.append(val_vid[vid])
        records.append({'id':sid,'video':vid,'qa':qa,'has_masks':found,'attention_index':attention,
                        'spans':[[w.start,w.end] for w in candidates], 'metrics':iop,
                        'target_case':qa and iop[0][1]<.5 and any(x[1]>=.5 for x in iop[1:]) and attention==0})
    (work/'SAMPLE_IDS.json').write_text(json.dumps([r['id'] for r in records]),encoding='utf-8')
    v_mask = torch.tensor(v_mask); vb = torch.tensor(v_bounds)
    v_features = summaries(F.normalize(val_features[v_videos],dim=-1)*math.sqrt(768),vb[:,:,0],vb[:,:,1]); del val_features
    v_x = [torch.tensor(x) for x in v_x]; v_y = [torch.tensor(y) for y in v_y]
    with torch.no_grad():
        t_features = summaries(F.normalize(train_features,dim=-1)*math.sqrt(768),*torch.tensor(WINDOWS).T)
    del train_features
    print(f'locked VAL 2000, {len(t_rows)} train QA, {len(voc)} vocab, {int(v_mask[:,1].sum())} active VAL',flush=True)
    predictions = {'baseline':np.zeros(2000,dtype=int),'attention':np.array([r['attention_index'] for r in records])}
    for seed in SEEDS:
        for j,arm in enumerate(('cue','generic')):
            torch.manual_seed(seed)
            rng = torch.Generator().manual_seed(seed)
            model = Scorer(len(voc))
            opt = torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01)
            model.train()
            for epoch in range(2):
                loss_sum = 0.0
                for index in torch.randperm(len(tr),generator=rng).split(64):
                    logits = model(t_features[tr[index]],tx[j][index])
                    logp = logits.log_softmax(-1).gather(-1,ty[j][index,None,None].expand(-1,len(WINDOWS),1)).squeeze(-1)
                    loss = (math.log(len(WINDOWS))-logp.logsumexp(dim=1)).mean()
                    opt.zero_grad(); loss.backward(); opt.step()
                    loss_sum += float(loss.detach())*len(index)
                print(f'{seed} {arm} epoch {epoch+1}, TRAIN loss={loss_sum/len(tr):.4f}, wall={time.monotonic()-t0:.0f}s',flush=True)
            model.eval()
            predictions[f'{arm}_{seed}'] = evaluate_arm(model,v_features,v_x[j],v_y[j],v_mask)
            if j==0:
                sham = evaluate_arm(model,torch.zeros_like(v_features),v_x[j],v_y[j],v_mask)
                assert np.array_equal(sham,predictions['baseline'])
    calculate(work,records,predictions,{'train_sha256':hashlib.sha256(train_raw).hexdigest(),'train_h5_sha256':train_sha,
              'val_h5_sha256':val_sha,'lock_sha256':hashlib.sha256(lock_path.read_bytes()).hexdigest(),
              'train_eligible':len(t_rows),'parameters':sum(p.numel() for p in model.parameters()),
              'val_active':int(v_mask[:,1].sum()),'wall_s':round(time.monotonic()-t0,1)})


def calculate(work,records,predictions,metadata):
    qa = np.array([r['qa'] for r in records]); target = np.array([r['target_case'] for r in records])
    unique,v_index = np.unique([r['video'] for r in records],return_inverse=True)
    draws = np.random.default_rng(SAMPLE_SEED).integers(len(unique),size=(10000,len(unique)))
    denom = np.bincount(v_index)[draws].sum(axis=1)
    scores = {}; hits = {}; base = None
    for name,p in predictions.items():
        values = np.array([r['metrics'][i] for r,i in zip(records,p)])
        iou,iop = values.T; hit = iop>=.5; hits[name] = hit
        if base is None:
            base = hit
        scores[name] = {'IoP05':int(hit.sum()), 'GQA':int((hit&qa).sum()), 'mIoP_pct':100*float(iop.mean()),
                        'mIoU_pct':100*float(iou.mean()),'IoU05':int((iou>=.5).sum()),
                        'rescued':int((hit&~base).sum()),'lost':int((~hit&base).sum()),
                        'switches':int((p!=0).sum()),'target_rescued':int((hit&target).sum())}
    ci = {}
    for control in ('baseline','attention','generic'):
        changes = np.array([hits[f'cue_{s}'].astype(float)-hits[f'generic_{s}' if control=='generic' else control].astype(float) for s in SEEDS])
        delta = changes.mean(axis=0)
        dist = np.bincount(v_index,weights=delta)[draws].sum(axis=1)*100/denom
        ci[control] = {'pp':100*float(delta.mean()),'seed_pp':(100*changes.mean(axis=1)).tolist(),
                       'CI98_333':np.quantile(dist,[.05/6,1-.05/6]).tolist()}
    result = dict(metadata,N=len(records),videos=len(unique),QA=int(qa.sum()),target_n=int(target.sum()),
                  metrics=scores,contrasts=ci,screen=all(x['CI98_333'][0]>0 and min(x['seed_pp'])>0 for x in ci.values()),gpu_minutes=0)
    (work/'RESULTS.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    with (work/'PER_QUESTION.csv').open('w',newline='',encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(['sample_id','video_id','QA','has_masks','target_case','candidate_spans','candidate_IoU_IoP',*predictions])
        for i,r in enumerate(records):
            writer.writerow([r['id'],r['video'],int(r['qa']),int(r['has_masks']),int(r['target_case']),r['spans'],r['metrics'],*(p[i] for p in predictions.values())])
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    run()
