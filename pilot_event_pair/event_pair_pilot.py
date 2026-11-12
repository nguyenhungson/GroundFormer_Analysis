"""CPU, TRAIN QA-only, frozen GroundFormer cache, fixed 2,000-question VAL."""
import ast
from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from event_pair_parser import parse
from observable_cue_probe import H5_ROOT, TRAIN, h5, hash_text
from evaluate import reference


ROOT = Path(__file__).resolve().parent
PROJECT = Path.home()/'Downloads/GroundFormer'
REF = PROJECT/'Observable_Cue_Audit_20261109/CPU_2000_V2'
META = PROJECT/'CMA_Error_Atlas_20261103/INPUTS/val.csv'
ROBERTA = Path.home()/'.cache/huggingface/hub/models--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b'
OUT = ROOT/'Event_Pair_Pilot_20261110'
ARMS = ('independent','raw_pair','role_swap')
SEEDS = (666,667,668)
ANCHORS = [(s,s+k-1) for k in (2,4,8) for s in range(33-k)]
TRAIN_A = [b for b in ANCHORS if b[0]%4==0]
TRAIN_T = [(s,s+k-1) for k in (4,8,12,16,24) for s in range(0,33-k,4)]
LOCK = dict(
    sample='exact 2,000 IDs of preceding addon pilot, same cache predictions and candidate spans, no TEST',
    features='canonical CLIP-L H5, SHA verified; frozen local RoBERTa-base, masked mean pooling',
    parser='single explicit before/after, known actor, five distinct action clauses; deterministic regex and verb list',
    fit='TRAIN only, exclude both ambiguous duplicate-key rows; reserve hash(video ID, 20261110) % 5 == 0 for calibration',
    minimum={'fit_questions':100,'cal_questions':30,'evaluable_val_without_gt':40},
    train_pool='GT-free windows, anchors 2/4/8 frames stride4, targets 4/8/12/16/24 stride4; cap 32 evenly spaced valid ordered pairs',
    fit_details={'epochs':2,'batch':12,'lr':.001,'optimizer':'AdamW','seeds':SEEDS,'threads':5},
    score='J=min(two role-specific cosine margins)+residual; raw h, two isolated h calls mean, or antisymmetric four-call D',
    infer='fixed anchor via cosine, no overlap with baseline or alternative; preserve same extent and original QA',
    calibration='TRAIN held-out mismatch text and video by nearest within same alternative-count bin; threshold q95 of max nonnegative gated gap, same group size; require two positive margins and positive residual for each arm',
    controls=['baseline','order-only same anchor as full','matched independent role scores','matched raw pair','same-video displaced anchor','reverse local feature order'],
    decision='IoP05: role_swap must beat baseline, independent and raw_pair in seed-mean and all individual seeds; paired video-bootstrap CI98.333 lower bounds above 0; GQA and rescued/lost secondary',
    bootstrap={'n':10000,'seed':20261110,'unit':'video'},
)


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def rows_from(data):
    return list(csv.DictReader(io.StringIO(data.decode('utf-8-sig'))))


def lookup(m):
    return parse(m['question'],[m[f'a{i}'] for i in range(5)])


def masks(ts,spans):
    out=[]
    for start,end in spans:
        indexes=np.flatnonzero((ts>=start-1e-7)&(ts<=end+1e-7))
        assert len(indexes)>0 and indexes[-1]-indexes[0]+1==len(indexes)
        out.append((int(indexes[0]),int(indexes[-1])))
    assert len(set(out))==len(out)
    return out


def gather_sources():
    cfg, ref, native = reference()
    meta_raw = META.read_bytes().replace(b'\r\n',b'\n')
    assert hashlib.sha256(meta_raw).hexdigest()==cfg['val_csv_sha256']
    meta = {f"{r['video_id']}_{r['qid']}":r for r in rows_from(meta_raw)}
    native_idx = {r['sample_id']:i for i,r in enumerate(native)}
    sample = json.loads((REF/'SAMPLE_IDS.json').read_bytes())
    # Read identifiers and proposal boundaries only. Ignore annotation-derived columns until scoring.
    frozen = rows_from((REF/'PER_QUESTION.csv').read_bytes())
    assert sample==[r['sample_id'] for r in frozen] and len(set(sample))==2000
    val=[]
    for old in frozen:
        sid=old['sample_id']; i=native_idx[sid]; r=native[i]; m=meta[sid]; vid=r['video_id']
        assert vid==old['video_id']
        events=lookup(m)
        pred=int(ref['prediction'][i])
        span=ast.literal_eval(old['candidate_spans'])
        assert span[0]==r['span']
        bounds=masks(np.asarray(r['timestamps']),span)
        val.append(dict(sid=sid,vid=vid,spans=span,bounds=bounds,events=events,pred=pred,marked=any(x in {'after','before'} for x in m['question'].lower().split())))
    raw=TRAIN.read_bytes(); tr=rows_from(raw)
    counter=Counter((r['video_id'],r['qid']) for r in tr)
    train=[]; cal=[]; n=0
    for r in tr:
        vid=r['video_id']
        if counter[vid,r['qid']]!=1:
            continue
        n+=1
        events=lookup(r)
        options=[r[f'a{i}'] for i in range(5)]
        if events is None or options.count(r['answer'])!=1:
            continue
        entry=dict(sid=vid+'_'+r['qid'],vid=vid,events=events,pred=options.index(r['answer']))
        (cal if hash_text('20261110|'+vid)%5==0 else train).append(entry)
    assert n==34130 and not ({r['vid'] for r in val}&{r['vid'] for r in cal+train})
    assert not ({r['vid'] for r in train}&{r['vid'] for r in cal})
    cover=dict(train_total=n,fit=len(train),cal=len(cal),fit_videos=len({r['vid'] for r in train}),cal_videos=len({r['vid'] for r in cal}),
               val=len(val),val_parsed=sum(r['events'] is not None for r in val),
               val_can_compare=sum(r['events'] is not None and len(r['bounds'])>1 for r in val),
               train_csv_sha=hashlib.sha256(raw).hexdigest())
    return cfg,train,cal,val,frozen,cover


def embeddings(entries):
    from transformers import AutoModel, AutoTokenizer
    sentences=sorted({t for r in entries if r['events'] for t in r['events'][1:]})
    tok=AutoTokenizer.from_pretrained(ROBERTA,local_files_only=True)
    model=AutoModel.from_pretrained(ROBERTA,local_files_only=True,add_pooling_layer=False).eval()
    z=[]
    with torch.inference_mode():
        for i in range(0,len(sentences),48):
            x=tok(sentences[i:i+48],padding=True,truncation=True,max_length=48,return_tensors='pt')
            h=model(**x).last_hidden_state
            mask=x['attention_mask'].unsqueeze(-1)
            z.append((h*mask).sum(1)/mask.sum(1))
            if i%960==0:
                print(f'Text encoded {min(i+48,len(sentences))}/{len(sentences)}',flush=True)
    bank=torch.cat(z)
    ids={s:i for i,s in enumerate(sentences)}
    for r in entries:
        if r['events']:
            r['text_ids']=[ids[t] for t in r['events'][1:]]
    return bank, len(sentences)


def ordered(anchor,target,rel):
    return anchor[1]<target[0] if rel=='after' else target[1]<anchor[0]


def pairs(rel):
    p=[(i,j) for i,a in enumerate(TRAIN_A) for j,t in enumerate(TRAIN_T) if ordered(a,t,rel)]
    return [p[k] for k in np.linspace(0,len(p)-1,32,dtype=int)]


TRAIN_PAIRS={r:torch.tensor(pairs(r)) for r in ('after','before')}


class Ranker(nn.Module):
    def __init__(self,arm):
        super().__init__()
        self.arm=arm
        self.v=nn.Linear(768,128,bias=False)
        self.t=nn.Linear(768,128,bias=False)
        self.conv=nn.Conv1d(128,128,3,padding=1,groups=128)
        self.h=nn.Sequential(nn.Linear(256,128),nn.GELU(),nn.Linear(128,1),nn.Tanh())

    def windows(self,frames,bounds,reverse=False):
        # Project all frames once, and run convolution separately within each window.
        x=self.v(F.normalize(frames,dim=-1)*math.sqrt(768))
        spans=torch.as_tensor(bounds,dtype=torch.long)
        lengths=spans[:,1]-spans[:,0]+1
        d=torch.arange(int(lengths.max()))
        mask=d[None,:]<lengths[:,None]
        ix=spans[:,1,None]-d if reverse else spans[:,0,None]+d
        clip=x[:,ix.clamp(0,31),:]*mask[None,:,:,None]
        b,w,k,_=clip.shape
        local=F.gelu(self.conv(clip.reshape(b*w,k,128).transpose(1,2))).transpose(1,2).reshape(b,w,k,128)
        return F.normalize(((clip+local)*mask[None,:,:,None]).sum(2)/lengths[None,:,None],dim=-1)

    def cmp(self,u,w,a,t):
        ua,wt,ut,wa = u*a,w*t,u*t,w*a
        da,dt=(ua-ut).sum(-1),(wt-wa).sum(-1)
        def h(x,y):
            return self.h(torch.cat([x,y],dim=-1)).squeeze(-1)
        if self.arm=='role_swap':
            residual=.5*(h(ua,wt)-h(ut,wa)-h(wa,ut)+h(wt,ua))
        elif self.arm=='raw_pair':
            residual=h(ua,wt)
        else:
            residual=.5*(h(ua,torch.zeros_like(wt))+h(torch.zeros_like(ua),wt))
        return torch.minimum(da,dt)+residual, da, dt, residual

    def train_logits(self,frames,text,rels):
        z=self.windows(frames,TRAIN_A+TRAIN_T)
        t=F.normalize(self.t(text),dim=-1)
        index=torch.stack([TRAIN_PAIRS[r] for r in rels])
        batch=torch.arange(len(frames))[:,None]
        u=z[batch,index[:,:,0]][:,:,None,:].expand(-1,-1,5,-1)
        w=z[batch,index[:,:,1]+len(TRAIN_A)][:,:,None,:].expand_as(u)
        a=t[:,None,:1,:].expand_as(u)
        e=t[:,None,1:,:].expand_as(u)
        j,*_=self.cmp(u,w,a,e)
        return j.logsumexp(1)-math.log(j.shape[1])


def fit_ranker(train,bank,features,video_index,arm,seed,epochs=2,on_epoch=None):
    torch.manual_seed(seed)
    g=torch.Generator().manual_seed(seed)
    model=Ranker(arm)
    optim=torch.optim.AdamW(model.parameters(),lr=.001)
    all_video=torch.tensor([video_index[r['vid']] for r in train])
    txt=bank[[r['text_ids'] for r in train]]
    targets=torch.tensor([r['pred'] for r in train])
    losses=[]
    for ep in range(epochs):
        total=0.
        for ii in torch.randperm(len(train),generator=g).split(12):
            logits=model.train_logits(features[all_video[ii]],txt[ii],[train[i]['events'][0] for i in ii])
            loss=F.cross_entropy(logits,targets[ii])
            optim.zero_grad(); loss.backward(); optim.step()
            total+=float(loss.detach())*len(ii)
        losses.append(total/len(train))
        print(f'{seed} {arm} TRAIN epoch {ep+1}: loss {losses[-1]:.4f}',flush=True)
        if on_epoch is not None:
            on_epoch(ep+1,model,losses[-1])
    return model.eval(), losses


def evaluate_one(model,frames,text,row,threshold=None,control='normal'):
    spans=row['bounds']; rel=row['events'][0]
    z=model.windows(frames[None],ANCHORS+spans,reverse=control=='reverse')[0]
    t=F.normalize(model.t(text),dim=-1)
    a,e=t[0],t[row['pred']+1]
    strength=z[:len(ANCHORS)]@a
    ix=int(strength.argmax()); u=ANCHORS[ix]
    if control=='displaced':
        distant=[j for j,an in enumerate(ANCHORS) if len(range(an[0],an[1]+1))==len(range(u[0],u[1]+1)) and (an[1]<u[0] or u[1]<an[0])]
        if not distant:
            return 0,0.,'no_displaced_anchor',ix
        ix=min(distant,key=lambda j:hash_text(f"{row['sid']}|{j}")); u=ANCHORS[ix]
    if not (u[1]<spans[0][0] or spans[0][1]<u[0]):
        return 0,0.,'anchor_overlaps_original',ix
    choices=[i for i,w in enumerate(spans) if i and ordered(u,w,rel)]
    if not choices:
        return 0,0.,'no_ordered_alternative',ix
    if control=='order':
        if ordered(u,spans[0],rel):
            return 0,0.,'original_ordered',ix
        j=min(choices,key=lambda k:(abs(sum(spans[k])-sum(u)),k))
        return j,0.,'switch_order',ix
    j,da,dt,res=model.cmp(z[ix].expand(len(spans),-1),z[len(ANCHORS):],a.expand(len(spans),-1),e.expand(len(spans),-1))
    eligible=[i for i in choices if da[i]>0 and dt[i]>0 and res[i]>0 and j[i]>j[0]]
    if not eligible:
        return 0,0.,'role_check_failed',ix
    best=max(eligible,key=lambda i:(float(j[i]),-i)); gap=float(j[best]-j[0])
    if threshold is not None and gap<=threshold:
        return 0,gap,'margin_below_threshold',ix
    return best,gap,'switch',ix


def calibration(model,cal,bank,features,vids):
    gaps=[]; attempted=Counter()
    for r in cal:
        v=features[vids[r['vid']]]
        donor_pool=[x for x in cal if x['vid']!=r['vid'] and x['events'][0]==r['events'][0]]
        if not donor_pool:
            continue
        donor=donor_pool[hash_text(r['sid']+'|null')%len(donor_pool)]
        t=bank[donor['text_ids']]
        targets=sorted(TRAIN_T,key=lambda w:(abs((w[1]-w[0]+1)-8),w[0]))
        for k in (1,2,3):
            # Four predetermined, same-width positions. No TRAIN GF span cache is assumed.
            bounds=[(s,s+7) for s in (8,0,16,24)[:k+1]]
            row={**r,'bounds':bounds,'pred':donor['pred']}
            _,gap,_,_=evaluate_one(model,v,t,row)
            gaps.append((k,gap)); attempted[k]+=1
    return {k:float(np.quantile([x for n,x in gaps if n==k],.95,method='higher')) for k in (1,2,3)},attempted


def score_results(val,frozen,choices,diagnostics,receipt):
    values = [ast.literal_eval(r['candidate_IoU_IoP']) for r in frozen]
    qa=np.asarray([r['QA']=='1' for r in frozen]); tagged=np.asarray([r['events'] is not None for r in val]);
    vids,vid=np.unique([r['vid'] for r in val],return_inverse=True)
    draws=np.random.default_rng(20261110).integers(len(vids),size=(10000,len(vids)))
    denom=np.bincount(vid)[draws].sum(1)
    stats={}; flags={}
    for arm,sel in choices.items():
        iou,iop=np.asarray([m[int(i)] for m,i in zip(values,sel)]).T
        hits=iop>=.5; flags[arm]=hits
        base=flags['baseline']
        stats[arm]=dict(iop05=int(hits.sum()),gqa=int((hits&qa).sum()),miop=100*float(iop.mean()),miou=100*float(iou.mean()),
                        iou05=int((iou>=.5).sum()),switches=int(np.count_nonzero(sel)),rescued=int((hits&~base).sum()),
                        lost=int((base&~hits).sum()),gqa_rescued=int((hits&~base&qa).sum()),gqa_lost=int((base&~hits&qa).sum()),
                        parsed_hits=int((hits&tagged).sum()),parsed_gqa=int((hits&tagged&qa).sum()))
    assert stats['baseline']['iop05']==589 and stats['baseline']['gqa']==388 and qa.sum()==1204
    contrasts={}
    for control in ('baseline','independent','raw_pair'):
        diffs=np.stack([flags[f'role_swap_{s}'].astype(float)-flags['baseline' if control=='baseline' else f'{control}_{s}'] for s in SEEDS])
        avg=diffs.mean(0)
        dist=100*np.bincount(vid,weights=avg)[draws].sum(1)/denom
        contrasts[control]=dict(pp=float(100*avg.mean()),seeds=(100*diffs.mean(1)).tolist(),ci98_333=np.quantile(dist,[.05/6,1-.05/6]).tolist())
    potential=np.asarray([m[0][1]<.5 and any(x[1]>=.5 for x in m[1:]) for m in values]);
    only_base=np.asarray([len(m)>1 and m[0][1]>=.5 and not any(x[1]>=.5 for x in m[1:]) for m in values])
    result=dict(coverage=receipt,metrics=stats,contrasts=contrasts,parse_recoverable_iop=int((tagged&potential).sum()),
                parse_recoverable_gqa=int((tagged&potential&qa).sum()),parse_only_original=int((tagged&only_base).sum()),
                gate=all(c['ci98_333'][0]>0 and min(c['seeds'])>0 for c in contrasts.values()),GPU_minutes=0)
    write(OUT/'RESULTS.json',result)
    with (OUT/'PER_QUESTION.csv').open('w',encoding='utf-8-sig',newline='') as f:
        out=csv.writer(f); out.writerow(['sample_id','video','parsed','QA','relation','anchor_clause','target_clause','spans','IoU_IoP',*choices,*[k+'_diag' for k in diagnostics]])
        for i,r in enumerate(val):
            e=r['events']; out.writerow([r['sid'],r['vid'],bool(e),int(qa[i]),e[0] if e else '',e[1] if e else '',e[r['pred']+2] if e else '',
                r['spans'],values[i],*[s[i] for s in choices.values()],*[d[i] for d in diagnostics.values()]])
    print(json.dumps(result,ensure_ascii=False),flush=True)


def main():
    start=time.monotonic(); torch.set_num_threads(5)
    OUT.mkdir(exist_ok=False)
    write(OUT/'LOCKED.json',LOCK)
    cfg,tr,cal,val,old,cover=gather_sources()
    write(OUT/'COVERAGE.json',cover)
    print(json.dumps(cover),flush=True)
    minimum=LOCK['minimum']
    if cover['fit']<minimum['fit_questions'] or cover['cal']<minimum['cal_questions'] or cover['val_can_compare']<minimum['evaluable_val_without_gt']:
        print('STOP_DATA_COVERAGE',flush=True)
        return
    train_idx,train_v,train_hash=h5(H5_ROOT/'CLIPL_I_train.h5','666e39260448b2c8773e2f3f0d6852e43b9b5573f7db6cb0971cd77b95eac0e2')
    val_idx,val_v,val_hash=h5(H5_ROOT/'CLIPL_I_val.h5',cfg['val_h5_sha256'])
    bank,n_text=embeddings(tr+cal+val)
    choices={'baseline':np.zeros(len(val),dtype=int),'attention':np.asarray([int(r['attention']) for r in old])}
    diagnostics={}; trains={}
    for seed in SEEDS:
        for arm in ARMS:
            model,loss=fit_ranker(tr,bank,train_v,train_idx,arm,seed)
            label=f'{arm}_{seed}'
            with torch.inference_mode():
                thresholds,ncal=calibration(model,cal,bank,train_v,train_idx)
                trains[label]=dict(train_loss=loss,thresholds=thresholds,null_counts=ncal,params=sum(p.numel() for p in model.parameters()))
                print(f'{label} locked TRAIN calibration: {thresholds}',flush=True)
                controls=['normal','order','displaced','reverse'] if arm=='role_swap' else ['normal']
                for control in controls:
                    tag=label if control=='normal' else f'{control}_{seed}'
                    pred=[]; detail=[]
                    for r in val:
                        k=len(r['bounds'])-1
                        if not r['events'] or not k:
                            pred.append(0); detail.append('parser_or_pool'); continue
                        i,gap,reason,a=evaluate_one(model,val_v[val_idx[r['vid']]],bank[r['text_ids']],r,thresholds[k],control)
                        pred.append(i); detail.append(f'{reason}|{a}|{gap:.6g}')
                    choices[tag]=np.asarray(pred)
                    diagnostics[tag]=detail
    cover.update(wall_s=round(time.monotonic()-start,2),train_h5=train_hash,val_h5=val_hash,text_count=n_text,models=trains,
                 lock_sha256=hashlib.sha256((OUT/'LOCKED.json').read_bytes()).hexdigest())
    score_results(val,old,choices,diagnostics,cover)


if __name__=='__main__':
    main()
