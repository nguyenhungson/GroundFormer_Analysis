"""Future authorized VAL-only session. NEVER called by the offline preparation."""
import argparse
import contextlib
import csv
import functools
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from core import ARMS, EXPECTED_SHA, digest, require_authorization, choose_donors, curve_metrics

HERE=Path(__file__).resolve().parent
RUNTIME_PACKAGES=('numpy','torch','transformers','h5py','pandas','matplotlib')

def write_json(path,data):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8');tmp.replace(path)

def read_bound(p,expected):
    raw=Path(p).read_bytes();actual=digest(raw)
    if actual!=expected:raise RuntimeError(f'INPUT_HASH_MISMATCH:{p}:{actual}')
    return raw

def save_npz(path,data,fingerprint):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    buf=io.BytesIO();np.savez_compressed(buf,**data);raw=buf.getvalue()
    tmp=path.with_suffix('.tmp');tmp.write_bytes(raw);tmp.replace(path)
    write_json(path.with_suffix('.json'),{'sha256':digest(raw),'bytes':len(raw),'run_fingerprint':fingerprint})

def open_npz(path,fingerprint):
    path=Path(path);entry=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    if entry['run_fingerprint']!=fingerprint:raise RuntimeError('SHARD_RUN_FINGERPRINT_MISMATCH')
    raw=read_bound(path,entry['sha256'])
    with np.load(io.BytesIO(raw),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def model_runtime(cfg,checkpoint,csv_raw,h5_raw,rows):
    import torch
    import h5py
    import transformers
    from transformers import RobertaTokenizerFast
    from model.TempVLM import VQA
    from dataloader.vqa_loader import videoqa_collate_fn
    from util import tokenize
    if not torch.cuda.is_available():raise RuntimeError('CUDA_REQUIRED_FOR_AUTHORIZED_SESSION')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(666)
    tokenizer=RobertaTokenizerFast.from_pretrained('roberta-base')
    model=VQA(bert_tokenizer=tokenizer,feature_dim=768,word_dim=768,N=2,h=8,d_model=768,d_ff=768,
              dropout=0.,Q=30,T=32,vocab_size=tokenizer.vocab_size,baseline='dual',feat_type='CLIPL',
              prop_num=1,sigma=9,n_qtypes=8,token_router='none')
    state=torch.load(io.BytesIO(checkpoint),map_location='cpu',weights_only=True)
    if isinstance(state,dict) and 'model_state_dict' in state:state=state['model_state_dict']
    if state and all(k.startswith('module.') for k in state):state={k[7:]:v for k,v in state.items()}
    model.load_state_dict(state,strict=True);del state,checkpoint
    model.eval();model.requires_grad_(False);model.float().cuda()
    features={}
    with h5py.File(io.BytesIO(h5_raw),'r') as f:
        wanted={r['video_id'] for r in rows.values()}
        for i,key in enumerate(f['vid']):
            vid=key.decode() if isinstance(key,bytes) else str(key)
            if vid not in wanted:continue
            a=np.asarray(f['CLIPL_I'][i],dtype=np.float32)
            if a.ndim!=2 or a.shape[1]!=768 or len(a)==0:raise RuntimeError('INVALID_FEATURE_SHAPE')
            features[vid]=a[np.linspace(0,len(a)-1,32,dtype=int)]
        if set(features)!=wanted:raise RuntimeError('VAL_FEATURE_COVERAGE_FAIL')
    native_rows=list(csv.DictReader(io.StringIO(csv_raw.decode('utf-8-sig'))));tokens={}
    for start in range(0,len(native_rows),64):
        samples=[];block=native_rows[start:start+64]
        for r in block:
            texts=[r['question']+' '+tokenizer.sep_token+' '+r[f'a{i}'] for i in range(5)]
            qa,_=tokenize(texts,tokenizer,add_special_tokens=True,max_length=38,dynamic_padding=False,truncation=True)
            q=torch.tensor(tokenizer.encode(r['question'],add_special_tokens=True,padding='longest',max_length=30,truncation=True),dtype=torch.long)
            samples.append({'qa_tokens':qa,'qsn_token':q})
        padded=videoqa_collate_fn(samples)
        for j,r in enumerate(block):
            sid=r['video_id']+'_'+r['qid']
            if sid in tokens:raise RuntimeError('DUPLICATE_VAL_ID')
            tokens[sid]=(padded['qa_tokens'][j:j+1].clone(),padded['qsn_token'][j:j+1].clone())
    if set(tokens)!=set(rows):raise RuntimeError('VAL_TOKEN_ID_MISMATCH')
    signatures=[]
    for sid in rows:
        qa=tokens[sid][0];m=(qa!=tokenizer.pad_token_id).numpy()
        signatures.append((tuple(qa.shape),digest(m.tobytes())))
    return torch,model,features,tokens,tokenizer.pad_token_id,signatures,{'torch':torch.__version__,'transformers':transformers.__version__,'GPU':torch.cuda.get_device_name(0),'FP32':True,'TF32':False,'AMP':False}

def verify_inputs(repo_path,h5_path,checkpoint_path):
    cfg_raw=(HERE/'CONFIG.json').read_bytes();cfg=json.loads(cfg_raw)
    locked=json.loads(read_bound(HERE/'LOCKED.json',cfg['lock_sha256']))
    if (cfg['split'],cfg['checkpoint_sha256'],locked['split'],locked['checkpoint'],locked['commit'],locked['arms']) != ('VAL',EXPECTED_SHA,'VAL',EXPECTED_SHA,cfg['commit'],ARMS):
        raise RuntimeError('LOCK_CONTRACT_MISMATCH')
    repo=Path(repo_path);code=repo/'code/GroundFormer'
    if subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()!=cfg['commit']:
        raise RuntimeError('WRONG_SOURCE_COMMIT')
    status=subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=all','--','code/GroundFormer'],text=True)
    if status.strip():raise RuntimeError('GROUND_FORMER_SOURCE_DIRTY')
    for rel,expected in cfg['source_text_sha256'].items():
        src=(code/rel).read_text(encoding='utf-8-sig')
        if digest(src.replace('\r\n','\n').encode())!=expected:raise RuntimeError(f'REVIEWED_SOURCE_MISMATCH:{rel}')
    # Accept Windows checkout line endings only if the canonical VAL bytes are identical.
    csv_raw=(repo/'datasets/nextgqa/val.csv').read_bytes().replace(b'\r\n',b'\n')
    if digest(csv_raw)!=cfg['val_csv_sha256']:raise RuntimeError('VAL_CSV_HASH_MISMATCH')
    h5_raw=read_bound(h5_path,cfg['val_h5_sha256'])
    checkpoint=read_bound(checkpoint_path,EXPECTED_SHA)
    if len(checkpoint)!=cfg['checkpoint_bytes']:raise RuntimeError('CHECKPOINT_SIZE_MISMATCH')
    reference=read_bound(HERE/'BASELINE_REFERENCE.npz',cfg['baseline_reference_sha256'])
    with np.load(io.BytesIO(reference),allow_pickle=False) as z:ref={k:z[k].copy() for k in z.files}
    rows_raw=read_bound(HERE/'VAL_REFERENCE_ROWS.json',cfg['val_reference_rows_sha256'])
    rows={r['sample_id']:r for r in json.loads(rows_raw)};ids=list(rows)
    if ids!=ref['ids'].tolist():raise RuntimeError('REFERENCE_ORDER_MISMATCH')
    native=list(csv.DictReader(io.StringIO(csv_raw.decode('utf-8-sig'))))
    native_ids=[r['video_id']+'_'+r['qid'] for r in native]
    labels=[[r[f'a{i}'] for i in range(5)].index(r['answer']) for r in native]
    if native_ids!=ids or any(label!=rows[sid]['answer'] for label,sid in zip(labels,ids)):
        raise RuntimeError('VAL_CSV_IDS_OR_LABELS_DIFFER')
    return cfg_raw,cfg,csv_raw,h5_raw,checkpoint,ref,rows

def preflight(repo_path,h5_path,checkpoint_path):
    _,cfg,_,_,_,_,rows=verify_inputs(repo_path,h5_path,checkpoint_path)
    packages={}
    for package in RUNTIME_PACKAGES:
        try:packages[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:packages[package]=None
    return {'status':'BLOCKED_GPU_AUTHORIZATION','input_contract':'PASS','commit':cfg['commit'],
            'checkpoint_sha256':cfg['checkpoint_sha256'],'val_h5_sha256':cfg['val_h5_sha256'],
            'val_csv_canonical_sha256':cfg['val_csv_sha256'],'source_hashes_verified':len(cfg['source_text_sha256']),
            'val_questions':len(rows),'val_videos':len({r['video_id'] for r in rows.values()}),
            'input_paths':{'repo':str(repo_path),'val_h5':str(h5_path),'checkpoint':str(checkpoint_path)},
            'dependencies':packages,'missing_dependencies':[p for p,v in packages.items() if v is None],
            'reference_software':cfg['source_cache_software'],'runtime_import_and_cuda_probe':'NOT_RUN',
            'gpu_minutes':0,'model_forwards':0,'training_steps':0,'test_opens':0}

def run(signal,out_path,repo_path,h5_path,checkpoint_path,max_minutes=90):
    require_authorization(signal)
    from instrumentation import CandidateCapture, LocalIntervention, UpstreamTrace, CaptureDonor, get_target
    from live_analysis import analyze, load_shards
    from pinned_metrics import get_tIoU,find_seg_ada
    clock=time.monotonic()
    if max_minutes<=0 or max_minutes>90:raise ValueError('GPU_BUDGET_MUST_BE_IN_0_90_MINUTES')
    cfg_raw,cfg,csv_raw,h5_raw,checkpoint,ref,rows=verify_inputs(repo_path,h5_path,checkpoint_path)
    ids=list(rows)
    sys.path.insert(0,str(Path(repo_path)/'code/GroundFormer'))
    root=Path(out_path);root.mkdir(parents=True,exist_ok=True)
    fingerprint=digest(cfg_raw)
    lock={'config_sha256':fingerprint,'runner_sha256':{p:digest((HERE/p).read_bytes()) for p in ('core.py','pinned_metrics.py','instrumentation.py','gpu_session.py','live_analysis.py')},
          'split':'VAL','ids_sha256':digest('\n'.join(ids).encode()),
          'checkpoint':EXPECTED_SHA,'arms':ARMS,'training_steps':0,'TEST_label_opens':0,
          'max_minutes':max_minutes,'donor_policy':'lexical first, different video, exact QA mask signature, no GT',
          'primary_contrasts':['SA1 ZERO - CONTROL','CMA ZERO - CONTROL','SA2 BLOCK - RANDOM'],
          'bootstrap':{'unit':'video','replicates':10000,'seed':20261101,'primary_GQA_alpha':.05/3}}
    lock_path=root/'SESSION_LOCK.json'
    if lock_path.exists() and json.loads(lock_path.read_text())!=lock:raise RuntimeError('RESUME_LOCK_DIFFERS')
    if not lock_path.exists():write_json(lock_path,lock)
    fingerprint=digest(json.dumps(lock,sort_keys=True).encode())
    if (root/'COMPLETE.json').exists():raise RuntimeError('SESSION_ALREADY_COMPLETE')
    # Single writer; stale active marker is recoverable only after operator confirms old process ended.
    writer=root/'ACTIVE_WRITER'
    with writer.open('x',encoding='utf-8') as f:f.write(str(os.getpid()))
    stats={'status':'RUNNING','answer_calls_this_session':0,'question_calls_this_session':0,
           'video_calls_this_session':0,'training_steps':0,'TEST_label_opens':0,'run_fingerprint':fingerprint,'checkpoint_before':EXPECTED_SHA}
    validations={'status':'NOT-RUN','tested':[]};gpu_start=None
    def heartbeat():
        if time.monotonic()-clock>max_minutes*60:raise TimeoutError('GPU_SESSION_BUDGET_REACHED')
    try:
        heartbeat();gpu_start=time.monotonic()
        torch,model,features,tokens,pad,signatures,software=model_runtime(cfg,checkpoint,csv_raw,h5_raw,rows)
        del checkpoint,h5_raw;stats['software']=software
        software_path=root/'SOFTWARE_LOCK.json'
        if software_path.exists() and json.loads(software_path.read_text())!=software:raise RuntimeError('RESUME_SOFTWARE_DIFFERS')
        if not software_path.exists():write_json(software_path,software)
        if any(p.requires_grad or p.dtype!=torch.float32 for p in model.parameters()):raise RuntimeError('FROZEN_FP32_REQUIRED')
        donor_map=choose_donors(ids,[rows[s]['video_id'] for s in ids],signatures)
        donors_needed={x for x in donor_map.values() if x is not None};write_json(root/'DONOR_MAP.json',donor_map)
        def forward(sid,arm='CONTROL',donor=None,trace_arm=None,bank=False):
            heartbeat()
            from contextlib import ExitStack
            qa,_=tokens[sid];qa=qa.cuda().repeat(2,1,1)
            f=torch.from_numpy(features[rows[sid]['video_id']]).cuda().unsqueeze(0).repeat(2,1,1)
            zeros=torch.zeros(2,dtype=torch.long,device='cuda')
            observation={};donor_values={};trace=None;intervention=None;residual={}
            with torch.no_grad(),torch.autocast(device_type='cuda',enabled=False):
                stats['video_calls_this_session']+=1
                frame=model.video(f,torch.ones(f.shape[:2],device='cuda'))
                # The pinned cache runner calls answer with None/None directly.
                with ExitStack() as stack:
                    # Trace must register its pre-hook before a possible SA2 mask mutation.
                    if trace_arm:trace=stack.enter_context(UpstreamTrace(get_target(model,trace_arm)))
                    if arm!='CONTROL':intervention=stack.enter_context(LocalIntervention(model,arm,sid,donor))
                    if bank:
                        donors={'SA1':stack.enter_context(CaptureDonor(model,'SA1_')),'CMA':stack.enter_context(CaptureDonor(model,'CMA_'))}
                    # Norms before/after each residual normalization, same scope for all arms.
                    layer=model.qformer.layers[1]
                    for j in (1,2,3):
                        def tap(module,args,out,j=j):
                            residual[f'norm{j}_input_L2']=float(args[0].float().norm())
                            residual[f'norm{j}_output_L2']=float(out.float().norm())
                        h=getattr(layer,f'norm{j}').register_forward_hook(tap);stack.callback(h.remove)
                    capture=stack.enter_context(CandidateCapture(model))
                    stats['answer_calls_this_session']+=1
                    answer=model.answer(qa,None,frame,None,(qa!=pad).float(),zeros)
                    result=capture.finish(answer)
                    if intervention:observation=intervention.observation
                    if bank:donor_values={key:value.value[:5] for key,value in donors.items()}
            observation['residual_norms']=residual
            return result,observation,donor_values,trace.events if trace else []
        def parity(result,i):
            r=rows[ids[i]];c=result['prediction']
            if c!=int(ref['prediction'][i]):raise RuntimeError(f'BASELINE_PREDICTION_PARITY_FAIL:{ids[i]}')
            rawerr=float(np.max(np.abs(result['layers'][c]-ref['layer_curves'][i])))
            finerr=float(np.max(np.abs(result['final'][c]-ref['final_curves'][i])))
            m=curve_metrics(result['final'][c],r['timestamps'],r['GT_intervals'],find_seg_ada,get_tIoU)
            if not np.isfinite(rawerr+finerr) or max(rawerr,finerr)>1e-6 or m['span']!=r['span']:
                raise RuntimeError(f'BASELINE_CURVE_OR_SPAN_PARITY_FAIL:{ids[i]}:{rawerr}:{finerr}')
            if bool(c==r['answer'] and m['IoP']>=.5)!=r['GQA'] or m['peak_correct']!=r['peak_correct']:
                raise RuntimeError(f'BASELINE_METRIC_PARITY_FAIL:{ids[i]}')
            return rawerr,finerr,m['IoU']
        @functools.lru_cache(maxsize=32)
        def donor_at(sid):return open_npz(root/'DONORS'/f'{ids.index(sid):04d}.npz',fingerprint)
        def run_shard(arm,start,stop):
            path=root/arm/f'shard_{start:04d}.npz'
            if path.exists() and path.with_suffix('.json').exists():
                d=open_npz(path,fingerprint)
                if d['ids'].tolist()!=ids[start:stop]:raise RuntimeError('SHARD_MEMBERSHIP_FAIL')
                return d
            records=[];observations=[];parities=[]
            for i in range(start,stop):
                sid=ids[i];donor=None;bank=arm=='CONTROL' and sid in donors_needed
                if 'DONOR' in arm:
                    a=donor_at(donor_map[sid])[arm.split('_')[0]];donor=np.concatenate([a,a])
                result,obs,dv,_=forward(sid,arm,donor,bank=bank)
                if arm=='CONTROL':parities.append(parity(result,i))
                if bank:save_npz(root/'DONORS'/f'{i:04d}.npz',dv,fingerprint)
                records.append(result);obs.update(sample_id=sid,donor_id=donor_map[sid] if 'DONOR' in arm else None);observations.append(obs)
            d={k:np.asarray([r[k] for r in records]) for k in records[0]}
            d['ids']=np.asarray(ids[start:stop]);d['tensor_log_json']=np.asarray([json.dumps(o) for o in observations]);d['parity']=np.asarray(parities)
            save_npz(path,d,fingerprint)
            print(f'{arm}: {stop}/{len(ids)}',flush=True);return d
        # B0 real identity on 16: every candidate curve, raw layer and logits bitwise.
        validation_path=root/'B2_CONTROL_VALIDATION.json'
        if validation_path.exists():
            validations=json.loads(validation_path.read_text())
            if validations.get('run_fingerprint')!=fingerprint:raise RuntimeError('VALIDATION_FINGERPRINT_MISMATCH')
        else:validations={'status':'NOT-RUN','tested':[],'identity':{},'run_fingerprint':fingerprint}
        if set(validations.get('identity',{}))!={'SA1','SA2','CMA'}:
            validations['identity']={}
            for i,sid in enumerate(ids[:16]):
                control,_,_,_=forward(sid);parity(control,i)
                for m in ['SA1','SA2','CMA']:
                    identity,_,_,_=forward(sid,f'{m}_IDENTITY')
                    if not all(np.array_equal(control[k],identity[k]) for k in ('logits','layers','raw','final','prediction')):
                        raise RuntimeError(f'IDENTITY_MODEL_PARITY_FAIL:{m}:{sid}')
            validations['identity']={m:{'status':'PASS','questions':16,'bitwise':True} for m in ['SA1','SA2','CMA']}
            write_json(validation_path,validations)
        smoke=run_shard('CONTROL',0,32)
        write_json(root/'B1_PARITY.json',{'status':'SMOKE_PASS','questions':32,'max_raw':float(smoke['parity'][:,0].max()),'max_final':float(smoke['parity'][:,1].max()),'run_fingerprint':fingerprint})
        schedule=[(0,32)]+[(i,min(i+32,len(ids))) for i in range(32,len(ids),32)]
        for start,stop in schedule[1:]:run_shard('CONTROL',start,stop)
        baseline=load_shards(root/'CONTROL',ids,fingerprint)
        summary={'status':'PASS','questions':len(ids),'max_raw':float(baseline['parity'][:,0].max()),'max_final':float(baseline['parity'][:,1].max()),'mIoU_percent':float(baseline['parity'][:,2].mean()*100),'run_fingerprint':fingerprint}
        summary.update(QA=sum(r['QA'] for r in rows.values()),GQA=sum(r['GQA'] for r in rows.values()),
                       peak_in_GT=sum(r['peak_correct'] for r in rows.values()),
                       native_selected_max_error=float(baseline['selected_error'].max()))
        if abs(summary['mIoU_percent']-16.911405757782664)>1e-6:raise RuntimeError('BASELINE_MIOU_PARITY_FAIL')
        write_json(root/'B1_PARITY.json',summary)
        stats['B1_questions_with_five_candidates']=len(ids)
        # Write B1 even if B2 validity blocks. Live analyses are CPU-only.
        b1_state=analyze(root,rows,include_b2=False,baseline=baseline)
        skipped={};ready=[]
        for arm in ARMS:
            if 'DONOR' in arm and any(v is None for v in donor_map.values()):
                skipped[arm]={'status':'BLOCKED_VALIDITY','missing_donors':sum(v is None for v in donor_map.values())};continue
            try:
                if arm not in validations.get('tested',[]):
                    for i,sid in enumerate(ids[:16]):
                        donor=None
                        if 'DONOR' in arm:
                            a=donor_at(donor_map[sid])[arm.split('_')[0]];donor=np.concatenate([a,a])
                        _,_,_,tr0=forward(sid,trace_arm=arm)
                        y,obs,_,tr1=forward(sid,arm,donor,trace_arm=arm)
                        if not tr0 or tr0!=tr1:raise RuntimeError(f'UPSTREAM_BITWISE_FAIL:{arm}:{sid}')
                        if i==0:
                            repeated,repeat_obs,_,_=forward(sid,arm,donor)
                            if not all(np.array_equal(y[k],repeated[k]) for k in ('logits','layers','raw','final','prediction')) or obs!=repeat_obs:
                                raise RuntimeError(f'INTERVENTION_REPEAT_FAIL:{arm}:{sid}')
                        if arm.startswith('CMA') and not np.array_equal(y['layers'][:,1],baseline['layers'][i,:,1]):
                            raise RuntimeError(f'CMA_SAME_LAYER_DET_CHANGED:{sid}')
                        if arm.startswith('SA2'):
                            if obs['new_masked_per_video_row']!=[31]*32 or not obs['token_rows_unchanged']:raise RuntimeError('SA2_MASK_CONTRACT_FAIL')
                        elif not obs.get('residual_consumption_exact'):raise RuntimeError('REPLACEMENT_CONTRACT_FAIL')
                    run_shard(arm,0,32)
                    validations.setdefault('tested',[]).append(arm);write_json(validation_path,validations)
                ready.append(arm)
            except (RuntimeError,ValueError) as e:
                # Do not convert a failed contract into a valid arm, or run its full cohort.
                skipped[arm]={'status':'BLOCKED_VALIDITY','error':str(e)}
        for arm in ready:
            for start,stop in schedule[1:]:run_shard(arm,start,stop)
        validations.update(status='PASS_FOR_EXECUTED_ARMS' if ready else 'BLOCKED_VALIDITY',skipped=skipped,
                           boundary_fingerprint_questions_per_arm=16,repeat_questions_per_arm=1)
        write_json(validation_path,validations)
        stats['B2_questions_per_completed_arm']={arm:len(ids) for arm in ready}
        torch.cuda.synchronize();stats['GPU_session_minutes']=(time.monotonic()-gpu_start)/60
        b1,b2=analyze(root,rows,include_b2=True,baseline=baseline,b1_state=b1_state,arms=ready,fingerprint=fingerprint)
        stats['status']='COMPLETE' if not skipped else 'COMPLETE_WITH_BLOCKED_VALIDITY_ARMS'
        stats['skipped']=skipped
    except BaseException as e:
        stats['status']='BLOCKED_VALIDITY' if isinstance(e,RuntimeError) else 'INTERRUPTED'
        stats['error']=f'{type(e).__name__}: {e}'
        raise
    finally:
        if gpu_start is not None:stats.setdefault('GPU_session_minutes',(time.monotonic()-gpu_start)/60)
        with open(checkpoint_path,'rb') as f:stats['checkpoint_after']=hashlib.file_digest(f,'sha256').hexdigest()
        stats['checkpoint_unchanged']=stats['checkpoint_after']==EXPECTED_SHA
        write_json(root/'EXECUTION_STATUS.json',stats);writer.unlink(missing_ok=True)
    if not stats['checkpoint_unchanged']:raise RuntimeError('CHECKPOINT_CHANGED_DURING_SESSION')
    write_json(root/'COMPLETE.json',stats)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--authorization');p.add_argument('--preflight',action='store_true')
    p.add_argument('--out');p.add_argument('--repo',required=True)
    p.add_argument('--h5',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--max-minutes',type=float,default=90)
    a=p.parse_args()
    if a.preflight:
        report=preflight(a.repo,a.h5,a.checkpoint)
        if a.out:write_json(Path(a.out),report)
        print(json.dumps(report,indent=2))
        return
    if not a.out:p.error('--out is required for model execution')
    run(a.authorization,a.out,a.repo,a.h5,a.checkpoint,a.max_minutes)

if __name__=='__main__':main()
