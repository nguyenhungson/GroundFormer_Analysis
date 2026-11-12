"""Non-model utilities. No torch import or forward is allowed in this module."""
import hashlib
import json
import numpy as np

SEED = 20261101
EXPECTED_SHA = '5a2f49eb92b2d9631d9787da34589a6c6a1c9ffc814d727c74cb7ad61ef5e776'
ARMS = [
    'SA1_L2_ATTN_ZERO', 'SA1_L2_ATTN_DONOR_MATCHED_TEXT_LENGTH',
    'SA1_L2_ATTN_NOISE_MATCHED_L2_NORM', 'CMA_L2_AFTER_WO_ZERO',
    'CMA_L2_AFTER_WO_DONOR_MATCHED', 'CMA_L2_AFTER_WO_NOISE_MATCHED_L2_NORM',
    'SA2_L2_BLOCK_CROSSFRAME', 'SA2_L2_RANDOM_MATCHED_LINK_MASK',
]

def digest(data):
    return hashlib.sha256(data).hexdigest()

def require_authorization(signal):
    # This gate is called before importing torch, constructing or forwarding a model.
    if signal != 'GPU-EVAL-OK':
        raise PermissionError('BLOCKED_GPU_AUTHORIZATION')

def stable_seed(sid, arm):
    return int.from_bytes(hashlib.sha256(f'{SEED}|{sid}|{arm}'.encode()).digest()[:8], 'little')

def tensor_stats(original, replacement):
    a, b = np.asarray(original, np.float64), np.asarray(replacement, np.float64)
    if a.shape != b.shape:
        raise ValueError('REPLACEMENT_SHAPE_MISMATCH')
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return {'original_norm': float(na), 'replacement_norm': float(nb),
            'delta_norm': float(np.linalg.norm(b-a)),
            'relative_delta': float(np.linalg.norm(b-a) / max(na, 1e-30)),
            'original_mean': float(a.mean()), 'replacement_mean': float(b.mean()),
            'original_std': float(a.std()), 'replacement_std': float(b.std())}

def match_norm(reference, candidate):
    a, b = np.asarray(reference), np.asarray(candidate)
    if a.shape != b.shape:
        raise ValueError('DONOR_SHAPE_MISMATCH')
    # Per B*candidate row, not one global norm across five answers.
    axes = tuple(range(1, a.ndim))
    na = np.linalg.norm(a.astype(np.float64), axis=axes, keepdims=True)
    nb = np.linalg.norm(b.astype(np.float64), axis=axes, keepdims=True)
    if np.any((nb == 0) & (na != 0)):
        raise ValueError('ZERO_DONOR_DIRECTION')
    return (b * np.divide(na, nb, out=np.zeros_like(na), where=nb != 0)).astype(a.dtype)

def replacement(a, kind, seed, donor=None):
    a = np.asarray(a)
    if kind == 'IDENTITY':
        return a.copy()
    if kind == 'ZERO':
        return np.zeros_like(a)
    if kind == 'NOISE':
        # Duplicated B=2 is deliberate: five candidates repeat identically.
        if a.shape[0] != 10 or not np.array_equal(a[:5], a[5:]):
            raise ValueError('EXPECTED_DUPLICATED_B2_CONTRACT')
        n = np.random.Generator(np.random.PCG64(seed)).normal(size=a[:5].shape).astype(a.dtype)
        n = np.concatenate([n, n], axis=0)
        return match_norm(a, n)
    if kind == 'DONOR':
        if donor is None:
            raise ValueError('MISSING_DONOR')
        return match_norm(a, donor)
    raise ValueError(kind)

def sa2_mask(original, num_tokens, kind, seed):
    m = np.asarray(original)
    if m.dtype != bool or m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError('EXPECTED_BOOL_SQUARE_MASK')
    nt, size = int(num_tokens), len(m)
    if not (0 < nt < size):
        raise ValueError('INVALID_TOKEN_BOUNDARY')
    if np.any(m[nt:]) or not np.all(m[:nt, nt:]):
        raise ValueError('SOURCE_SA2_MASK_DIFFERS')
    out = m.copy()
    rng = np.random.Generator(np.random.PCG64(seed))
    for row in range(nt, size):
        blocked = [col for col in range(nt, size) if col != row]
        if kind == 'RANDOM':
            allowed = [col for col in range(size) if col != row and not m[row,col]]
            blocked = rng.choice(allowed, size=len(blocked), replace=False)
        elif kind != 'BLOCK':
            raise ValueError(kind)
        out[row, blocked] = True
    return out

def choose_donors(ids, videos, signatures):
    # No labels, QA outcomes or temporal GT used. Exact mask/length signature.
    buckets = {}
    for sid, signature in zip(ids, signatures):
        buckets.setdefault(signature, []).append(sid)
    vids = dict(zip(ids, videos))
    result = {}
    for sid, signature in zip(ids, signatures):
        options = sorted(x for x in buckets[signature] if x != sid and vids[x] != vids[sid])
        if not options:
            result[sid] = None  # no silent shape coercion or dropped row
        else:
            # First compatible ID in a locked lexical order; a small donor bank
            # can serve a whole signature bucket, without extra donor forwards.
            result[sid] = options[0]
    return result

def curve_metrics(curve, timestamps, intervals, decode, overlap):
    a, t = np.asarray(curve), np.asarray(timestamps)
    if a.shape != t.shape or a.ndim != 1 or not np.all(np.isfinite(a)):
        raise ValueError('INVALID_CURVE')
    peak = int(np.argmax(a))
    # Match JSON-loaded native decoder input; get_tIoU(loc=GT, span=pred).
    span = [float(x) for x in decode(a.tolist(), t)]
    values = [overlap(g, span) for g in intervals]
    # Evaluator initializes maxima at zero; preserve point interval semantics.
    iou = max([0.] + [float(x[0]) for x in values])
    iop = max([0.] + [float(x[1]) for x in values])
    inside = np.array([any(s <= ts <= e for s,e in intervals) for ts in t])
    p = a.astype(np.float64); p = p / p.sum() if p.sum() > 0 else np.zeros_like(p)
    nz = p > 0
    return {'IoU':iou, 'IoP':iop, 'peak_correct':bool(inside[peak]),
            'peak_index':peak, 'peak_timestamp':float(t[peak]), 'span':span,
            'span_length':span[1]-span[0], 'point':span[1]==span[0],
            'GTmass':float(p[inside].sum()),
            'entropy':float(-(p[nz]*np.log(p[nz])).sum()/np.log(len(p))),
            'peak_to_GT':float(min(0 if s<=t[peak]<=e else min(abs(t[peak]-s),abs(t[peak]-e)) for s,e in intervals)),
            'per_GT_IoU_IoP':[[float(x[0]),float(x[1])] for x in values]}

def video_bootstrap(delta, videos, replicates=10000, seed=SEED, alpha=.05):
    x = np.asarray(delta, np.float64)
    if x.ndim == 1:
        x = x[:,None]
    unique, inv = np.unique(np.asarray(videos), return_inverse=True)
    sums = np.zeros((len(unique),x.shape[1])); np.add.at(sums,inv,x)
    counts = np.bincount(inv)
    rng = np.random.Generator(np.random.PCG64(seed)); boots=[]
    for start in range(0,replicates,100):
        ix = rng.integers(0,len(unique),size=(min(100,replicates-start),len(unique)))
        boots.append(sums[ix].sum(1)/counts[ix].sum(1)[:,None])
    b = np.concatenate(boots)
    return {'mean':x.mean(0).tolist(), 'lower':np.quantile(b,alpha/2,axis=0).tolist(),
            'upper':np.quantile(b,1-alpha/2,axis=0).tolist(), 'replicates':replicates,
            'unit':'video', 'videos':len(unique), 'seed':seed, 'alpha':alpha}
