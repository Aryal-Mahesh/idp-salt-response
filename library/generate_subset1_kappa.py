"""
Subset 1: Das-Pappu kappa-variants.
Fixed length and FCR, varied charge patterning (kappa).
Pappu G/S/A/P background. Saves incrementally per cell.
"""
import numpy as np
import pandas as pd
from localcider.sequenceParameters import SequenceParameters
import os
import time

LENGTHS = [50, 100, 200]
FCRS = [0.30, 0.50]
KAPPA_TARGETS = np.round(np.arange(0.05, 0.56, 0.05), 2)  # 0.05, 0.10, ..., 0.55
N_PER_BIN = 3
KAPPA_TOLERANCE = 0.04   # slightly looser than before
MAX_ATTEMPTS_PER_CELL = 4000

FILLERS = list('GSAP')
POS_RES = 'K'
NEG_RES = 'E'

OUT_PARTIAL_DIR = 'subset1_parts'
OUT_FINAL = 'subset1_kappa_variants.csv'

os.makedirs(OUT_PARTIAL_DIR, exist_ok=True)

def get_kappa(seq):
    try:
        return SequenceParameters(seq).get_kappa()
    except Exception:
        return None

def make_random_sequence(length, n_pos, n_neg, rng):
    """Random placement of charges on G/S/A/P background."""
    seq = list(rng.choice(FILLERS, size=length))
    positions = rng.choice(length, size=n_pos + n_neg, replace=False)
    for i in positions[:n_pos]:
        seq[i] = POS_RES
    for i in positions[n_pos:]:
        seq[i] = NEG_RES
    return ''.join(seq)

def make_clustered_sequence(length, n_pos, n_neg, cluster_strength, rng):
    """
    Generate sequences with clustered charges (high kappa).
    cluster_strength in [0, 1]: 0 = random, 1 = all + on one side, all - on other.
    """
    seq = list(rng.choice(FILLERS, size=length))
    
    # Place + charges on one side, - on the other, with cluster_strength controlling spread
    if rng.random() < 0.5:
        pos_center, neg_center = length * 0.25, length * 0.75
    else:
        pos_center, neg_center = length * 0.75, length * 0.25
    
    # Width of cluster: cluster_strength=1 -> width=length*0.1, =0 -> width=length
    width = length * (1 - cluster_strength * 0.85)
    
    def sample_positions(center, n, used):
        positions = []
        attempts = 0
        while len(positions) < n and attempts < 200:
            p = int(np.clip(rng.normal(center, width / 2), 0, length - 1))
            if p not in used and p not in positions:
                positions.append(p)
            attempts += 1
        # fill remaining with any unused position
        if len(positions) < n:
            remaining = [i for i in range(length) if i not in used and i not in positions]
            rng.shuffle(remaining)
            positions.extend(remaining[:n - len(positions)])
        return positions
    
    used = set()
    pos_positions = sample_positions(pos_center, n_pos, used)
    used.update(pos_positions)
    neg_positions = sample_positions(neg_center, n_neg, used)
    
    for i in pos_positions:
        seq[i] = POS_RES
    for i in neg_positions:
        seq[i] = NEG_RES
    return ''.join(seq)

def generate_cell(length, fcr, n_per_bin, rng):
    """Generate kappa-binned sequences for one (length, FCR) cell."""
    n_charged = int(round(length * fcr))
    n_pos = n_charged // 2
    n_neg = n_charged - n_pos
    if n_charged % 2 == 0:
        n_neg = n_pos
    
    print(f"  length={length}, FCR={fcr}, charges +{n_pos}/-{n_neg}", flush=True)
    
    binned = {k: [] for k in KAPPA_TARGETS}
    attempts = 0
    t0 = time.time()
    
    while not all(len(v) >= n_per_bin for v in binned.values()):
        if attempts >= MAX_ATTEMPTS_PER_CELL:
            break
        attempts += 1
        
        # Identify which bins still need sequences
        unfilled = [k for k, v in binned.items() if len(v) < n_per_bin]
        if not unfilled:
            break
        max_unfilled_kappa = max(unfilled)
        
        # If high-kappa bins remain, generate clustered; otherwise random
        if max_unfilled_kappa > 0.30 and rng.random() < 0.7:
            cs = rng.uniform(0.3, 1.0)
            seq = make_clustered_sequence(length, n_pos, n_neg, cs, rng)
        else:
            seq = make_random_sequence(length, n_pos, n_neg, rng)
        
        kappa = get_kappa(seq)
        if kappa is None:
            continue
        
        nearest = KAPPA_TARGETS[np.argmin(np.abs(KAPPA_TARGETS - kappa))]
        if abs(nearest - kappa) <= KAPPA_TOLERANCE and len(binned[nearest]) < n_per_bin:
            binned[nearest].append((seq, kappa))
    
    elapsed = time.time() - t0
    filled = sum(1 for v in binned.values() if len(v) >= n_per_bin)
    print(f"    {attempts} attempts, {elapsed:.1f}s, {filled}/{len(KAPPA_TARGETS)} bins filled", flush=True)
    
    records = []
    for target_k, items in binned.items():
        for s, actual_k in items:
            records.append({
                'sequence': s, 'length': length, 'fcr_target': fcr,
                'kappa_target': target_k, 'kappa_actual': actual_k,
                'subset': 'kappa_variant',
            })
    return records

def main():
    rng = np.random.default_rng(42)
    all_records = []
    
    for length in LENGTHS:
        for fcr in FCRS:
            cell_label = f'L{length}_FCR{int(fcr*100)}'
            part_path = os.path.join(OUT_PARTIAL_DIR, f'{cell_label}.csv')
            
            if os.path.exists(part_path):
                print(f"Skipping (already done): {cell_label}")
                df_part = pd.read_csv(part_path)
                all_records.extend(df_part.to_dict('records'))
                continue
            
            print(f"\nGenerating cell: {cell_label}")
            records = generate_cell(length, fcr, N_PER_BIN, rng)
            
            # Save per-cell immediately
            pd.DataFrame(records).to_csv(part_path, index=False)
            all_records.extend(records)
    
    df = pd.DataFrame(all_records)
    df.insert(0, 'seq_id', [f'kvar_{i:04d}' for i in range(len(df))])
    df.to_csv(OUT_FINAL, index=False)
    
    print(f"\n=== Wrote {len(df)} sequences to {OUT_FINAL} ===")
    print(df.groupby(['length', 'fcr_target', 'kappa_target']).size().unstack(fill_value=0))

if __name__ == '__main__':
    main()
