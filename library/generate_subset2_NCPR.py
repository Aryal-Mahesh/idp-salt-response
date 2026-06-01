"""
Subset 2: Polyelectrolyte NCPR series.
Vary net charge per residue at fixed length and FCR.
Pappu G/S/A/P background.
"""
import numpy as np
import pandas as pd
from localcider.sequenceParameters import SequenceParameters

np.random.seed(42)

LENGTHS = [50, 100, 200]
NCPR_TARGETS = np.round(np.arange(-0.4, 0.41, 0.1), 1)   # -0.4, -0.3, ..., 0.4
FCR_TARGET = 0.40   # fixed; allows wide NCPR range
N_PER_BIN = 4

FILLERS = list('GSAP')
POS_RES = 'K'
NEG_RES = 'E'

def make_NCPR_sequence(length, n_pos, n_neg, rng):
    """Build sequence with given charge composition; randomize positions."""
    seq = list(rng.choice(FILLERS, size=length))
    positions = rng.choice(length, size=n_pos + n_neg, replace=False)
    for i in positions[:n_pos]:
        seq[i] = POS_RES
    for i in positions[n_pos:]:
        seq[i] = NEG_RES
    return ''.join(seq)

def compute_n_charges(length, ncpr_target, fcr_target):
    """
    Given target NCPR and FCR, compute n_pos and n_neg.
    NCPR = (n_pos - n_neg) / length
    FCR  = (n_pos + n_neg) / length
    """
    n_charged = int(round(length * fcr_target))
    n_net = int(round(length * ncpr_target))
    # Solve: n_pos - n_neg = n_net, n_pos + n_neg = n_charged
    n_pos = (n_charged + n_net) // 2
    n_neg = n_charged - n_pos
    return n_pos, n_neg, n_charged

def main():
    rng = np.random.default_rng(42)
    records = []
    
    for length in LENGTHS:
        for ncpr in NCPR_TARGETS:
            n_pos, n_neg, n_charged = compute_n_charges(length, ncpr, FCR_TARGET)
            
            # Skip if composition is impossible (negative count)
            if n_pos < 0 or n_neg < 0:
                print(f"  Skip: length={length}, NCPR={ncpr}, would need {n_pos}/{n_neg}")
                continue
            
            actual_ncpr = (n_pos - n_neg) / length
            actual_fcr = (n_pos + n_neg) / length
            
            for replicate in range(N_PER_BIN):
                seq = make_NCPR_sequence(length, n_pos, n_neg, rng)
                sp = SequenceParameters(seq)
                
                records.append({
                    'sequence': seq,
                    'length': length,
                    'ncpr_target': ncpr,
                    'ncpr_actual': sp.get_NCPR(),
                    'fcr_actual': sp.get_FCR(),
                    'kappa_actual': sp.get_kappa() if (n_pos > 0 and n_neg > 0) else None,
                    'n_pos': n_pos,
                    'n_neg': n_neg,
                    'subset': 'ncpr_series',
                })
        print(f"  Done length={length}")
    
    df = pd.DataFrame(records)
    df.insert(0, 'seq_id', [f'ncpr_{i:04d}' for i in range(len(df))])
    out_path = 'subset2_ncpr_series.csv'
    df.to_csv(out_path, index=False)
    
    print(f"\n=== Wrote {len(df)} sequences to {out_path} ===")
    print(df.groupby(['length', 'ncpr_target']).size().unstack(fill_value=0))
    print(f"\nNCPR actual range: {df['ncpr_actual'].min():.3f} to {df['ncpr_actual'].max():.3f}")

if __name__ == '__main__':
    main()
