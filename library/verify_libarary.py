"""verify_library.py — sanity-check the sequence library CSVs.
Run from the library/ directory: python verify_library.py
"""
import pandas as pd
import numpy as np

STD_AA = set("ACDEFGHIKLMNPQRSTVWY")

subsets = {
    'subset1_kappa_variants.csv':   {'n': 195, 'name': 'kappa_variant'},
    'subset2_ncpr_series.csv':      {'n': 108, 'name': 'ncpr_series'},
    'subset3_idrome_stratified.csv':{'n': 150, 'name': 'idrome_stratified'},
    'subset4_idrome_lowfcr.csv':    {'n': 58,  'name': 'idrome_lowfcr'},
}

print("="*60)
print("1. COUNTS")
total = 0
frames = {}
for f, meta in subsets.items():
    df = pd.read_csv(f)
    frames[f] = df
    ok = "OK" if len(df) == meta['n'] else f"MISMATCH (expected {meta['n']})"
    print(f"  {f:35s} {len(df):4d}  {ok}")
    total += len(df)
print(f"  {'TOTAL':35s} {total:4d}  {'OK' if total==511 else 'MISMATCH (expected 511)'}")

lib = pd.read_csv('library.csv')
print(f"  {'library.csv':35s} {len(lib):4d}  {'OK' if len(lib)==511 else 'MISMATCH'}")

# pick the sequence column name (adjust if yours differs)
SEQ = 'sequence' if 'sequence' in lib.columns else [c for c in lib.columns if 'seq' in c.lower()][0]
print(f"\n  (using sequence column: '{SEQ}')")

print("="*60)
print("2. DUPLICATES")
dups = lib[lib.duplicated(subset=[SEQ], keep=False)]
print(f"  duplicate sequences in library.csv: {len(dups)}  {'OK' if len(dups)==0 else 'PROBLEM'}")
# also check seq_id uniqueness if present
if 'seq_id' in lib.columns:
    idd = lib['seq_id'].duplicated().sum()
    print(f"  duplicate seq_id: {idd}  {'OK' if idd==0 else 'PROBLEM'}")

print("="*60)
print("3. SEQUENCE VALIDITY")
bad_aa = lib[~lib[SEQ].apply(lambda s: set(str(s)).issubset(STD_AA))]
print(f"  rows with non-standard residues: {len(bad_aa)}  {'OK' if len(bad_aa)==0 else 'PROBLEM'}")
lib['_len_check'] = lib[SEQ].str.len()
if 'length' in lib.columns:
    mismatch = (lib['_len_check'] != lib['length']).sum()
    print(f"  rows where len(seq) != length column: {mismatch}  {'OK' if mismatch==0 else 'PROBLEM'}")
print(f"  length range: {lib['_len_check'].min()}–{lib['_len_check'].max()}")

print("="*60)
print("4. DESCRIPTOR RECOMPUTE (spot check on 5 random rows)")

def fcr_ncpr(seq):
    pos = sum(c in 'KR' for c in seq)
    neg = sum(c in 'DE' for c in seq)
    n = len(seq)
    return (pos+neg)/n, (pos-neg)/n

def scd(seq):
    q = np.array([1 if c in 'KR' else -1 if c in 'DE' else 0 for c in seq])
    n = len(seq); s = 0.0
    for i in range(1, n):
        for j in range(i):
            if q[i] and q[j]:
                s += q[i]*q[j]*np.sqrt(i-j)
    return s/n

sample = lib.sample(min(5, len(lib)), random_state=0)
for _, row in sample.iterrows():
    seq = str(row[SEQ])
    f_calc, n_calc = fcr_ncpr(seq)
    s_calc = scd(seq)
    print(f"  {row.get('seq_id','?')}: "
          f"FCR {f_calc:.3f} vs {row.get('fcr', np.nan):.3f} | "
          f"NCPR {n_calc:+.3f} vs {row.get('ncpr', np.nan):+.3f} | "
          f"SCD {s_calc:+.2f} vs {row.get('scd', np.nan):+.2f}")
print("\n  (FCR/NCPR/SCD recomputed from sequence should match the CSV columns)")
print("="*60)
