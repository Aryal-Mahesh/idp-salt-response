"""
Merge subsets 1-4 into a single master library CSV.
Compute missing polymer-physics features for synthetic sequences via localCIDER.
"""
import numpy as np
import pandas as pd
from localcider.sequenceParameters import SequenceParameters

# Canonical feature columns we want in the master library
FEATURES = [
    'seq_id', 'subset', 'sequence', 'length',
    'fcr', 'ncpr', 'kappa', 'scd', 'shd',
    'mean_hydropathy', 'f_aro', 'f_pos', 'f_neg',
    'source_name',           # protein name or synthetic ID
    # CALVADOS @ 150 mM (only IDRome entries have these)
    'Rg_150', 'Ree_150', 'nu_150', 'Delta_150', 'Rh_150',
]

# Kyte-Doolittle hydropathy (for synthetic sequences; IDRome uses CALVADOS lambda)
KD_HYDROPATHY = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'E': -3.5, 'Q': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
}
AROMATIC = set('FWY')
POSITIVE = set('KR')
NEGATIVE = set('DE')

def compute_synthetic_features(seq):
    """Compute features for synthetic sequences using localCIDER + manual descriptors."""
    sp = SequenceParameters(seq)
    
    fcr = sp.get_FCR()
    ncpr = sp.get_NCPR()
    
    # kappa is undefined if only one charge type present
    try:
        kappa = sp.get_kappa()
        if not np.isfinite(kappa):
            kappa = np.nan
    except Exception:
        kappa = np.nan
    
    # SCD (Sawle-Ghosh): localCIDER has it
    try:
        scd = sp.get_SCD()
    except AttributeError:
        # Older versions of localCIDER lack get_SCD; compute manually
        scd = compute_scd_manual(seq)
    
    n = len(seq)
    f_aro = sum(1 for a in seq if a in AROMATIC) / n
    f_pos = sum(1 for a in seq if a in POSITIVE) / n
    f_neg = sum(1 for a in seq if a in NEGATIVE) / n
    mean_hydro = np.mean([KD_HYDROPATHY.get(a, 0.0) for a in seq])
    
    # SHD (sequence hydropathy decoration) - Zheng et al.
    shd = compute_shd(seq, KD_HYDROPATHY)
    
    return {
        'fcr': fcr, 'ncpr': ncpr, 'kappa': kappa, 'scd': scd, 'shd': shd,
        'mean_hydropathy': mean_hydro,
        'f_aro': f_aro, 'f_pos': f_pos, 'f_neg': f_neg,
    }

def compute_scd_manual(seq):
    """Sawle-Ghosh SCD: sum over all i<j of q_i q_j sqrt(|i-j|) / N."""
    charges = np.array([1 if a in POSITIVE else (-1 if a in NEGATIVE else 0) for a in seq])
    n = len(seq)
    s = 0.0
    for i in range(n):
        if charges[i] == 0:
            continue
        for j in range(i + 1, n):
            if charges[j] == 0:
                continue
            s += charges[i] * charges[j] * np.sqrt(j - i)
    return s / n

def compute_shd(seq, hydropathy_scale):
    """SHD (sequence hydropathy decoration): Zheng et al. 2020 definition."""
    h = np.array([hydropathy_scale.get(a, 0.0) for a in seq])
    n = len(seq)
    s = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            s += (h[i] + h[j]) / (j - i)
    return s / n

# ---------- Loaders ----------

def load_subset1():
    df = pd.read_csv('subset1_kappa_variants.csv')
    print(f"Subset 1 (kappa variants): {len(df)}")
    return df, 'synthetic'

def load_subset2():
    df = pd.read_csv('subset2_ncpr_series.csv')
    print(f"Subset 2 (NCPR series): {len(df)}")
    return df, 'synthetic'

def load_subset3():
    df = pd.read_csv('subset3_idrome_stratified.csv')
    print(f"Subset 3 (IDRome stratified): {len(df)}")
    return df, 'natural'

def load_subset4():
    df = pd.read_csv('subset4_idrome_lowfcr.csv')
    print(f"Subset 4 (IDRome low-FCR): {len(df)}")
    return df, 'natural'

def to_master_row_synthetic(row):
    """Convert a synthetic-subset row into the master schema."""
    seq = row['sequence']
    feats = compute_synthetic_features(seq)
    return {
        'seq_id': row['seq_id'],
        'subset': row['subset'],
        'sequence': seq,
        'length': len(seq),
        **feats,
        'source_name': row['seq_id'],
        'Rg_150': np.nan, 'Ree_150': np.nan, 'nu_150': np.nan,
        'Delta_150': np.nan, 'Rh_150': np.nan,
    }

def to_master_row_natural(row):
    """Convert an IDRome row into the master schema (mostly column renaming)."""
    seq = row['fasta']
    return {
        'seq_id': row['seq_id'],
        'subset': row['subset'],
        'sequence': seq,
        'length': int(row['N']),
        'fcr': row['fcr'],
        'ncpr': row['ncpr'],
        'kappa': row['kappa'] if pd.notna(row['kappa']) else np.nan,
        'scd': row['scd'],
        'shd': row['shd'],
        'mean_hydropathy': row['mean_lambda'],   # CALVADOS lambda scale, not KD
        'f_aro': row['faro'],
        'f_pos': row['fK'] + row['fR'],
        'f_neg': row['fE'] + row['fD'],
        'source_name': row.get('protein_name', row['seq_name']),
        'Rg_150': row['Rg/nm'],
        'Ree_150': row['Ree/nm'],
        'nu_150': row['nu'],
        'Delta_150': row['Delta'],
        'Rh_150': row['Rh/nm'],
    }

def main():
    print("=== Loading subsets ===\n")
    
    s1, _ = load_subset1()
    s2, _ = load_subset2()
    s3, _ = load_subset3()
    s4, _ = load_subset4()
    
    print("\n=== Converting to master schema ===\n")
    
    rows = []
    print("Subset 1 (computing features)...")
    for _, row in s1.iterrows():
        rows.append(to_master_row_synthetic(row))
    
    print("Subset 2 (computing features)...")
    for _, row in s2.iterrows():
        rows.append(to_master_row_synthetic(row))
    
    print("Subset 3 (using IDRome precomputed)...")
    for _, row in s3.iterrows():
        rows.append(to_master_row_natural(row))
    
    print("Subset 4 (using IDRome precomputed)...")
    for _, row in s4.iterrows():
        rows.append(to_master_row_natural(row))
    
    df = pd.DataFrame(rows)[FEATURES]
    
    # Final sanity: drop any duplicates by sequence (shouldn't happen, but check)
    n_before = len(df)
    df = df.drop_duplicates(subset='sequence').reset_index(drop=True)
    n_after = len(df)
    if n_before != n_after:
        print(f"  Removed {n_before - n_after} duplicate sequences")
    
    out_path = 'library.csv'
    df.to_csv(out_path, index=False)
    
    print(f"\n=== Wrote {len(df)} sequences to {out_path} ===\n")
    print("Counts by subset:")
    print(df['subset'].value_counts())
    print("\nFeature ranges:")
    print(df[['length', 'fcr', 'ncpr', 'kappa', 'scd', 'shd', 'mean_hydropathy']].describe())

if __name__ == '__main__':
    main()
