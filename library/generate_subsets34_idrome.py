"""
Subsets 3 & 4: IDRome-derived natural sequences.

Subset 3: stratified by (SCD, length) for broad natural coverage.
Subset 4: low-FCR sequences (FCR < 0.10) for salting-out regime.
"""
import numpy as np
import pandas as pd

# Path to the IDRome CSV
IDROME_PATH = '_2023_Tesei_IDRome-5/IDRome_DB.csv'

# Subset 3 design
N_SCD_BINS = 5
N_LEN_BINS = 5
N_PER_CELL_SUBSET3 = 6
LEN_MIN, LEN_MAX = 30, 300   # cap length range for tractable simulation

# Subset 4 design
FCR_THRESHOLD = 0.10
N_FARO_BINS = 5
N_LEN_BINS_S4 = 4
N_PER_CELL_SUBSET4 = 5

# Reproducibility
RNG_SEED = 42

def load_idrome():
    df = pd.read_csv(IDROME_PATH)
    print(f"Loaded {len(df)} entries from IDRome")
    # Keep only entries flagged as IDPs by their criterion
    df = df[df['is_idp'] == True].copy()
    print(f"After is_idp filter: {len(df)}")
    # Length filter for simulation tractability
    df = df[(df['N'] >= LEN_MIN) & (df['N'] <= LEN_MAX)].copy()
    print(f"After length filter [{LEN_MIN}, {LEN_MAX}]: {len(df)}")
    return df

def stratified_sample(df, x_col, y_col, n_x_bins, n_y_bins, n_per_cell, rng):
    """Sample n_per_cell entries from each (x_col, y_col) bin using quantile binning."""
    # Use quantile-based bin edges so each bin has comparable population
    x_edges = np.quantile(df[x_col], np.linspace(0, 1, n_x_bins + 1))
    y_edges = np.quantile(df[y_col], np.linspace(0, 1, n_y_bins + 1))
    
    # Ensure unique edges (handle ties at boundaries)
    x_edges = np.unique(x_edges)
    y_edges = np.unique(y_edges)
    
    df = df.copy()
    df['_x_bin'] = pd.cut(df[x_col], x_edges, include_lowest=True, labels=False)
    df['_y_bin'] = pd.cut(df[y_col], y_edges, include_lowest=True, labels=False)
    
    sampled = []
    for (xb, yb), cell in df.groupby(['_x_bin', '_y_bin']):
        if len(cell) == 0:
            continue
        n_take = min(n_per_cell, len(cell))
        picks = cell.sample(n=n_take, random_state=rng.integers(0, 2**31))
        sampled.append(picks)
    
    out = pd.concat(sampled).reset_index(drop=True)
    out = out.drop(columns=['_x_bin', '_y_bin'])
    return out

def main():
    rng = np.random.default_rng(RNG_SEED)
    df = load_idrome()
    
    # ----- Subset 3: stratified by (SCD, length) -----
    print("\n--- Subset 3: stratified by (SCD, length) ---")
    s3 = stratified_sample(df, 'scd', 'N', N_SCD_BINS, N_LEN_BINS,
                            N_PER_CELL_SUBSET3, rng)
    print(f"Sampled {len(s3)} sequences")
    s3 = s3.assign(subset='idrome_stratified')
    s3.insert(0, 'seq_id', [f'idr_{i:04d}' for i in range(len(s3))])
    s3.to_csv('subset3_idrome_stratified.csv', index=False)
    print(f"Wrote subset3_idrome_stratified.csv")
    
    # ----- Subset 4: low-FCR, stratified by aromatic fraction & length -----
    print("\n--- Subset 4: low-FCR (FCR < {}), stratified by (faro, length) ---".format(FCR_THRESHOLD))
    df_low = df[df['fcr'] < FCR_THRESHOLD].copy()
    print(f"After fcr<{FCR_THRESHOLD} filter: {len(df_low)}")
    
    # Avoid overlap with Subset 3 sequences
    overlap = df_low['seq_name'].isin(s3['seq_name'])
    df_low = df_low[~overlap].copy()
    print(f"After de-duplication vs Subset 3: {len(df_low)}")
    
    s4 = stratified_sample(df_low, 'faro', 'N', N_FARO_BINS, N_LEN_BINS_S4,
                            N_PER_CELL_SUBSET4, rng)
    print(f"Sampled {len(s4)} sequences")
    s4 = s4.assign(subset='idrome_lowfcr')
    s4.insert(0, 'seq_id', [f'lowfcr_{i:04d}' for i in range(len(s4))])
    s4.to_csv('subset4_idrome_lowfcr.csv', index=False)
    print(f"Wrote subset4_idrome_lowfcr.csv")
    
    # ----- Quick summary -----
    print("\n--- Summary ---")
    print(f"Subset 3: {len(s3)} sequences")
    print(f"  Length range: {s3['N'].min()}-{s3['N'].max()}, median {s3['N'].median():.0f}")
    print(f"  SCD range:    {s3['scd'].min():.2f} to {s3['scd'].max():.2f}")
    print(f"  FCR range:    {s3['fcr'].min():.2f} to {s3['fcr'].max():.2f}")
    print(f"  Rg range (nm): {s3['Rg/nm'].min():.2f}-{s3['Rg/nm'].max():.2f}")
    
    print(f"\nSubset 4: {len(s4)} sequences (all FCR<{FCR_THRESHOLD})")
    print(f"  Length range: {s4['N'].min()}-{s4['N'].max()}")
    print(f"  Aromatic fraction range: {s4['faro'].min():.3f} to {s4['faro'].max():.3f}")
    print(f"  Rg range (nm): {s4['Rg/nm'].min():.2f}-{s4['Rg/nm'].max():.2f}")

if __name__ == '__main__':
    main()
