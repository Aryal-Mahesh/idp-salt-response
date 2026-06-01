#!/usr/bin/env python3
"""Generate Supplementary Table S1: per-subset descriptor ranges.
Run on the cluster from the sweep/ directory: python make_table_s1.py
"""
import pandas as pd

df = pd.read_csv("master_table.csv")

# --- column name resolution (edit here if auto-detect picks wrong) ---
def pick(df, *candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of {candidates} found. Columns: {list(df.columns)}")

col_subset = pick(df, "subset", "Subset", "group")
col_len    = pick(df, "length", "N", "seq_length", "Nres")
col_fcr    = pick(df, "fcr", "FCR")
col_ncpr   = pick(df, "ncpr", "NCPR")
col_kappa  = pick(df, "kappa", "Kappa")
col_scd    = pick(df, "scd", "SCD")

# --- pretty subset names + display order ---
name_map = {
    "kappa_variant":   r"$\kappa$-variants",
    "ncpr_series":     "NCPR series",
    "idrome_stratified": "IDRome (stratified)",
    "idrome_lowfcr":   "IDRome (low-FCR)",
}
order = ["kappa_variant", "ncpr_series", "idrome_stratified", "idrome_lowfcr"]

def rng(s):
    return f"{s.min():.2f}--{s.max():.2f}"

def len_rng(s):
    return f"{int(s.min())}--{int(s.max())}"

rows = []
for key in order:
    sub = df[df[col_subset] == key]
    if len(sub) == 0:
        print(f"WARNING: no rows for subset '{key}' (check label spelling)")
        continue
    rows.append([
        name_map.get(key, key), len(sub),
        len_rng(sub[col_len]), rng(sub[col_fcr]),
        rng(sub[col_ncpr]), rng(sub[col_kappa]), rng(sub[col_scd]),
    ])

# full-library row
rows.append([
    r"\textbf{Full library}", len(df),
    len_rng(df[col_len]), rng(df[col_fcr]),
    rng(df[col_ncpr]), rng(df[col_kappa]), rng(df[col_scd]),
])

# --- emit LaTeX ---
print(r"\begin{table}[t]")
print(r"\centering")
print(r"\caption{Composition and charge-patterning ranges of the four "
      r"sequence-library subsets. Values are reported as min--max across "
      r"sequences in each subset. $N$, sequence length; $\FCR$, fraction of "
      r"charged residues; $\NCPR$, net charge per residue; $\kappa$, "
      r"charge-patterning parameter~\cite{daspappu2013}; $\SCD$, sequence "
      r"charge decoration~\cite{sawleghosh2015}.}")
print(r"\label{tab:library}")
print(r"\begin{tabular}{lcccccc}")
print(r"\hline")
print(r"Subset & Count & $N$ & $\FCR$ & $\NCPR$ & $\kappa$ & $\SCD$ \\")
print(r"\hline")
for r in rows:
    print(" & ".join(str(x) for x in r) + r" \\")
print(r"\hline")
print(r"\end{tabular}")
print(r"\end{table}")
