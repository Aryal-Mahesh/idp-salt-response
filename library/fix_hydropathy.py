"""
Recompute mean_hydropathy uniformly across the library using Kyte-Doolittle.
Overwrites the inconsistent column inherited from IDRome.
"""
import pandas as pd
import numpy as np

KD_HYDROPATHY = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'E': -3.5, 'Q': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
}

def kd_mean(seq):
    return float(np.mean([KD_HYDROPATHY.get(a, 0.0) for a in seq]))

df = pd.read_csv('library.csv')

# Save the original IDRome lambda (which is also useful as a separate column)
df['mean_lambda_calvados'] = df['mean_hydropathy']

# Recompute mean_hydropathy uniformly with Kyte-Doolittle for ALL sequences
df['mean_hydropathy'] = df['sequence'].apply(kd_mean)

df.to_csv('library.csv', index=False)

# Verify
print("Uniform Kyte-Doolittle mean hydropathy by subset:")
print(df.groupby('subset')['mean_hydropathy'].describe()[['mean','min','max']].round(2))
print()
print("CALVADOS lambda (preserved as mean_lambda_calvados) by subset:")
print(df.groupby('subset')['mean_lambda_calvados'].describe()[['mean','min','max']].round(2))
print()
print(f"Saved updated library.csv with {len(df)} sequences and {len(df.columns)} columns")
