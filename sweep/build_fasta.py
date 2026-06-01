"""Convert library.csv to a single FASTA file for CALVADOS to read."""
import pandas as pd

df = pd.read_csv('library.csv')
print(f"Building FASTA for {len(df)} sequences")

with open('input/library.fasta', 'w') as f:
    for _, row in df.iterrows():
        f.write(f">{row['seq_id']}\n{row['sequence']}\n")

print("Wrote input/library.fasta")
