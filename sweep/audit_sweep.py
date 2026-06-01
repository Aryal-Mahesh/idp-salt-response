"""Audit which (seq_id, salt) pairs have completed conf_prop.csv files."""
import pandas as pd
import os
import sys

jobs = pd.read_csv('jobs.csv')

done = []
missing = []
for _, row in jobs.iterrows():
    seq_id, salt = row['seq_id'], row['salt']
    salt_mm = int(salt * 1000)
    sysname = f"{seq_id}_salt{salt_mm:03d}mM"
    if os.path.exists(f"data/{sysname}/conf_prop.csv"):
        done.append(sysname)
    else:
        missing.append((row['seq_id'], row['salt'], sysname))

print(f"Total jobs: {len(jobs)}")
print(f"Completed:  {len(done)}  ({100*len(done)/len(jobs):.1f}%)")
print(f"Missing:    {len(missing)}")

if missing and len(sys.argv) > 1 and sys.argv[1] == '--list':
    print("\nMissing jobs:")
    for seq_id, salt, sysname in missing[:50]:
        print(f"  {sysname}")
    if len(missing) > 50:
        print(f"  ... and {len(missing) - 50} more")

if missing and len(sys.argv) > 1 and sys.argv[1] == '--write-missing':
    df_missing = pd.DataFrame(missing, columns=['seq_id', 'salt', 'sysname'])
    df_missing.to_csv('missing_jobs.csv', index=False)
    print(f"\nWrote missing_jobs.csv ({len(missing)} rows) for resubmission.")
