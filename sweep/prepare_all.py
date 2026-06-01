"""Pre-prepare all (seq_id, salt) sim directories on the login node."""
import pandas as pd
import subprocess
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys

jobs = pd.read_csv('jobs.csv')
print(f"Pre-preparing {len(jobs)} simulations...")

def prepare_one(row):
    seq_id, salt = row['seq_id'], row['salt']
    salt_mm = int(salt * 1000)
    sysname = f"{seq_id}_salt{salt_mm:03d}mM"
    
    # Skip if already prepared (idempotent)
    if os.path.exists(f"sims/{sysname}/config.yaml"):
        return sysname, "skip"
    
    result = subprocess.run(
        ['python', 'prepare_one.py', '--seq_id', seq_id, '--salt', str(salt)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return sysname, f"FAIL: {result.stderr[:200]}"
    return sysname, "ok"

# Sequential is plenty fast; localCIDER calls during prep aren't the bottleneck
n_done = 0
n_ok = 0
n_skip = 0
n_fail = 0
fails = []

for _, row in jobs.iterrows():
    sysname, status = prepare_one(row)
    n_done += 1
    if status == "ok":
        n_ok += 1
    elif status == "skip":
        n_skip += 1
    else:
        n_fail += 1
        fails.append((sysname, status))
    
    if n_done % 100 == 0:
        print(f"  {n_done}/{len(jobs)} — ok={n_ok} skip={n_skip} fail={n_fail}", flush=True)

print(f"\nDone: {n_ok} prepared, {n_skip} already prepared, {n_fail} failures")
if fails:
    print("\nFailures:")
    for sysname, err in fails[:20]:
        print(f"  {sysname}: {err}")
    if len(fails) > 20:
        print(f"  ... and {len(fails) - 20} more")
