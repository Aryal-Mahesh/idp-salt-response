"""Prepare one (seq_id, salt) simulation. Generalized version for the full sweep."""
import os
import subprocess
import argparse
from calvados.cfg import Config, Components

parser = argparse.ArgumentParser()
parser.add_argument('--seq_id', required=True, type=str)
parser.add_argument('--salt', required=True, type=float, help='Ionic strength in M')
args = parser.parse_args()

cwd = os.getcwd()
sysname = f'{args.seq_id}_salt{int(args.salt*1000):03d}mM'

# Choose box size adaptively from sequence length (estimate from FASTA)
fasta_path = f'{cwd}/input/library.fasta'
seq_len = None
with open(fasta_path) as f:
    capture = False
    for line in f:
        line = line.strip()
        if line.startswith('>'):
            if capture:
                break
            if line[1:] == args.seq_id:
                capture = True
        elif capture:
            seq_len = len(line)
            break

if seq_len is None:
    raise ValueError(f"seq_id {args.seq_id} not found in {fasta_path}")

# Adaptive box: 4 * R_max heuristic, minimum 50 nm
L = max(50, int(4 * (seq_len ** 0.6)))

N_save = 7000
N_frames = 1010
residues_file = f'{cwd}/input/residues_CALVADOS2.csv'

config = Config(
    sysname=sysname,
    box=[L, L, L],
    temp=293.15,
    ionic=args.salt,
    pH=7.5,
    topol='center',
    wfreq=N_save,
    steps=N_frames * N_save,
    runtime=0,
    platform='CUDA',
    restart='checkpoint',
    frestart='restart.chk',
    verbose=True,
)

path = f'{cwd}/sims/{sysname}'
subprocess.run(f'mkdir -p {path}', shell=True)
subprocess.run(f'mkdir -p data/{sysname}', shell=True)

analyses = f"""
from calvados.analysis import save_conf_prop
save_conf_prop(path="{path:s}",name="{sysname:s}",residues_file="{residues_file:s}",output_path="data/{sysname:s}",start=10,is_idr=True,select='all')
"""

config.write(path, name='config.yaml', analyses=analyses)

components = Components(
    molecule_type='protein',
    nmol=1,
    restraint=False,
    charge_termini='both',
    fresidues=residues_file,
    ffasta=fasta_path,
)
components.add(name=args.seq_id)
components.write(path, name='components.yaml')

print(f"Prepared {sysname} (length {seq_len}, box {L} nm)")
