"""
Extract velocity and pressure from DOLFINx checkpoints into a numpy archive.

Run from /Users/yeye/NSx with:
    PYTHONPATH=/Users/yeye/Common conda run -n dolfinX \
        python tests/extract_dolfinx.py
"""
import os
import numpy as np
import h5py
from mpi4py import MPI
from dolfinx.io import XDMFFile

MESH_FILE  = '/Users/yeye/NSx/tests/rectangle_shared.xdmf'
CHECKPOINT = '/Users/yeye/NSx/tests/results/checkpoint'
OUTPUT     = '/Users/yeye/NSx/tests/results/dolfinx_fields.npz'

# ── load mesh geometry ────────────────────────────────────────────────────────
with XDMFFile(MPI.COMM_WORLD, MESH_FILE, 'r') as xf:
    mesh = xf.read_mesh(name='mesh')

# For CG1 elements DOF coords == geometry node coords (verified to machine eps).
# mesh.geometry.x is (N, 3); take first 2 columns for 2-D coords.
coords = mesh.geometry.x[:, :2].copy()  # (N, 2)

steps = sorted(int(d) for d in os.listdir(CHECKPOINT) if d.isdigit())
print(f'Found {len(steps)} checkpoints: {steps[0]} … {steps[-1]}')

arrays = {'coords': coords, 'steps': np.array(steps)}

for step in steps:
    base = os.path.join(CHECKPOINT, str(step))

    # DOLFINx write_function stores values at geometry nodes.
    # Shape (N, 3) for 2-D vector (3rd col = 0), (N, 1) for scalar.
    with h5py.File(base + '/u.h5', 'r') as f:
        key = next(iter(f['Function']['u'].keys()))
        u_vtx = f['Function']['u'][key][:, :2].astype(np.float64)  # (N, 2)

    with h5py.File(base + '/p.h5', 'r') as f:
        key = next(iter(f['Function']['p'].keys()))
        p_vtx = f['Function']['p'][key][:, 0].astype(np.float64)   # (N,)

    arrays[f'u_{step}'] = u_vtx
    arrays[f'p_{step}'] = p_vtx

    u_norm = np.linalg.norm(u_vtx)
    p_norm = np.linalg.norm(p_vtx)
    print(f'  step {step:4d}: |u|={u_norm:.6e}  |p|={p_norm:.6e}')

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
np.savez(OUTPUT, **arrays)
print(f'\nSaved → {OUTPUT}')
