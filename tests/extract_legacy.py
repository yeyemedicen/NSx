"""
Extract velocity and pressure from legacy FEniCS checkpoints into a numpy archive.

Run from /Users/yeye/NavierStokes with:
    PYTHONPATH=/Users/yeye/Common conda run -n FenicsLegacy \
        python /Users/yeye/NSx/tests/extract_legacy.py
"""
import sys, os
sys.path.insert(0, '/Users/yeye/Common')

from dolfin import (Mesh, HDF5File, VectorFunctionSpace, FunctionSpace,
                    Function, MPI)
import numpy as np

MESH_FILE  = '/Users/yeye/NavierStokes/tests/rectangle.h5'
CHECKPOINT = '/Users/yeye/NavierStokes/tests/results/checkpoint'
OUTPUT     = '/Users/yeye/NSx/tests/results/legacy_fields.npz'

# ── load mesh ─────────────────────────────────────────────────────────────────
mesh = Mesh()
with HDF5File(MPI.comm_world, MESH_FILE, 'r') as hdf:
    hdf.read(mesh, '/mesh', False)

V = VectorFunctionSpace(mesh, 'CG', 1)
Q = FunctionSpace(mesh, 'CG', 1)

# vertex coordinates in FEniCS vertex-index order, shape (N, 2)
coords = mesh.coordinates().copy()
N = len(coords)

u = Function(V)
p = Function(Q)

steps = sorted(int(d) for d in os.listdir(CHECKPOINT) if d.isdigit())
print(f'Found {len(steps)} checkpoints: {steps[0]} … {steps[-1]}')

arrays = {'coords': coords, 'steps': np.array(steps)}

for step in steps:
    base = os.path.join(CHECKPOINT, str(step))

    with HDF5File(MPI.comm_world, base + '/u.h5', 'r') as hdf:
        hdf.read(u, '/u')
    with HDF5File(MPI.comm_world, base + '/p.h5', 'r') as hdf:
        hdf.read(p, '/p')

    # compute_vertex_values returns values at mesh vertices in vertex-index order.
    # For VectorFunctionSpace: first N values are ux, next N are uy.
    u_vtx = u.compute_vertex_values(mesh).reshape(2, N).T  # (N, 2)
    p_vtx = p.compute_vertex_values(mesh)                   # (N,)

    arrays[f'u_{step}'] = u_vtx.astype(np.float64)
    arrays[f'p_{step}'] = p_vtx.astype(np.float64)

    u_norm = np.linalg.norm(u_vtx)
    p_norm = np.linalg.norm(p_vtx)
    print(f'  step {step:4d}: |u|={u_norm:.6e}  |p|={p_norm:.6e}')

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
np.savez(OUTPUT, **arrays)
print(f'\nSaved → {OUTPUT}')
