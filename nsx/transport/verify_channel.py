''' Verification of nsx.transport on a straight channel, through the OFFLINE path:

    mpirun -n 2 python -m nsx.transport.verify_channel --out /tmp/channel

A box [0, L] x [0, H]^2 (tags: 3 inlet x=0, 2 outlet x=L, 1 wall) carries the uniform,
time-varying velocity u = (U(t), 0, 0), U = U0 (1 + a t), written to u.h5 every
`--write-dt` like a flow run's output, then read back by StoredVelocity (KD-tree match,
linear interpolation in time; exact here since U is linear in t). A step c = 1 enters
at x = 0. The 1D solution with front X(t) = U0 (t + a t^2/2) is
    c = 0.5 erfc((x - X)/(2 sqrt(D t)))       (+ an exponentially small term)
Checks: the front position (c = 0.5 crossing) against X, the L2 error against the
erfc profile, c within [0, 1] (to SUPG overshoot), and the mass balance residual
(div u = 0 here, so the advective and conservative forms must agree).
'''
import argparse
from math import erfc, sqrt
from pathlib import Path

import numpy as np
from dolfinx import fem, io, mesh as dmesh
from mpi4py import MPI

from .offline import run


def build(out, L, H, nx, ny, U0, a, T, wdt, comm):
    msh = dmesh.create_box(comm, [[0, 0, 0], [L, H, H]], [nx, ny, ny], dmesh.CellType.tetrahedron)
    fdim = 2
    msh.topology.create_connectivity(fdim, 3)
    loc = [(3, lambda x: np.isclose(x[0], 0)), (2, lambda x: np.isclose(x[0], L))]
    allf = dmesh.exterior_facet_indices(msh.topology)
    tag = np.ones(len(allf), dtype=np.int32)
    for t, f in loc:
        fac = dmesh.locate_entities_boundary(msh, fdim, f)
        tag[np.isin(allf, fac)] = t
    ft = dmesh.meshtags(msh, fdim, allf, tag)
    ft.name = 'boundaries'
    out.mkdir(parents=True, exist_ok=True)
    with io.XDMFFile(comm, str(out/'channel.xdmf'), 'w') as f:
        f.write_mesh(msh)
        msh.topology.create_connectivity(fdim, 3)
        f.write_meshtags(ft, msh.geometry)
    W = fem.functionspace(msh, ('Lagrange', 1, (3,)))
    u = fem.Function(W, name='u')
    with io.XDMFFile(comm, str(out/'u.xdmf'), 'w') as f:
        f.write_mesh(msh)
        for t in np.arange(0, T + 0.5*wdt, wdt):
            U = U0*(1 + a*t)
            u.interpolate(lambda x: np.vstack([np.full(x.shape[1], U), 0*x[0], 0*x[0]]))
            f.write_function(u, float(t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='./channel_verify')
    ap.add_argument('--n', type=int, default=80, help='cells along the channel')
    ap.add_argument('--dt', type=float, default=0.005)
    ap.add_argument('--write-dt', type=float, default=0.05)
    ap.add_argument('--D', type=float, default=1e-3)
    ap.add_argument('--form', default='advective')
    ap.add_argument('--sc', type=float, default=0.0, help='shock_capturing coefficient')
    a = ap.parse_args()
    comm = MPI.COMM_WORLD
    out = Path(a.out)
    L, H, U0, acc, T = 2.0, 0.25, 1.0, 0.5, 1.0
    build(out, L, H, a.n, max(2, a.n//16), U0, acc, T, a.write_dt, comm)
    cfg = {'mesh': str(out/'channel.xdmf'),
           'velocity': {'file': str(out/'u.h5')},
           'time': {'T': T, 'dt': a.dt, 'write_dt': 0.25},
           'transport': {'D': a.D, 'form': a.form, 'shock_capturing': a.sc, 'inflow': [{'id': 3, 'c': 1.0}], 'outlets': [2]},
           'output': {'path': str(out/('transport_' + a.form))}}
    tr = run(cfg, comm)

    X = U0*(T + acc*T**2/2)
    x = tr.V.tabulate_dof_coordinates()[:tr.V.dofmap.index_map.size_local, 0]
    c = tr.c.x.array[:tr.V.dofmap.index_map.size_local]
    ex = np.array([0.5*erfc((xi - X)/(2*sqrt(a.D*T))) for xi in x])
    err = sqrt(comm.allreduce(((c - ex)**2).sum(), op=MPI.SUM)/comm.allreduce(len(x), op=MPI.SUM))
    xs, cs = comm.gather(x, root=0), comm.gather(c, root=0)
    lo, hi = tr.range()
    if comm.rank == 0:
        xs, cs = np.concatenate(xs), np.concatenate(cs)
        xb = np.unique(np.round(xs, 9)); cb = np.array([cs[np.isclose(xs, v)].mean() for v in xb])
        k = np.where((cb[:-1] >= 0.5) & (cb[1:] < 0.5))[0][0]
        xf = xb[k] + (cb[k] - 0.5)/(cb[k] - cb[k + 1])*(xb[k + 1] - xb[k])
        res = np.loadtxt(out/('transport_' + a.form)/'balance.csv', delimiter=',', skiprows=1)
        M = res[-1, 1]
        print('front x(c=0.5) %.4f vs exact %.4f (error %.2e, h = %.4f)' % (xf, X, xf - X, L/a.n))
        print('RMS error vs erfc profile %.3e; c in [%.4f, %.4f]' % (err, lo, hi))
        print('mass M %.5f vs exact %.5f; max |balance residual| %.2e (relative to inflow rate %.2e)'
              % (M, H*H*X, np.abs(res[:, -4]).max(), H*H*U0))


if __name__ == '__main__':
    main()
