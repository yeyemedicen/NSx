''' Offline contrast transport on a finished flow run:

    mpirun -n 4 python -m nsx.transport.offline transport.yaml

transport.yaml:

    mesh: /path/mesh.xdmf                  # the flow run's mesh (with its boundary tags)
    velocity:
      file: /path/results/u.h5             # the flow run's velocity output
      replay: {t0: 0.8, T: 0.8}            # optional: replay one stored cycle periodically
      ct_correction:                       # optional: advect with the CT field w = u* - (dt/rho) grad p
        pressure: /path/results/p.h5       #   rebuilt from the saved UPDATE u and p (velocity.CTVelocity)
        dt: 5.0e-4                         #   the FLOW run's dt and rho
        rho: 1.06
        dirichlet: [1]                     #   the flow run's velocity-Dirichlet tags (no-slip wall)
      scale: 1.0                           # optional (2026-11-21): advect with k x the stored field
      time_scale: 1.0                      # optional: play the stored flow s times faster: transport
                                           #   time t reads stored (replay) time s*t. scale = time_scale
                                           #   = k is the exactly self-similar flow (Q x k, period T/k).
    time: {T: 4.0, dt: 0.001, write_dt: 0.01}
    transport:                             # nsx.transport.ScalarTransport options
      D: 2.03e-5                           # cm2/s
      theta: 0.5
      supg: true
      form: advective                      # or conservative, skew
      inflow: [{id: 3, c: "1.0*(t < 2.0)"}]
      outlets: [2, 101, 102]
      backflow: natural                    # or zero
      sources: [{center: [x, y, z], radius: 0.05, rate: "1.0*(t < 2.0)"}]
    output: {path: ./results_transport/}

Writes c.xdmf/c.h5 every write_dt and balance.csv every step: t, M = int c,
the outward solute flux of every tag, int c div u, int s, and the residual
dM/dt + sum(flux) - c_div_u - source (advective form; without c_div_u for conservative).
The velocity is linearly interpolated in time between stored frames (exact when dt
is the write interval).
'''
import argparse
import time as _time
from pathlib import Path

import numpy as np
import yaml
from dolfinx import io
from mpi4py import MPI

from common.inout import read_mesh
from .scalar import ScalarTransport
from .velocity import StoredVelocity, CTVelocity


def run(cfg, comm=MPI.COMM_WORLD):
    mesh, _, ft = read_mesh(cfg['mesh'])
    vcfg = cfg['velocity']
    topts = dict(cfg.get('transport', {}))
    ctc = vcfg.get('ct_correction')
    if ctc:
        topts['pressure_correction'] = float(ctc['dt'])/float(ctc['rho'])
    tr = ScalarTransport(mesh, ft, topts)
    if ctc:
        cv = CTVelocity(vcfg['file'], ctc['pressure'], tr.W, tr.V, ft, ctc['dt'], ctc['rho'],
                        dirichlet=ctc.get('dirichlet', [1]), replay=vcfg.get('replay'),
                        name_u=vcfg.get('name', 'u'))
        assign = lambda t: cv.assign(tr.u, tr.p, t)
    else:
        sv = StoredVelocity(vcfg['file'], tr.W, name=vcfg.get('name', 'u'), replay=vcfg.get('replay'))
        assign = lambda t: sv.assign(tr.u, t)
    kv, sv_ = float(vcfg.get('scale', 1.0)), float(vcfg.get('time_scale', 1.0))
    if kv != 1.0 or sv_ != 1.0:
        # w = u* - (dt/rho) grad p is linear in (u*, p): scaling both scales the advecting field exactly
        base = assign

        def assign(t):
            base(sv_*t)
            fs = [tr.u] + ([tr.p] if ctc else [])
            for f in fs:
                f.x.array[:] *= kv
                f.x.scatter_forward()
        if comm.rank == 0:
            print('velocity: scale %g, time_scale %g' % (kv, sv_), flush=True)
    T, dt = float(cfg['time']['T']), float(cfg['time']['dt'])
    wdt = float(cfg['time'].get('write_dt', dt))
    th = float(tr.o['theta'])
    out = Path(cfg.get('output', {}).get('path', './results_transport/'))
    if comm.rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        with open(out/'transport.yaml', 'w') as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
    comm.barrier()

    xdmf = io.XDMFFile(comm, str(out/'c.xdmf'), 'w')
    xdmf.write_mesh(mesh)
    xdmf.write_function(tr.c, 0.0)
    tags = sorted(tr._f_flux)
    bal = open(out/'balance.csv', 'w') if comm.rank == 0 else None
    if bal:
        bal.write('t,M,' + ','.join('flux%d' % t for t in tags)
                  + ',c_div_u,source,residual,c_min,c_max,its\n')
    nsteps = int(round(T/dt))
    nwrite = max(1, int(round(wdt/dt)))
    t0w = _time.time()
    for i in range(1, nsteps + 1):
        assign((i - 1 + th)*dt)                   # velocity at t^{n+theta}
        tr.set_boundary_values(tr.t + th*dt)
        b0 = tr.balance()                         # old c, this step's u and g
        its = tr.step(dt)
        b1 = tr.balance()                         # new c, same u and g
        # trapezoid of the rates at the two ends of the step
        kdiv = {'advective': 1.0, 'skew': 0.5}.get(tr.o['form'], 0.0)
        rate = lambda b: (-sum(b['flux'].values()) + b['source'] + kdiv*b['c_div_u'])
        res = (b1['M'] - b0['M'])/dt - 0.5*(rate(b0) + rate(b1))
        lo, hi = tr.range()
        if bal:
            bal.write('%.6g,%.8e,' % (tr.t, b1['M']) + ','.join('%.6e' % b1['flux'][t] for t in tags)
                      + ',%.6e,%.6e,%.6e,%.4e,%.4e,%d\n' % (b1['c_div_u'], b1['source'], res, lo, hi, its))
        if i % nwrite == 0:
            xdmf.write_function(tr.c, tr.t)
            xdmf.flush()        # keep c.h5 readable if the run is killed (see inline.py)
            if comm.rank == 0:
                bal.flush()
                el = _time.time() - t0w
                print('t=%.4f  M=%.4e  c in [%.3g, %.3g]  its %d  %.2f s/step, ETA %.0f s'
                      % (tr.t, b1['M'], lo, hi, its, el/i, el/i*(nsteps - i)), flush=True)
    xdmf.close()
    if bal:
        bal.close()
    return tr


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('config')
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    run(cfg)


if __name__ == '__main__':
    main()
