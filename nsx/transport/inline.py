''' Contrast transport solved INSIDE an NSx run (one sub-step per flow step), optionally two-way.

Input (NSx options; JellyFSI: under the fluid block):

    contrast:
      enabled: true
      t0: 1.6                 # flow time of exam time 0 (default: the flow time at start-up)
      D: 2.03e-5
      write_dt: 0.01          # c.xdmf interval (exam output); balance.csv every step
      numerics: {theta: 0.5, supg: true, shock_capturing: 0.2, backflow: zero}
      inflow: [{id: 3, c: 0.0}]
      outlets: [2, 101, ...]
      injection:              # one catheter tip
        center: [x, y, z]
        radius: 0.15
        shape: gaussian
        rate: "1.0*(t >= 0.0)*(t < 2.0)"   # injectate VOLUME rate [mL/s], t = EXAM time
        c_inj: 1.0
        volume: true          # TWO-WAY mass: the projection gets div u = s_v (false: massless)
        momentum:             # catheter jet: rho s_v (u_inj - u), u_inj = q(t)/A along direction
          enabled: true
          direction: [ex, ey, ez]          # vessel axis, downstream (MeasureIt angio.find_tip)
          catheter_diameter: 0.14          # lumen [cm] (6F diagnostic ~1.4 mm)
      mixture:                # TWO-WAY through the properties (problem.py, lagged c^n)
        density: 1.36         # of the injectate; = blood density switches the body force off
        viscosity: 0.10       # of the injectate; = blood viscosity switches it off
        viscosity_mixing: log # ln mu = (1-c) ln mu_b + c ln mu_c, c clipped to [0, 1]
        gravity: [0, -981, 0] # Boussinesq: (rho(c) - rho_b) g in the momentum, rho_b elsewhere

The flow is the fixed-mesh CT projection. The contrast is advected with the CT field
w = u* - (dt/rho) grad p (P1 + P0), which the projection makes weakly divergence free (with
`volume`: weak div w = s_v, the SAME s_v as the transport source), so no solute is created
(nsx.transport.CTVelocity explains why the nodal update u is not used). Time: Crank-Nicolson
with w^{n+1/2} = (w^n + w^{n+1})/2 and the source rate averaged over the step, which is exactly
the divergence of that w.

Output in <write_path>/contrast/: c.xdmf/h5 every write_dt, balance.csv every step (columns as
nsx.transport.offline; t is EXAM time). Checkpoints: c, c_n, c_nm1 and the previous u*, p go to
<checkpoint>/c.h5 (by node coordinates, any rank count); a restart from a checkpoint without
c.h5 (a flow-only run) starts with c = 0.
'''
from pathlib import Path
import os

import h5py
import numpy as np
import ufl
from dolfinx import fem, io
from dolfinx.fem import petsc as fem_petsc
from mpi4py import MPI
from petsc4py import PETSc
from scipy.spatial import cKDTree

from .scalar import ScalarTransport


def contrast_options(options):
    c = options.get('contrast') or {}
    return c if c.get('enabled', False) else None


class InlineContrast:
    def __init__(self, solver):
        self.s = solver
        o = contrast_options(solver.options)
        self.o = o
        if solver._using_ale:
            raise NotImplementedError('contrast: fixed mesh only (ALE transport not built yet)')
        if solver.options['timemarching']['fractionalstep']['scheme'] != 'CT':
            raise NotImplementedError('contrast: CT scheme only')
        if not solver.options['fem'].get('DC_in_update', False):
            raise ValueError('contrast needs fem: DC_in_update: true (u* = u - du, wall du = 0)')
        mesh = solver.u.function_space.mesh
        self.comm = mesh.comm
        self.dt = float(solver.options['timemarching']['dt'])
        self.rho = float(solver.options['fluid']['density'])
        num = dict(o.get('numerics') or {})
        inj = dict(o.get('injection') or {})
        mom = dict(inj.get('momentum') or {})
        self.jet = bool(mom.get('enabled', False))
        if self.jet and not inj.get('volume', False):
            raise ValueError('contrast injection momentum needs volume: true')
        src = []
        if inj:
            src = [{'center': inj['center'], 'radius': inj['radius'],
                    'shape': inj.get('shape', 'gaussian'), 'rate': inj['rate'],
                    'c': inj.get('c_inj', 1.0), 'volume': bool(inj.get('volume', False))}]
            if not src[0]['volume']:      # massless: 'rate' becomes the solute rate
                src[0]['rate'] = '(%s)*%r' % (inj['rate'], float(inj.get('c_inj', 1.0)))
        topts = {'D': float(o.get('D', 0.0)), 'form': 'advective',
                 'theta': float(num.get('theta', 0.5)), 'supg': bool(num.get('supg', True)),
                 'shock_capturing': float(num.get('shock_capturing', 0.0)),
                 'backflow': num.get('backflow', 'zero'),
                 'inflow': o.get('inflow', []), 'outlets': o.get('outlets', []),
                 'sources': src, 'pressure_correction': self.dt/self.rho}
        self.tr = tr = ScalarTransport(mesh, solver.bnds, topts)
        self.volume = bool(inj.get('volume', False))
        self.rate = tr._sources[0][2] if src else (lambda t: 0.0)
        # exam time 0: given, or the flow time of the first step minus dt (set lazily: JellyFSI
        # sets the restart clock AFTER the solver is built)
        self.t0 = float(o['t0']) if 't0' in o else None
        # u* and p of the previous and current step (w^{n+1/2} = mean)
        W, V = tr.W, tr.V
        self.us_old, self.us_new = fem.Function(W), fem.Function(W)
        self.p_old, self.p_new = fem.Function(V), fem.Function(V)
        self._have_old = False
        # the projection's source: k rho (s_v, q) with k = 1/dt (problem.form_pressure:
        # laplacian (grad p, grad q) = -k rho (div u*, q)  ->  + k rho (s_v, q))
        self.gv = fem.Constant(mesh, 0.0)            # s_v rate used by the PRESSURE step
        if self.volume:
            ind, vol = tr._sources[0][0], tr._sources[0][1]
            q = ufl.TestFunction(solver.p.function_space)
            self._f_ps = fem.form((self.rho/self.dt)*self.gv/vol*ind*q*ufl.dx)
        # catheter jet (two-way momentum): the injectate enters with u_inj = q(t)/A e, A the catheter
        # lumen; momentum source rho s_v (u_inj - u^n) [+ rho s_v u^n/2 with skew convection, which
        # adds rho/2 (div u) u* = rho s_v u*/2 that a div-free derivation assumes zero]
        self.gm = fem.Constant(mesh, 0.0)
        self._f_mom = None
        if self.jet:
            e = np.asarray(mom['direction'], float); e = e/np.linalg.norm(e)
            A = np.pi*float(mom['catheter_diameter'])**2/4
            ind, vol = tr._sources[0][0], tr._sources[0][1]
            Vi = solver.u_lst[0].function_space
            vi = ufl.TestFunction(Vi)
            skew = bool(solver.options['fem'].get('convection_skew_symmetric', False))
            cu = 0.5 if skew else 1.0
            sv = self.gm/vol*ind
            self._f_mom = [fem.form(self.rho*sv*(self.gm/A*float(e[i]) - cu*solver.u_lst[i])*vi*ufl.dx)
                           for i in range(solver.ndim)]
            if self.comm.rank == 0:
                print('[contrast] catheter jet: direction %s, lumen %.4g cm2 (u_inj = q/A)'
                      % (np.round(e, 4).tolist(), A), flush=True)
        # where the problem's c lives (mixture properties, lagged)
        self.c_prob = getattr(solver, 'contrast_c', None)
        # output
        wp = Path(solver.options['io']['write_path'])
        self.out = wp/'contrast'
        if self.comm.rank == 0:
            self.out.mkdir(parents=True, exist_ok=True)
        self.comm.barrier()
        self.write_dt = float(o.get('write_dt', 0.01))
        self._restore()
        new = not self._restored
        self._new = new
        self.xdmf = None          # opened at the first step (the restart clock is set by then)
        self._mesh = mesh
        tags = sorted(tr._f_flux)
        self._tags = tags
        self.bal = None
        if self.comm.rank == 0:
            fresh = not (self.out/'balance.csv').exists() or new
            self.bal = open(self.out/'balance.csv', 'w' if fresh else 'a')
            if fresh:
                self.bal.write('t,M,' + ','.join('flux%d' % t for t in tags)
                               + ',c_div_u,source,residual,c_min,c_max,its\n')
        self._nwrite = max(1, int(round(self.write_dt/self.dt)))
        self._i = 0
        if self.comm.rank == 0:
            print('[contrast] volume (two-way mass)=%s, t0=%s, k=%g, restored=%s'
                  % (self.volume, self.t0, self.dt/self.rho, self._restored), flush=True)

    def texam(self, t=None):
        if self.t0 is None:
            self.t0 = float(self.s.t) - self.dt
        return (self.s.t if t is None else t) - self.t0

    # ---- tentative step: catheter-jet momentum, rate at t^{n+1} ---------------
    def momentum_source(self, i):
        ''' Component i of the jet's momentum source (a new vector) or None. '''
        if self._f_mom is None:
            return None
        self.gm.value = self.rate(round(self.texam(), 12))
        if self.gm.value == 0.0:
            return None
        b = fem_petsc.assemble_vector(self._f_mom[i])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        return b

    # ---- pressure step: + k rho (s_v(t^{n+1}), q) -------------------------------------
    def pressure_source(self):
        ''' The projection's extra RHS (a new ghosted vector) or None. Rate at t^{n+1}. '''
        if not self.volume:
            return None
        self.gv.value = self.rate(round(self.texam(), 12))
        if self.gv.value == 0.0:
            return None
        b = fem_petsc.assemble_vector(self._f_ps)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        return b

    # ---- the transport sub-step, after the velocity update ----------------------------
    def step(self):
        s, tr = self.s, self.tr
        # u* = u - du (du = the update increment, DC_in_update: 0 on the wall)
        for i in range(s.ndim):
            us_i = s.u_lst[i].x.array - s.du[i].x.array
            self.us_new.x.array[i::s.ndim] = us_i
        self.us_new.x.scatter_forward()
        self.p_new.x.array[:] = s.p.x.array
        first = not self._have_old
        if first:             # no previous field (fresh start / flow-only restart): w^{n+1} alone
            self.us_old.x.array[:] = self.us_new.x.array
            self.p_old.x.array[:] = self.p_new.x.array
            self._have_old = True
        tr.u.x.array[:] = 0.5*(self.us_old.x.array + self.us_new.x.array)
        tr.p.x.array[:] = 0.5*(self.p_old.x.array + self.p_new.x.array)
        tr.u.x.scatter_forward(); tr.p.x.scatter_forward()
        te1 = round(self.texam(), 12); te0 = round(te1 - self.dt, 12)
        if self.xdmf is None:
            name = 'c.xdmf' if self._new else 'c_from_%.4f.xdmf' % te0
            self.xdmf = io.XDMFFile(self.comm, str(self.out/name), 'w')
            self.xdmf.write_mesh(self._mesh)
            self.xdmf.write_function(tr.c, te0)
            self.xdmf.flush()
        th = float(tr.o['theta'])
        tr.t = te0
        tr.set_boundary_values(te0 + th*self.dt)
        if self.volume:     # = the divergence of w^{n+1/2}: the mean of the two projections' rates
            tr._sources[0][3].value = (self.rate(te1) if first
                                       else 0.5*(self.rate(te0) + self.rate(te1)))
        b0 = tr.balance()
        # step() re-sets the boundary values at t^{n+theta}; keep the averaged rate
        its = self._solve()
        b1 = tr.balance()
        rate = lambda b: (-sum(b['flux'].values()) + b['source'] + b['c_div_u'])
        res = (b1['M'] - b0['M'])/self.dt - 0.5*(rate(b0) + rate(b1))
        lo, hi = tr.range()
        if self.bal:
            self.bal.write('%.6g,%.8e,' % (te1, b1['M']) + ','.join('%.6e' % b1['flux'][t] for t in self._tags)
                           + ',%.6e,%.6e,%.6e,%.4e,%.4e,%d\n' % (b1['c_div_u'], b1['source'], res, lo, hi, its))
        self._i += 1
        if self._i % self._nwrite == 0:
            self.xdmf.write_function(tr.c, te1)
            # flush like solver.py's u/p writers: without it an aborted run (MPI_ABORT on a NaN) leaves a
            # c.h5 with no superblock -- unreadable (MEASURED 2026-11-16, brem production runs)
            self.xdmf.flush()
            if self.bal:
                self.bal.flush()
        # lagged c for the mixture properties of the NEXT flow step
        if self.c_prob is not None:
            self.c_prob.x.array[:] = tr.c.x.array
        self.us_old.x.array[:] = self.us_new.x.array
        self.p_old.x.array[:] = self.p_new.x.array
        return its

    def _solve(self):
        tr = self.tr
        g = [src[3].value for src in tr._sources]
        tr.dt.value = self.dt
        tr.problem.solve()
        reason = tr.problem.solver.getConvergedReason()
        if reason < 0:
            raise RuntimeError('contrast solve did not converge (KSP %d) at exam t=%g'
                               % (reason, self.texam()))
        tr.c.x.scatter_forward()
        tr.c_nm1.x.array[:] = tr.c_n.x.array
        tr.c_n.x.array[:] = tr.c.x.array
        tr.dt_prev.value = self.dt
        tr.t += self.dt
        for src, v in zip(tr._sources, g):
            src[3].value = v
        return tr.problem.solver.getIterationNumber()

    # ---- checkpoints (by node coordinates) ------------------------------------------
    def _fields(self):
        tr = self.tr
        return {'c': tr.c, 'c_n': tr.c_n, 'c_nm1': tr.c_nm1, 'us_old': self.us_old, 'p_old': self.p_old}

    def write_checkpoint(self, path):
        if self.t0 is None:      # no contrast step yet (c = 0): a restart without c.h5 is the same
            return
        V = self.tr.V
        nloc = V.dofmap.index_map.size_local
        X = V.tabulate_dof_coordinates()[:nloc]
        data = {k: f.x.array[:nloc*f.function_space.dofmap.index_map_bs].reshape(nloc, -1)
                for k, f in self._fields().items()}
        allX = self.comm.gather(X, root=0)
        alld = {k: self.comm.gather(v, root=0) for k, v in data.items()}
        if self.comm.rank == 0:
            with h5py.File(os.path.join(path, 'c.h5'), 'w') as f:
                f.create_dataset('x', data=np.concatenate(allX))
                for k, v in alld.items():
                    f.create_dataset(k, data=np.concatenate(v))
                f.attrs['t0'] = self.t0
                f.attrs['have_old'] = int(self._have_old)
                f.attrs['dt_prev'] = float(self.tr.dt_prev.value)
        self.comm.barrier()

    def _restore(self):
        self._restored = False
        rst = (self.s.options.get('io') or {}).get('restart') or {}
        if not rst.get('path'):
            return
        fn = Path(rst['path'])/'c.h5'
        if not fn.exists():
            return            # flow-only checkpoint: start the injection with c = 0
        with h5py.File(fn, 'r') as f:
            X = f['x'][:]
            vals = {k: f[k][:] for k in self._fields()}
            self.t0 = float(f.attrs['t0'])
            self._have_old = bool(f.attrs['have_old'])
            self.tr.dt_prev.value = float(f.attrs['dt_prev'])
        V = self.tr.V
        xd = V.tabulate_dof_coordinates()
        d, j = cKDTree(X).query(xd)
        if self.comm.allreduce(d.max() if d.size else 0.0, op=MPI.MAX) > 1e-9*np.ptp(X, 0).max():
            raise ValueError('contrast checkpoint geometry does not match the mesh')
        for k, f in self._fields().items():
            f.x.array[:] = vals[k][j].ravel()
            f.x.scatter_forward()
        if self.c_prob is not None:
            self.c_prob.x.array[:] = self.tr.c.x.array
        self._restored = True

    def close(self):
        if self.xdmf is not None:
            self.xdmf.close()
        if self.bal:
            self.bal.close()
