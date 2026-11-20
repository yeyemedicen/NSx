''' Passive scalar (contrast agent) transport on a fluid mesh, given a velocity field.

    dc/dt + u . grad c - div(D grad c) = s          ('advective' form, default)
    dc/dt + div(u c)  - div(D grad c) = s           ('conservative' form)
    dc/dt + u . grad c + (div u) c/2 - div(D grad c) = s   ('skew' form: energy-neutral
        whatever div u is, like NSx's skew-symmetric convection of the flow solve)

P1 elements, theta scheme in time (theta = 0.5: Crank-Nicolson), SUPG stabilisation
(Brooks & Hughes 1982) with the usual tau = ((2/dt)^2 + (2|u|/h)^2 + 9(4D/h^2)^2)^(-1/2).

Boundaries (facet tags):
  open boundaries (`inflow` and `outlets` ids): inflow is imposed WEAKLY and UPWIND,
      - int_G min(u.n, 0) (c - g) w ds,
  so the condition acts only where the flow enters the domain (a closed valve, u ~ 0,
  imports nothing). g = c_in(t) on an `inflow` id, g = 0 on an `outlets` id with
  backflow: zero (blood re-entering through an outlet carries no contrast) or no term
  with backflow: natural. Every other boundary (the wall) is zero-flux.
  sources: volumetric injection (a catheter tip), s = rate(t) phi(x)/int phi, phi a Gaussian
  exp(-|x - x0|^2/r^2) cut at 3r (default) or a ball of radius r; solute added without
  volume (the injection's own flow belongs to the flow solution).
  shock_capturing C (default 0, off): isotropic diffusion C h |R(c_n)|/|grad c_n| at sharp
  fronts (Codina type, lagged one step so the system stays linear); 0 switches it off.

The class owns NO velocity: the caller sets `self.u` (a P1 vector Function on the same
mesh) to the velocity at t^{n+theta} before each `step()`. That keeps it usable both
OFFLINE (nsx.transport.offline: velocity read from a finished run and interpolated in
time) and INSIDE a flow solver (the solver's own u, set every step).

Mass bookkeeping, `balance()`: with M = int c dx, the continuous equations give
    dM/dt = - sum_G int_G c u.n ds + int c div u dx + int s dx        (advective)
    dM/dt = - sum_G int_G c u.n ds + int s dx                        (conservative)
so a velocity that is not divergence-free creates or destroys solute in the advective
form (it keeps c within the injected range), while the conservative form keeps M but lets
c leave that range. `balance()` returns every term, so the effect is measured, not guessed.
'''
import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem import petsc as fem_petsc
from mpi4py import MPI
from petsc4py import PETSc


def _expr(e):
    ''' float, callable(t), or a string in t (numpy available as np) -> callable(t). '''
    if callable(e):
        return e
    if isinstance(e, str):
        code = compile(e, '<transport expression>', 'eval')
        return lambda t: float(eval(code, {'np': np, 't': t}))
    v = float(e)
    return lambda t: v


class ScalarTransport:
    DEFAULTS = {'D': 0.0, 'theta': 0.5, 'supg': True, 'form': 'advective',
                'inflow': [], 'outlets': [], 'backflow': 'natural', 'sources': [],
                'c0': 0.0, 'petsc_options': None, 'quadrature_degree': 2,
                'shock_capturing': 0.0, 'pressure_correction': 0.0}

    def __init__(self, mesh, facet_tags, options):
        o = {**self.DEFAULTS, **(options or {})}
        if o['form'] not in ('advective', 'conservative', 'skew'):
            raise ValueError("transport form must be 'advective', 'conservative' or 'skew'")
        if o['backflow'] not in ('natural', 'zero'):
            raise ValueError("transport backflow must be 'natural' or 'zero'")
        self.o, self.mesh, self.ft = o, mesh, facet_tags
        self.comm = mesh.comm
        gdim = mesh.geometry.dim
        self.V = fem.functionspace(mesh, ('Lagrange', 1))
        self.W = fem.functionspace(mesh, ('Lagrange', 1, (gdim,)))
        self.u = fem.Function(self.W, name='u')           # set by the caller
        # pressure_correction k > 0: advect with the CT field w = u - k grad p (u = the tentative
        # u*, p = the pressure, both set by the caller; see velocity.CTVelocity), P1 + P0 per cell
        self.p = fem.Function(self.V, name='p')           # set by the caller (if k > 0)
        k = float(o['pressure_correction'])
        self.vel = self.u - fem.Constant(mesh, k)*ufl.grad(self.p) if k > 0 else self.u
        self.c = fem.Function(self.V, name='c')
        self.c_n = fem.Function(self.V, name='c_n')
        self.c_nm1 = fem.Function(self.V, name='c_nm1')      # for the lagged residual
        for f in (self.c, self.c_n, self.c_nm1):
            f.x.array[:] = float(o['c0'])
        self.dt_prev = fem.Constant(mesh, 1.0)
        self.dt = fem.Constant(mesh, 1.0)
        self.t = 0.0

        self.ds = ufl.Measure('ds', domain=mesh, subdomain_data=facet_tags,
                              metadata={'quadrature_degree': 2})
        # cap the quadrature: tau holds sqrt and powers, and UFL's degree estimate for
        # them is very high (7 s/step on the coronary mesh without the cap)
        q = o['quadrature_degree']
        md = {'quadrature_degree': int(q)} if q else {}
        self.dx = ufl.Measure('dx', domain=mesh, metadata=md)
        self.n = ufl.FacetNormal(mesh)
        self._inflow = [(int(b['id']), _expr(b.get('c', 1.0)), fem.Constant(mesh, 0.0))
                        for b in o['inflow']]
        self._outlets = [int(i) for i in o['outlets']]
        self._sources = []
        for s in o['sources']:
            x0 = np.asarray(s['center'], float); r = float(s['radius'])
            ind = fem.Function(self.V)
            if s.get('shape', 'gaussian') == 'gaussian':      # smooth: exp(-|x-x0|^2/r^2), cut at 3r
                ind.interpolate(lambda x: np.where(np.sum((x.T - x0)**2, 1) <= 9*r*r,
                                                   np.exp(-np.sum((x.T - x0)**2, 1)/(r*r)), 0.0))
            else:                                              # 'ball': top hat
                ind.interpolate(lambda x: (np.sum((x.T - x0)**2, 1) <= r*r).astype(float))
            vol = self.comm.allreduce(fem.assemble_scalar(fem.form(ind*self.dx)), op=MPI.SUM)
            if vol <= 0:
                raise ValueError('transport source at %s, radius %g, contains no mesh node'
                                 % (x0.tolist(), r))
            # volume: true -> 'rate' is the injectate VOLUME rate q(t) and the source is a mass
            # source of the flow (div u = s_v = q phi / int phi, the flow solver must use the same
            # s_v): the solute source is c_inj s_v and the advective form gets the dilution term
            # -c s_v. volume: false (default) -> 'rate' is the SOLUTE rate (q c_inj), no volume.
            self._sources.append((ind, vol, _expr(s['rate']), fem.Constant(mesh, 0.0),
                                  float(s.get('c', 1.0)), bool(s.get('volume', False))))
        if any(v for *_, v in self._sources) and o['form'] != 'advective':
            raise ValueError('volume sources are implemented for the advective form')
        self._build()

    # ------------------------------------------------------------------
    def _build(self):
        o, u, n, ds, dx, dt = self.o, self.vel, self.n, self.ds, self.dx, self.dt
        th = float(o['theta'])
        D = fem.Constant(self.mesh, float(o['D']))
        c, w = ufl.TrialFunction(self.V), ufl.TestFunction(self.V)
        cn = self.c_n
        ct = th*c + (1 - th)*cn
        un = ufl.dot(u, n)
        neg = 0.5*(un - abs(un))                            # min(u.n, 0)
        zero = fem.Constant(self.mesh, 0.0)
        src = sum(((ci if v else 1.0)*g/vol*ind for ind, vol, _, g, ci, v in self._sources), zero)
        sv = sum((g/vol*ind for ind, vol, _, g, ci, v in self._sources if v), zero)
        self.sv = sv                                        # volume source (UFL), 0 if none

        if o['form'] in ('advective', 'skew'):
            F = ((c - cn)/dt*w + ufl.dot(u, ufl.grad(ct))*w + D*ufl.dot(ufl.grad(ct), ufl.grad(w))
                 + sv*ct*w - src*w)*dx
            if o['form'] == 'skew':
                F += 0.5*ufl.div(u)*ct*w*dx
            # weak upwind inflow: - min(u.n,0) (c - g) w on open boundaries
            for bid, _, g in self._inflow:
                F += -neg*(ct - g)*w*ds(bid)
            if o['backflow'] == 'zero':
                for bid in self._outlets:
                    F += -neg*ct*w*ds(bid)
        else:
            pos = 0.5*(un + abs(un))
            F = ((c - cn)/dt*w - ct*ufl.dot(u, ufl.grad(w)) + D*ufl.dot(ufl.grad(ct), ufl.grad(w))
                 - src*w)*dx
            open_ids = [b for b, _, _ in self._inflow] + self._outlets
            for bid in open_ids:                            # outflow carries c out
                F += pos*ct*w*ds(bid)
            for bid, _, g in self._inflow:                  # inflow carries g in
                F += neg*g*w*ds(bid)
            if o['backflow'] == 'natural':                  # re-entering blood carries c
                for bid in self._outlets:
                    F += neg*ct*w*ds(bid)
            # backflow 'zero': no term, re-entering blood carries no solute

        if o['supg']:
            h = ufl.CellDiameter(self.mesh)
            umag = ufl.sqrt(ufl.dot(u, u) + 1e-30)
            tau = ((2/dt)**2 + (2*umag/h)**2 + 9*(4*D/h**2)**2)**(-0.5)
            R = (c - cn)/dt + ufl.dot(u, ufl.grad(ct)) + sv*ct - src  # P1: no 2nd-derivative term
            if o['form'] == 'conservative':
                R += ct*ufl.div(u)
            elif o['form'] == 'skew':
                R += 0.5*ct*ufl.div(u)
            F += tau*ufl.dot(u, ufl.grad(w))*R*dx

        if o['shock_capturing']:
            # discontinuity capturing (Codina 1993 type), linearised on c_n so the system
            # stays linear: isotropic nu_dc = C h |R(c_n)| / |grad c_n| where the residual of
            # the previous solution is large, i.e. at sharp fronts and around the source;
            # 0 where the solution is smooth. Removes the SUPG over/undershoots.
            h = ufl.CellDiameter(self.mesh)
            # residual of the previous solution, WITH its time derivative: a front that
            # moves correctly has R ~ 0 and gets no extra diffusion
            Rn = (cn - self.c_nm1)/self.dt_prev + ufl.dot(u, ufl.grad(cn)) + sv*cn - src
            gn = ufl.sqrt(ufl.dot(ufl.grad(cn), ufl.grad(cn)) + 1e-12)
            nu = float(o['shock_capturing'])*h*abs(Rn)/gn
            F += nu*ufl.dot(ufl.grad(ct), ufl.grad(w))*dx

        self.a, self.L = ufl.lhs(F), ufl.rhs(F)
        popts = o['petsc_options'] or {'ksp_type': 'gmres', 'pc_type': 'bjacobi',
                                       'sub_pc_type': 'ilu', 'ksp_rtol': 1e-10,
                                       'ksp_atol': 1e-14}
        self.problem = fem_petsc.LinearProblem(self.a, self.L, bcs=[], u=self.c,
                                               petsc_options=popts,
                                               petsc_options_prefix='nsx_transport_')

        # diagnostics (evaluated on the new c)
        cc = self.c
        self._f_mass = fem.form(cc*dx)
        # int c div u, written as int_dOmega c u.n ds - int u.grad c dx: identical for a continuous
        # u, and still the exact creation term of the advective form for a cell-wise field
        # (pressure_correction), whose div has facet jumps
        # minus the dilution int c s_v: c_div_u is the SPURIOUS creation (0 when div u = s_v)
        self._f_divsrc = fem.form(cc*un*ds - ufl.dot(u, ufl.grad(cc))*dx - sv*cc*dx)
        self._f_csv = fem.form(sv*cc*dx)
        self._f_src = fem.form(src*dx)
        # boundary exchange as the scheme sees it: outflow carries c, inflow carries g
        # (inflow ids), 0 (outlets, backflow 'zero') or c (otherwise)
        pos_ = 0.5*(un + abs(un))
        gin = {bid: g for bid, _, g in self._inflow}
        self._f_flux = {}
        # the GLOBAL tag set: a rank's own facets may miss some tags, and every rank must
        # build (and reduce) the same forms
        all_tags = sorted(set().union(*self.comm.allgather(set(np.unique(self.ft.values).tolist()))))
        for t in all_tags:
            t = int(t)
            if t in gin:
                e = pos_*cc + neg*gin[t]
            elif t in self._outlets and o['backflow'] == 'zero':
                e = pos_*cc
            else:
                e = cc*un
            self._f_flux[t] = fem.form(e*ds(t))
        self._f_min = None
        # pressure_correction: the field w is P1 + P0, its facet trace is not the flux the
        # scheme exchanges (at a pressure-driven inlet the inflow lives in the weak divergence,
        # MEASURED angio kim4). Book it NODALLY with d_j = int w.grad(phi_j) dx (testing the
        # advective form with v = 1): outward flux of tag t = sum_{j in t} c_j d_j minus the
        # weak inflow/backflow term of t; creation (reported as c_div_u) = -sum_{j not open}
        # c_j d_j, which is 0 for a CT-divergence-free w (interior and no-slip wall nodes).
        self._nodal = float(o['pressure_correction']) > 0
        if self._nodal:
            if o['form'] != 'advective':
                raise ValueError('pressure_correction is implemented for the advective form')
            fdim = self.mesh.topology.dim - 1
            nloc = self.V.dofmap.index_map.size_local
            taken = np.zeros(nloc, bool)
            self._mask = {}
            for t in [b for b, _, _ in self._inflow] + self._outlets:
                dofs = fem.locate_dofs_topological(self.V, fdim, self.ft.find(t))
                m = np.zeros(nloc, bool); m[dofs[dofs < nloc]] = True
                m &= ~taken; taken |= m
                self._mask[t] = m
            self._rest = ~taken
            self._f_d = fem.form(ufl.dot(u, ufl.grad(ufl.TestFunction(self.V)))*dx)
            self._f_bt = {}
            for t in all_tags:
                t = int(t)
                if t in gin:
                    self._f_bt[t] = fem.form(-neg*(cc - gin[t])*ds(t))
                elif t in self._outlets and o['backflow'] == 'zero':
                    self._f_bt[t] = fem.form(-neg*cc*ds(t))

    # ------------------------------------------------------------------
    def step(self, dt):
        ''' Advance c from t to t + dt with the velocity currently in self.u (t^{n+theta}). '''
        th = float(self.o['theta'])
        self.dt.value = dt
        self.set_boundary_values(self.t + th*dt)
        self.problem.solve()
        reason = self.problem.solver.getConvergedReason()
        if reason < 0:
            raise RuntimeError('transport solve did not converge (KSP reason %d) at t=%g'
                               % (reason, self.t + dt))
        self.c.x.scatter_forward()
        self.c_nm1.x.array[:] = self.c_n.x.array
        self.c_n.x.array[:] = self.c.x.array
        self.dt_prev.value = dt
        self.t += dt
        return self.problem.solver.getIterationNumber()

    def set_boundary_values(self, t):
        ''' Inflow concentrations and source rates at time t (step() uses t^{n+theta}). '''
        for _, fn, g in self._inflow:
            g.value = fn(t)
        for _, _, fn, g, _, _ in self._sources:
            g.value = fn(t)

    def _sum(self, f):
        return self.comm.allreduce(fem.assemble_scalar(f), op=MPI.SUM)

    def balance(self):
        ''' Mass M = int c, the outward solute flux per tag (as the scheme exchanges it),
        int c div u and int s, at the current c and u. Advective form:
        dM/dt = -sum(flux) + c_div_u + source; skew: -sum(flux) + c_div_u/2 + source;
        conservative: dM/dt = -sum(flux) + source. '''
        if self._nodal:
            d = fem_petsc.assemble_vector(self._f_d)
            d.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            nloc = self.V.dofmap.index_map.size_local
            cd = d.array[:nloc]*self.c.x.array[:nloc]
            d.destroy()
            red = lambda x: self.comm.allreduce(float(x), op=MPI.SUM)
            flux = {}
            for t in self._f_flux:
                v = red(cd[self._mask[t]].sum()) if t in self._mask else 0.0
                if t in self._f_bt:
                    v += self._sum(self._f_bt[t])
                flux[t] = v
            return {'M': self._sum(self._f_mass),
                    'c_div_u': -red(cd[self._rest].sum()) - self._sum(self._f_csv),
                    'source': self._sum(self._f_src), 'flux': flux}
        return {'M': self._sum(self._f_mass), 'c_div_u': self._sum(self._f_divsrc),
                'source': self._sum(self._f_src),
                'flux': {t: self._sum(f) for t, f in self._f_flux.items()}}

    def range(self):
        a = self.c.x.array[:self.V.dofmap.index_map.size_local]
        lo = self.comm.allreduce(a.min() if a.size else np.inf, op=MPI.MIN)
        hi = self.comm.allreduce(a.max() if a.size else -np.inf, op=MPI.MAX)
        return lo, hi
