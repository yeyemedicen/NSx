''' The velocity of a finished run, read from its XDMF/HDF5 output and interpolated in time.

    sv = StoredVelocity('results/u.h5', W, replay={'t0': 0.8, 'T': 0.8})
    sv.assign(u, t)        # u: P1 vector Function on W; t: transport time

The run's u.h5 holds Function/u/<time> (one row per mesh node, in the WRITING run's
node order) and Mesh/mesh/geometry. Nodes are matched to the local dofs of W by
COORDINATES once (a KD-tree; refused if any match is farther than 1e-9 of the domain
size), so any partitioning / rank count works.

Time: linear interpolation between the two stored frames that bracket the requested
time -- the default whatever the transport dt is (dt equal to the write interval makes it
exact at the frames). `replay: {t0, T}` maps transport time t to stored time
t0 + (t mod T): a periodic flow is stored for one cycle [t0, t0 + T] and replayed for as
long as the transport needs. Without replay, t must lie within the stored range.
'''
from pathlib import Path

import h5py
import numpy as np
from mpi4py import MPI
from scipy.spatial import cKDTree


class StoredVelocity:
    def __init__(self, h5file, W, name='u', replay=None):
        self.f = h5py.File(Path(h5file), 'r')
        g = self.f['Function'][name]
        keys = sorted(g.keys(), key=lambda k: float(k.replace('_', '.')))
        self.times = np.array([float(k.replace('_', '.')) for k in keys])
        self.data = [g[k] for k in keys]
        X = self.f['Mesh/mesh/geometry'][:]
        xd = W.tabulate_dof_coordinates()[:, :X.shape[1]]
        d, self.j = cKDTree(X).query(xd)
        span = np.ptp(X, 0).max()
        dmax = W.mesh.comm.allreduce(d.max() if d.size else 0.0, op=MPI.MAX)
        if dmax > 1e-9*span:
            raise ValueError('%s: stored geometry does not match the mesh (max distance %.3g)'
                             % (h5file, dmax))
        self.bs = W.dofmap.index_map_bs
        self.replay = replay
        if replay:
            t0, T = float(replay['t0']), float(replay['T'])
            if t0 < self.times[0] - 1e-12 or t0 + T > self.times[-1] + 1e-9:
                raise ValueError('replay window [%g, %g] outside the stored times [%g, %g]'
                                 % (t0, t0 + T, self.times[0], self.times[-1]))
        self._cache = {}

    def stored_time(self, t):
        if self.replay:
            t0, T = float(self.replay['t0']), float(self.replay['T'])
            return t0 + np.mod(t, T)
        if t < self.times[0] - 1e-12 or t > self.times[-1] + 1e-12:
            raise ValueError('t = %g outside the stored times [%g, %g]; use replay'
                             % (t, self.times[0], self.times[-1]))
        return t

    def _frame(self, k):
        if k not in self._cache:
            if len(self._cache) > 3:
                self._cache.pop(next(iter(self._cache)))
            self._cache[k] = self.data[k][:][self.j, :self.bs]
        return self._cache[k]

    def values(self, t):
        ts = self.stored_time(t)
        k = int(np.clip(np.searchsorted(self.times, ts) - 1, 0, len(self.times) - 2))
        t0, t1 = self.times[k], self.times[k + 1]
        a = float(np.clip((ts - t0)/(t1 - t0), 0.0, 1.0))
        if a == 0.0:
            return self._frame(k)
        if a == 1.0:
            return self._frame(k + 1)
        return (1 - a)*self._frame(k) + a*self._frame(k + 1)

    def assign(self, u, t):
        u.x.array[:] = self.values(t).ravel()
        u.x.scatter_forward()


class CTVelocity:
    ''' The weakly divergence-free Chorin-Temam velocity w = u* - (dt/rho) grad p of a finished
    NSx run that saved the UPDATE velocity u (write_velocity: update, fem: DC_in_update: true)
    and p, rebuilt with no flow re-run:

        M_D du = -(dt/rho) (grad p, v)   (P1 consistent mass, du = 0 on the velocity-Dirichlet
                                          facets: NSx solve_velocity_update)
        u* = u - du,     w = u* - (dt/rho) grad p   (P1 + P0 per cell)

    The saved u is the M_D projection of w onto nodal P1; it cannot hold the one-cell-layer
    pressure correction that carries the inflow at a pressure-driven (condensed) inlet, so it
    is not divergence-free there. MEASURED (angio kim4, 2026-11-14): u* rebuilt this way equals
    a run that saved u* to 1e-11.

        cv = CTVelocity('res/u.h5', 'res/p.h5', W, V, ft, dt_flow, rho, dirichlet=[1], replay=..)
        cv.assign(ustar, p, t)    # then the transport advects with ustar - k grad p, k = cv.k
    u and p are interpolated linearly in time (the map to u* is linear, so this is consistent).
    '''
    def __init__(self, ufile, pfile, W, V, facet_tags, dt, rho, dirichlet=(1,), replay=None,
                 name_u='u', name_p='p'):
        import ufl
        from dolfinx import fem
        from dolfinx.fem import petsc as fem_petsc
        from petsc4py import PETSc
        self.su = StoredVelocity(ufile, W, name=name_u, replay=replay)
        self.sp = StoredVelocity(pfile, V, name=name_p, replay=replay)
        self.k = float(dt)/float(rho)
        mesh = W.mesh
        self.W = W
        self.pf = fem.Function(V)
        u, v = ufl.TrialFunction(W), ufl.TestFunction(W)
        fdim = mesh.topology.dim - 1
        facets = np.concatenate([facet_tags.find(int(t)) for t in dirichlet])
        dofs = fem.locate_dofs_topological(W, fdim, facets)
        zero = fem.Function(W)
        self.bcs = [fem.dirichletbc(zero, dofs)]
        self.M = fem_petsc.assemble_matrix(fem.form(ufl.inner(u, v)*ufl.dx), bcs=self.bcs)
        self.M.assemble()
        self.L = fem.form(-self.k*ufl.inner(ufl.grad(self.pf), v)*ufl.dx)
        self.du = fem.Function(W)
        self.b = self.du.x.petsc_vec.duplicate()
        ksp = PETSc.KSP().create(mesh.comm)
        ksp.setOperators(self.M)
        ksp.setType('cg'); ksp.getPC().setType('jacobi')
        ksp.setTolerances(rtol=1e-13, atol=1e-30, max_it=1000)
        self.ksp = ksp
        self._fem_petsc = fem_petsc

    def assign(self, ustar, p, t):
        from petsc4py import PETSc
        self.sp.assign(self.pf, t)
        with self.b.localForm() as bl:
            bl.set(0.0)
        self._fem_petsc.assemble_vector(self.b, self.L)
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        self._fem_petsc.set_bc(self.b, self.bcs, alpha=0.0)
        self.ksp.solve(self.b, self.du.x.petsc_vec)
        if self.ksp.getConvergedReason() < 0:
            raise RuntimeError('CTVelocity mass solve failed (reason %d)' % self.ksp.getConvergedReason())
        self.du.x.scatter_forward()
        self.su.assign(ustar, t)
        ustar.x.array[:] -= self.du.x.array
        ustar.x.scatter_forward()
        p.x.array[:] = self.pf.x.array
        p.x.scatter_forward()
