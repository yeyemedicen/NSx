''' Fractional-Step Navier-Stokes problem module

Author: Jeremias Garay
Date: 
'''

from .streamline_diffusion import SDParameter
from ..logger.logger import LoggerBase
from pathlib import Path
from scipy.interpolate import interp1d
import csv
from numpy import load
from mpi4py import MPI
import dolfinx
import dolfinx.fem as fem
from dolfinx.fem import (
    Function, functionspace, dirichletbc,
    locate_dofs_topological, locate_dofs_geometrical,
    form as fem_form, assemble_scalar,
)
import basix
import basix.ufl as bufl
import ufl
from ufl import (
    TrialFunction, TestFunction,
    as_vector, inner, sym, grad, dx, dot, div,
    FacetNormal, Measure, Identity, inv, det, sqrt, CellVolume, CellDiameter,
)
import numpy as np
from petsc4py import PETSc
from common import inout, utils


def _compute_inlet_local_frame(inlet_coords):
    """Compute an orthonormal local frame for an inlet face via SVD.

    Works for any inlet orientation in 2D or 3D.  For 2D meshes DOLFINx
    stores z=0, so the third singular value is ~0 — callers should guard
    against R2 < eps before using t2.

    Parameters
    ----------
    inlet_coords : (N, 3) array of inlet DOF positions.

    Returns
    -------
    centroid : (3,) centroid of the inlet point cloud
    t1       : (3,) principal tangent (max variance)
    t2       : (3,) secondary tangent (mid variance; ~zero for 2-D meshes)
    n_hat    : (3,) unit normal (min variance; arbitrary sign — caller orients)
    R1       : half-extent along t1
    R2       : half-extent along t2 (≈ 0 for 2-D meshes)
    """
    centroid = inlet_coords.mean(axis=0)
    centered = inlet_coords - centroid
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    t1, t2, n_hat = Vt[0], Vt[1], Vt[2]

    # 2-D meshes: DOLFINx stores z=0 for all DOFs, so the SVD sees two
    # near-zero singular values (x and z both have zero variance) and picks
    # z=[0,0,1] as n_hat instead of the actual in-plane normal.
    # Fix: if z-variance is negligible, compute n_hat as the in-plane
    # perpendicular to t1 (rotate 90° in the x-y plane).
    z_std = np.std(centered[:, 2])
    xy_rms = np.sqrt(np.mean(centered[:, :2] ** 2)) + 1e-30
    if z_std < 1e-8 * xy_rms:
        n_hat = np.array([-t1[1], t1[0], 0.0])
        n_hat /= np.linalg.norm(n_hat)

    s1 = centered @ t1
    s2 = centered @ t2
    R1 = (s1.max() - s1.min()) / 2.0
    R2 = (s2.max() - s2.min()) / 2.0
    return centroid, t1, t2, n_hat, R1, R2


def _dof_count(V):
    ''' Global DOF count for a FunctionSpace. '''
    return V.dofmap.index_map.size_global * V.dofmap.index_map_bs


def _project(expr, V):
    ''' L2-project a UFL expression into FunctionSpace V. '''
    from dolfinx.fem.petsc import LinearProblem
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    problem = LinearProblem(
        fem_form(inner(u, v) * dx),
        fem_form(inner(expr, v) * dx),
        bcs=[])
    return problem.solve()


_TIME_EXPR_NS = {
    'sin': np.sin,  'cos': np.cos,  'tan': np.tan,
    'exp': np.exp,  'log': np.log,  'sqrt': np.sqrt,
    'abs': np.abs,  'tanh': np.tanh, 'pi': np.pi,
    'min': min,     'max': max,
}


def _make_time_expression(expr_str, params):
    ''' Compile a scalar waveform string in `t` into a float-valued
    callable. `params` supplies named constants; the usual math functions
    and `pi` are in scope. Booleans evaluate numerically, so
    '(t<Th)' works as a gate.

    Args:
        expr_str (str):  e.g. 'P*sin(pi*t/Th)*(t<Th)'
        params (dict):   named constants, e.g. {'P': 1e3, 'Th': 0.3}

    Returns:
        callable: f(t) -> float
    '''
    if 't' in params:
        raise KeyError("'t' is reserved for simulation time and cannot be "
                       "used as a waveform parameter")

    def _f(t, _e=expr_str, _p=dict(params)):
        ns = {'t': float(t)}
        ns.update(_p)
        ns.update(_TIME_EXPR_NS)
        return float(eval(_e, {'__builtins__': {}}, ns))

    return _f


def rank0(func):
    ''' Rank 0 decorator: decorated function "does nothing" if rank > 0 '''
    def inner(*args, **kwargs):
        if MPI.COMM_WORLD.rank == 0:
            func(*args, **kwargs)
    return inner


class Problem(LoggerBase):
    ''' NavierStokes Problem base class '''

    # =========================================================================
    # Initialization
    # =========================================================================

    def __init__(self, inputfile=None):
        ''' Initialize FractionalStep problem.

        Args:
            inputfile (str):     path to YAML input file
        '''
        super().__init__()
        self.check_version()

        self.options = None
        self.inputfile = inputfile
        if inputfile:
            self.get_parameters(inputfile)

        self._logging_filehandler = None
        self.setup_logger()

        module = self.__module__.split('.')[0]
        #self.logger.info('{} {}'.format(module,
        #                                utils.get_git_rev_hash(__file__)))

        self.logger.info('Initializing')
        self.logger.info('Number of parallel tasks: {}'.format(
            MPI.COMM_WORLD.size))
        self.logger.info('Write out path: {}'.format(
            self.options['io']['write_path']))

        # mesh, boundaries, subdomains
        self.mesh = None
        self.bnds = None
        self.ds = None
        self.mu = None
        self.rho = None
        self.k = None

    def init(self):
        ''' Initialize problem, performing the actions:

        * set constants
        * read mesh
        * create ALE operators
        * create function spaces
        * process boundary conditions
        * create variational form
        * read checkpoints
        '''
        if not self.options:
            raise Exception('Options not set. call get_parameters(optfile) '
                            'first!')
        self.init_mesh()
        self.set_constants()
        self.init_ale_operators()
        self.create_functionspaces()
        self.boundary_conditions()
        self.variational_form()

    def check_version(self):
        ''' Check if compatible dolfinx version is installed '''
        from packaging.version import Version as _V
        if _V(dolfinx.__version__) < _V('0.7'):
            raise Exception('DOLFINx version 0.7 or higher required!')

    @staticmethod
    def default_ale():
        ''' Implements default ALE dictionary '''
        ale_dict = {
            'type': 'default',
            'io': {
                'read_checkpoints': False,
                'fem_type': 'p1'
            },
            'timemarching': None,
            'fem': {
                'displacement_space': 'p1'
            },
            'lifting': {
                'type': None
            },
            'deformations': []
        }
        return ale_dict

    def get_parameters(self, inputfile):
        ''' Read parameters from YAML input file into options dictionary

        Args:
            inputfile (str):     path to YAML file
        '''
        self.options = inout.read_parameters(inputfile)
        self.ale = Problem.default_ale()
        if 'ale' in self.options.keys():
            def _deep_merge_ale(base, override):
                result = dict(base)
                for k, v in override.items():
                    if k in result and hasattr(result[k], 'items') and hasattr(v, 'items'):
                        result[k] = _deep_merge_ale(result[k], v)
                    else:
                        result[k] = v
                return result
            self.ale = _deep_merge_ale(self.ale, self.options['ale'])
            self._using_ale = True
        else:
            self._using_ale = False
     
    def setup_logger(self):
        ''' Create logging File Handler '''
        MPI.COMM_WORLD.Barrier()
        path = Path(self.options['io']['write_path']).joinpath('nsx_solver.log')
        if MPI.COMM_WORLD.rank == 0:
            utils.trymkdir(str(path.parent))
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        MPI.COMM_WORLD.Barrier()
        self.set_log_filehandler(str(path))

    def _C(self, val):
        ''' Shorthand: fem.Constant(self.mesh, val) '''
        return fem.Constant(self.mesh, np.array(val, dtype=PETSc.ScalarType))

    # =========================================================================
    # Mesh, function spaces, and operators
    # =========================================================================

    def set_constants(self):
        ''' Set Constants from options file as instance attributes '''
        if 'fluid' not in self.options:
            raise DeprecationWarning('Fluid properties should be defined in '
                                     'the input file as items of a \'fluid\' '
                                     'section, e.g.,\n fluid:\n\tdensity: 1.\n'
                                     '\tdynamic_viscosity: 0.001')

            self.mu = self._C(self.options['dynamic_viscosity'])
            self.rho = self._C(self.options['density'])

        else:

            self.mu = self._C(self.options['fluid']['dynamic_viscosity'])
            self.rho = self._C(self.options['fluid']['density'])

        dt = self.options['timemarching']['dt']
        self.k = self._C(1./dt)

    def init_mesh(self):
        ''' Read in mesh, subdomains and boundary information. '''
        self.logger.info('Reading mesh {}'.format(self.options['mesh']))
        self.mesh, self.subdomains, self.facet_tags = \
            inout.read_mesh(self.options['mesh'])
        self.bnds = self.facet_tags   # preserve legacy name used downstream
        self.ndim = self.mesh.topology.dim
        
        if self.ale['type'] == 'external':
            self.mesh_ext, _, _ = \
                inout.read_mesh(self.ale['io']['mesh_path'])

    def init_ale_operators(self):
        ''' Define common operators used in ALE framework. '''
        if not self.ale['type'] in ('default', 'manual', 'external'):
            raise Exception('Only available: \'default\', \'manual\''
                            'and \'external\' options.')
        
        if self._using_ale:
            self.F_ = lambda f: Identity(self.ndim) + grad(f)
        else:
            self.F_ = lambda f: Identity(self.ndim)
        
        self.J_ = lambda g: det(self.F_(g))

    def create_functionspaces(self):
        r''' Create function spaces for velocity and pressure, namely:

        * D:   VectorFunctionSpace for the displacement
        * Di:  FunctionSpace for a 'generic' displacement component
        * V:   VectorFunctionSpace for the velocity
        * Vi:  FunctionSpace for a 'generic' velocity component
        * Q:   FunctionSpace for the pressure

        and initialize functions :math: `d \in D`, :math:`u \in V`,
        :code:`u_lst = [u_i \in V_i for all i]`, :math:`p \in Q`.

        The elements are specified via options :code:`ale: displacement_space`,
        :code:`fem: velocity_space` and :code:`pressure_space`, 
        where possible options are:

        * displacement: p1, p2
        * velocity:     p1, p2, p1b/p1+ (bubble enriched)
        * pressure:     p1, p0/dg0, p1-/dg1
        '''
        if 'elements' in self.options['fem']:
            raise Exception('Testing new interface: \'velocity_space\', '
                            '\'pressure_space\'')
        
        d_space = self.ale['fem']['displacement_space'].lower()
        u_space = self.options['fem']['velocity_space'].lower()
        p_space = self.options['fem']['pressure_space'].lower()

        cell = self.mesh.basix_cell()

        if self._using_ale:
            if hasattr(self, 'mesh_ext'):
                s_space = self.ale['io']['fem_type'].lower()
                self.logger.info('Creating external displacement space: {}'.
                    format(s_space.capitalize()))
                deg = int(s_space[1])
                self.S = functionspace(self.mesh_ext,
                                       ("Lagrange", deg, (self.ndim,)))
                self.d_s = Function(self.S)
                self.v_s = Function(self.S)

            self.logger.info('Creating displacement space: {}'.format(
                d_space.capitalize()))
            if d_space in ('p1', 'p2'):
                deg = int(d_space[1])
                self.D  = functionspace(self.mesh, ("Lagrange", deg, (self.ndim,)))
                self.Di = functionspace(self.mesh, ("Lagrange", deg))
                self.DG = functionspace(self.mesh, ("DG", 0))

                self.logger.info('Number of displacement (per comp.) DOFs: {}'.format(
                    _dof_count(self.Di)))

                self.d = Function(self.D, name='d')
                self.d_lst = [Function(self.Di, name='d{}'.format(i))
                                for i in range(self.ndim)]
                self.d0 = Function(self.D, name='d0')
                self.d0_lst = [Function(self.Di, name='d{}'.format(i))
                                for i in range(self.ndim)]

                self.F, self.J = self.F_(self.d), self.J_(self.d)
                self.F0, self.J0 = self.F_(self.d0), self.J_(self.d0)
        else:
            self.F, self.J = Identity(self.ndim), self._C(1.)
            self.F0, self.J0 = Identity(self.ndim), self._C(1.)

        self.logger.info('Creating velocity space: {}'.format(
            u_space.capitalize()))
        if u_space in ('p1', 'p2'):
            deg = int(u_space[1])
            self.V  = functionspace(self.mesh, ("Lagrange", deg, (self.ndim,)))
            self.Vi = functionspace(self.mesh, ("Lagrange", deg))
        elif u_space in ('p1b', 'p1+'):
            deg = int(u_space[1])
            P1_s = bufl.element("Lagrange", cell, deg)
            B_s  = bufl.element("Bubble",   cell, deg + self.ndim)
            P1_v = bufl.element("Lagrange", cell, deg, shape=(self.ndim,))
            B_v  = bufl.element("Bubble",   cell, deg + self.ndim, shape=(self.ndim,))
            self.Vi = functionspace(self.mesh, bufl.enriched_element([P1_s, B_s]))
            self.V  = functionspace(self.mesh, bufl.enriched_element([P1_v, B_v]))

        self.logger.info('Creating pressure space: {}'.format(
            p_space.upper()))
        if p_space == 'p1':
            self.Q = functionspace(self.mesh, ("Lagrange", 1))
        elif p_space in ('p0', 'dg0'):
            self.Q = functionspace(self.mesh, ("DG", 0))
        elif p_space in ('p1-', 'dg1'):
            self.Q = functionspace(self.mesh, ("DG", 1))

        self.logger.info('Number of velocity (per component) DOFs: {}'.format(
            _dof_count(self.Vi)))
        self.logger.info('Number of pressure DOFs:                 {}'.format(
            _dof_count(self.Q)))

        self.u = Function(self.V, name='u')
        self.u_lst = [Function(self.Vi, name='u{}'.format(i))
                      for i in range(self.ndim)]
        self.upd = Function(self.V, name='u')
        self.upd_lst = [Function(self.Vi, name='u{}'.format(i))
                        for i in range(self.ndim)]

        self.p = Function(self.Q, name='p')
        
        self.u0 = Function(self.V, name='u0')
        self.u0_lst = [Function(self.Vi, name='u{}'.format(i))
                        for i in range(self.ndim)]
        self.u0_mapdd_lst = [Function(self.Vi, name='u{}'.format(i))
                        for i in range(self.ndim)]

    # =========================================================================
    # Variational forms
    # =========================================================================

    def variational_form(self):
        ''' Set up variational forms of the problem and save in dictionary for
        later use (reassembly of parts).

        Separate forms to be stored:
        Displacement space:
            * difussion
            * div form div(d)*div(e)

        Velocity (component-wise) space:
            * du/dt  "mass system" contribution
            * diffusion
            * convection (incl. backflow stabilization + supg)

        Velocity vector space:
            * divergence for projection step

        Pressure space:
            * Laplacian
            * pressure derivatives dp/dx_i
        '''
        self.forms = {'u': {}, 'p': {}}
        if self._using_ale:
            self.forms.update({'d': {}})
            self.form_lifting()

        self.form_velocity_tentative()
        self.form_pressure()
        self.form_velocity_update()

    def form_lifting(self):
        ''' Forms definition for lifting step.

            Depending on the user choice, the lifting operator can be chosen as:
            * \'harmonic\'        : solving a laplacian problem with bdry data.
            * \'elastic\'         : solving a pseudo elastic problem with bdry data
            *                       based on FSI - Richter book, page 101.
            * \'elastic_element\' : solving a pseudo elastic problem with bdry data
            *                       based on Landajuela's PhD. Thesis, page 114.
        '''
        d = TrialFunction(self.D)
        e = TestFunction(self.D)
        lift_dict = self.ale['lifting']
        # XXX: use already defined deformations 
        # here we assume all of them are the same!
        def_dict = self.ale['deformations']
        #d_vec = as_vector(self.d_lst)
        a_rhs = inner(self._C(self.ndim*[0.]), e)*dx

        if lift_dict['type'] == None:
            raise Exception("Incompatible lifting configuration.")
        elif lift_dict['type'] == 'harmonic':
            a_diff = inner(grad(d), grad(e))*dx
            a_div = self._C(0.)*inner(d, e)*dx
        elif lift_dict['type'] in ('elastic', 'elastic_element'):
            lambda_, mu_ = self.params_lifting()
            a_diff = lambda_*inner(sym(grad(d)), sym(grad(e)))*dx
            a_div = mu_*div(d)*div(e)*dx
        else:
            raise Exception("Lifting type: {} not implemented.".format(
                                lift_dict['type']))

        self.forms['d'].update({
            'diff': a_diff,
            'div': a_div,
            'rhs_const': a_rhs
        })

    def params_lifting(self):
        ''' Handles input parameters for lifting operators. '''
        lifting = self.ale['lifting']
        params = lifting['parameters']
        if lifting['type'] == 'elastic':
            lambda_ = self._C(1.)
            mu_ = self._C(params['mu'])
        elif lifting['type'] == 'elastic_element':
            vol = CellVolume(self.mesh)
            lambda_ = self._C(1.)/vol
            mu_ = self._C(params['mu'])/vol
        return lambda_, mu_

    def form_velocity_tentative(self):
        ''' Definition of forms of tentative velocity step. '''
        rho = self.rho
        mu = self.mu
        k = self.k
        F, J = self.F, self.J
        F0, J0 = self.F0, self.J0

        if self._using_ale:
            d, d0 = self.d, self.d0
            w_conv = k*(d - d0)
        else:
            w_conv = Function(self.V)

        ui = TrialFunction(self.Vi)
        vi = TestFunction(self.Vi)
        p = TrialFunction(self.Q)

        def diff(u):
            # inner(2*mu*sym(grad(u)), sym(grad(vi)))*dx
            if self.options['fem']['strain_symmetric']:
                return J*inner(2*mu*sym(dot(grad(u), inv(F))),
                                    sym(dot(grad(vi), inv(F))))*dx
            else:
                # inner(mu*grad(u), grad(vi))*dx
                return J*inner(mu*dot(grad(u), inv(F)),
                            dot(grad(vi), inv(F)))*dx

        def conv(u, u_conv):
            if self.options['fluid'].get('stokes', False):
                a_conv = self._C(0)*u*vi*dx
                return a_conv
            # rho*dot(u_conv, grad(u))*vi*dx
            a_conv = (rho*J*dot(dot(grad(u), inv(F)), u_conv - w_conv)*vi*dx
                - 0.5*rho*div(J*dot(inv(F), w_conv))*u*vi*dx
                + 0.5*rho*k*(J - J0)*u*vi*dx)
            if self.options['fem']['convection_skew_symmetric']:
                # 0.5*rho*div(u_conv)*u*vi*dx
                a_conv += 0.5*rho*div(J*dot(inv(F), u_conv))*u*vi*dx

            return a_conv

        if self.options['timemarching']['fractionalstep']['scheme'] == 'IPCS':
            # BDF2 coefficients
            u_vec = as_vector(self.u_lst)
            u0_vec = as_vector(self.u0_lst)
            u_conv = 2.0*u_vec - u0_vec
            self.u_conv_assigned = Function(self.V)
        else:
            if self._using_ale:
                u_conv = as_vector(self.upd_lst)
                self.u_conv_assigned = self.upd
            else:
                u_conv = as_vector(self.u_lst)
                self.u_conv_assigned = self.u

        a_mass = rho*J0*k*dot(ui, vi)*dx
        a_diff = diff(ui)
        a_conv = conv(ui, u_conv)

        # FSI Robin-Neumann interface: boundary impedance term on the LHS
        # (component-independent, so it joins the shared a_conv form; the
        # data-carrying RHS counterpart is per-component, registered below
        # and assembled fresh each solve in build_rhs_tentative_velocity).
        # See BoundaryConditions._fsi_robin_velocity for the construction.
        if self.bc_dict['u'].get('fsi_robin'):
            a_conv += self.bc_dict['u']['fsi_robin']['lhs']
        if self.bc_dict['u'].get('fsi_nitsche'):
            a_conv += self.bc_dict['u']['fsi_nitsche']['lhs']

        # [-p*vi.dx(i)*dx for i in range(self.ndim)]
        a_pres_lst = [-p*J0*dot(grad(vi), inv(F0))[i]*dx for i in range(self.ndim)]

        self.forms['u'].update({'mass': a_mass,
                                'diff': a_diff,
                                'conv': a_conv})

        if self.options['timemarching']['fractionalstep']['scheme'] == 'CT':
            self.forms['u'].update({'pres': None})
        else:
            self.forms['u'].update({'pres': a_pres_lst})

        if self._using_mapdd:
            self.forms['u'].update({'pres': a_pres_lst})


        self.forms['u'].update({
            'neumann': {i: sum(self.bc_dict['u']['neumann'][i]) for i in
                        range(self.ndim)},
            'inflow': {i: sum(self.bc_dict['u']['inflow'][i]) for i in
                        range(self.ndim)},
            'mapdd': {i: sum(self.bc_dict['u']['mapdd'][i]) for i in
                        range(self.ndim)},
            'inflow_lhs': None,
            'navierslip': self.bc_dict['u']['navierslip'],
            'transpiration': self.bc_dict['u']['transpiration'],
            'fsi_robin_rhs': (self.bc_dict['u']['fsi_robin']['rhs']
                              if self.bc_dict['u'].get('fsi_robin')
                              else None),
            'fsi_nitsche_rhs': (self.bc_dict['u']['fsi_nitsche']['rhs']
                                if self.bc_dict['u'].get('fsi_nitsche')
                                else None),
        })

        if 'inflow_lhs' in self.bc_dict['u']:
            self.forms['u']['inflow_lhs'] = self.bc_dict['u']['inflow_lhs']
        
        if 'mapdd_lhs' in self.bc_dict['u']:
            self.forms['u']['conv'] += self.bc_dict['u']['mapdd_lhs'][0]

        forms_stab = self.stabilization(u_conv)

        # TODO maybe do this in self.stabilization()?
        if 'bfs' in forms_stab:
            self.forms['u']['conv'] += forms_stab['bfs']
        if 'fnv' in forms_stab:
            self.forms['u'].update({'fnv': forms_stab['fnv'], 'fnv_type': forms_stab['fnv_type']})
        if 'supg_convdiff' in forms_stab:
            self.forms['u']['conv'] += forms_stab['supg_convdiff']
        if 'supg_time' in forms_stab:
            self.forms['u'].update({'supg_time': forms_stab['supg_time']})
        # if 'supg_gradp' in forms_stab:
        #     self.forms['u'].update({'supg_gradp': forms_stab['supg_gradp']})

    def form_pressure(self):
        ''' Definition of forms of pressure projection step. '''
        k = self.k
        rho = self.rho

        p = TrialFunction(self.Q)
        q = TestFunction(self.Q)
        u = TrialFunction(self.V)

        F, J = self.F, self.J

        # a_lap = inner(grad(p), grad(q))*dx
        # a_divu = -k*rho*div(u)*q*dx
        a_lap = J*inner(dot(grad(p), inv(F)),
                    dot(grad(q), inv(F)))*dx
        a_divu = -k*rho*div(J*dot(inv(F), u))*q*dx

        self.forms['p'].update({
            'laplacian': a_lap,
            'rhs_u': a_divu + sum(self.bc_dict['p']['robin']['forms_u']),
            'neumann': sum(self.bc_dict['p']['neumann']),
            'robin': self.bc_dict['p']['robin']['forms_p'],
            'transpiration_dirichlet_u': self.bc_dict['p']['transpiration']
            ['dirichlet_forms_u'],
            'transpiration_dirichlet_p': self.bc_dict['p']['transpiration']
            ['dirichlet_forms_p'],
        })

        # FSI semi-implicit projection coupling (Fernandez-Gerbeau-Grandmont;
        # driven by JellyFSI, see jellyfsi/solver.py::_timestep_semi_implicit).
        # The projection sub-step imposes the SOLID's interface normal
        # velocity on the end-of-step velocity:
        #     rho*k*(u^{n+1} - u_tent) + grad(phi) = 0,   div(u^{n+1}) = 0,
        #     u^{n+1}.n = v_si.n   on ds(id)
        # which adds Neumann data  dphi/dn = rho*k*(u_tent - v_si).n  to the
        # pressure Poisson, i.e. the RHS boundary term
        #     += rho*k*(u_tent - v_si).n_def q ds(id),   n_def = J F^{-T} N.
        # v_si (fem.Function on V) is updated by the coupler every FSI
        # sub-iteration; u_tent is self.u (holds the tentative velocity at
        # pressure-solve time). Assembled fresh each solve in
        # Solver.build_rhs_pressure().
        # YAML:  fem: fsi_semi_implicit: {enabled: true, id: 5}
        si = self.options['fem'].get('fsi_semi_implicit')
        if si and si.get('enabled'):
            n_si = FacetNormal(self.mesh)
            nans_si = J*inv(F).T*n_si
            self.v_si = Function(self.V, name='fsi_si_velocity')
            self.forms['p']['fsi_si_rhs'] = (
                k*rho*dot(self.u - self.v_si, nans_si)*q*self.ds(si['id']))


        if self._using_mapdd:
            ds = self.ds
            #n = FacetNormal(self.mesh)
            #N = J*inv(F).T*n
            #t_p = dot(grad(p), inv(F)) - dot(dot(grad(p), inv(F)),N)*N
            #t_q = dot(grad(q), inv(F)) - dot(dot(grad(q), inv(F)),N)*N
            
            for bid, prm in self.bc_dict['p']['mapdd']['params'].items():
                #const = self._C(prm['eps_gradp'])
                #self.forms['p']['laplacian'] += const*inner(t_p, t_q)*ds(bid)
                self.forms['p']['laplacian'] += (1/prm['l'])*p*q*ds(bid)
                

        if self._using_wk:
            ds = self.ds
            n = FacetNormal(self.mesh)
            N = J*inv(F).T*n
            t_p = dot(grad(p), inv(F)) - dot(dot(grad(p), inv(F)),N)*N
            t_q = dot(grad(q), inv(F)) - dot(dot(grad(q), inv(F)),N)*N

            # PSPG-like stabilization
            for bid, prm in self.bc_dict['p']['windkessel']['params'].items():
                const = k*rho*prm['eps']
                self.forms['p']['laplacian'] += const*inner(t_p, t_q)*ds(bid)

            # Tangential pressure-gradient penalty g*(grad_t p, grad_t q)*ds:
            # drives p -> const across the outlet face -> promotes normal (1D)
            # flow, damping diastolic recirculation (a soft Windkessel
            # condensation). Enable via windkessel: tangent_penalty:
            # {enabled: true, gamma: G} (high G = stronger toward uniform p).
            _tp = self.options.get('windkessel', {}).get('tangent_penalty', {})
            if _tp.get('enabled', False):
                _g = float(_tp.get('gamma', 0.0))
                for bid, prm in self.bc_dict['p']['windkessel']['params'].items():
                    self.forms['p']['laplacian'] += _g*inner(t_p, t_q)*ds(bid)

            if self.wk['implicit']:
                sqrt_fac = sqrt(k*rho) # + 1e-14 
                self.forms['p'].update({
                    'windkessel_lhs': {},
                    'windkessel_rhs': 0
                    })

                for bid, prm in self.bc_dict['p']['windkessel']['params'].items():
                    area = prm['area']
                    self.forms['p']['windkessel_lhs'][bid] = (1/area)*sqrt_fac*p*ds(bid)
                    self.forms['p']['windkessel_rhs'] += (1/area)*k*rho*prm['rhs_p']*q*ds(bid)

    def form_velocity_update(self):
        ''' Definition of forms of velocity update. '''        
        p = TrialFunction(self.Q)
        vi = TestFunction(self.Vi)

        F, J = self.F, self.J
        # [-p.dx(i)*vi*dx for i in range(self.ndim)]
        a_gradp_lst = [-J*dot(inv(F), grad(p))[i]*vi*dx for i in range(self.ndim)]

        self.forms['u'].update({'gradp': a_gradp_lst})

    # =========================================================================
    # Stabilization
    # =========================================================================

    def stabilization(self, u_conv=None):
        ''' Call stabilization methods as specified in options dict.

        Args:
            u_conv (Function): convecting velocity

        Returns:
            forms_stab (dict):  dictionary with keys 'bfs', 'supg_*',
              containing the weak forms of the corresponding stabilization
              terms
        '''
        forms_stab = {}
        opt = self.options['fem']['stabilization']
        if opt['backflow_boundaries']:
            forms_stab.update(self.stab_backflow(u_conv))
        # XXX warning for changed option format
        if 'supg' in self.options['fem']['stabilization']:
            raise Exception('fem>stabilization>supg key was replaced by '
                            'fem>stabilization>streamline_diffusion!\n'
                            'Possible values for parameter are: \n'
                            '\t - standard/default (length_scale: metric, '
                            'average)\n'
                            '\t - shakib (length_scale: metric, max, '
                            'average) \n'
                            '\t - klr\n'
                            '\t - codina'
                            )
        if opt['streamline_diffusion']['enabled']:
            forms_stab.update(self.streamline_diffusion(u_conv))
        if 'forced_normal' in opt and opt['forced_normal']['enabled']:
            forms_stab.update(self.forced_normal(u_conv))

        return forms_stab

    def forced_normal(self, uprev):
        ''' Forms of forced normal velocities using a semi-implicit implementation as:
                
                Ai = gamma*(ui - {sum_[i!=j](u0j*nj) + ui*ni }*ni )*vi

            In this way, a coupling between the components is avoided since in the dot() product between u and n,
            only the "solving" i-component is used implicitly, while the rest are taken from the velocity of the
            previous time-step u0.

        Args:
            uprev:      Velocity of the previous time-step used in the dot product

        Returns:
            forms dict: forced normal form
        '''

        term_dict = self.options['fem']['stabilization']['forced_normal']
        gamma = self._C(term_dict['gamma'])
        bind_lst = (term_dict['boundaries'])

        self.logger.info('Adding normal forced velocity at '
                         'boundary id {}'.format(bind_lst))

        ui = TrialFunction(self.Vi)
        vi = TestFunction(self.Vi)

        F, J = self.F, self.J
        n = FacetNormal(self.mesh)
        # Using the reference system ds = dS J |inv(F).T N|
        elem = sqrt(dot(J*inv(F).T*n, J*inv(F).T*n))

        ds = ufl.Measure('ds', domain=self.mesh, subdomain_data=self.facet_tags)


        if self._using_mapdd:
            u0 = as_vector(self.u0_mapdd_lst)
        else:
            u0 = uprev
        
        a_fnv = {}


        if term_dict['type'] == 'semi-implicit':
            # semi-implicit
            a_lhs = 0
            a_lhs_semi_1 = 0
            a_lhs_semi_2 = 0
            a_lhs_semi_3 = 0
            a_rhs_1 = 0
            a_rhs_2 = 0
            a_rhs_3 = 0

            for bid in bind_lst:
                a_lhs += gamma*ui*vi*elem*ds(bid)
                a_lhs_semi_1 += gamma*ui*n[0]*n[0]*vi*elem*ds(bid)
                a_lhs_semi_2 += gamma*ui*n[1]*n[1]*vi*elem*ds(bid)
                a_lhs_semi_3 += gamma*ui*n[2]*n[2]*vi*elem*ds(bid)
                a_rhs_1 += gamma*(u0[1]*n[1] + u0[2]*n[2])*n[0]*vi*elem*ds(bid)
                a_rhs_2 += gamma*(u0[0]*n[0] + u0[2]*n[2])*n[1]*vi*elem*ds(bid)
                a_rhs_3 += gamma*(u0[0]*n[0] + u0[1]*n[1])*n[2]*vi*elem*ds(bid)
                
            a_fnv = {
                0: [a_lhs - a_lhs_semi_1 , a_rhs_1],
                1: [a_lhs - a_lhs_semi_2 , a_rhs_2],    
                2: [a_lhs - a_lhs_semi_3 , a_rhs_3],
            }

        elif term_dict['type'] == 'explicit':
            a_lhs = 0
            a_rhs_1 = 0
            a_rhs_2 = 0
            a_rhs_3 = 0
            
            for bid in bind_lst:
                a_lhs += gamma*ui*vi*elem*ds(bid)
                a_rhs_1 += gamma*dot(u0,n)*n[0]*vi*elem*ds(bid)
                a_rhs_2 += gamma*dot(u0,n)*n[1]*vi*elem*ds(bid)
                a_rhs_3 += gamma*dot(u0,n)*n[2]*vi*elem*ds(bid)
            
            a_fnv = {
                0: [a_lhs , a_rhs_1],
                1: [a_lhs , a_rhs_2],    
                2: [a_lhs , a_rhs_3],
            }
        
        else:
            raise Exception('tangential penalization type not recongnized!')
                


        return {'fnv': a_fnv, 'fnv_type': term_dict['type']}

    def stab_backflow(self, u_conv=None):
        ''' Backflow stabilization.

        Args:
            u_conv (Function): convecting velocity, for example Adams-Bashforth
                        interpolated for BDF2

        Returns:
            forms dict: backflow stab form
        '''
        def abs_n(x):
            return 0.5*(x - abs(x))

        ui = TrialFunction(self.Vi)
        vi = TestFunction(self.Vi)
        F, J = self.F, self.J
        rho = self.rho

        if not u_conv:
            if self._using_ale:
                u_conv = as_vector(self.upd_lst)
            else:
                u_conv = as_vector(self.u_lst)

        n = FacetNormal(self.mesh)

        ind = self.options['fem']['stabilization']['backflow_boundaries']
        self.logger.info('adding backflow stabilization on boundaries {}'.
                         format(ind))
        ds = ufl.Measure('ds', domain=self.mesh, subdomain_data=self.facet_tags)

        # a_bfs = sum([-0.5*rho*abs_n(dot(u_conv, n))*ui*vi*ds(i) for i in ind])
        if self._using_ale:
            w_conv = self.k*(self.d - self.d0)
            a_bfs = sum([-0.5*rho*abs_n(J*dot(inv(F)*(u_conv - w_conv), n))*ui*vi*ds(i) for i in ind])
        else:
            a_bfs = sum([-0.5*rho*abs_n(J*dot(inv(F)*u_conv, n))*ui*vi*ds(i) for i in ind])

        return {'bfs': a_bfs}

    def streamline_diffusion(self, u_conv):
        ''' Streamline Diffusion stabilization.
        There are different definitions for the stabilization parameter tau if
        'length_scale' is 'metric' and otherwise (average or max).
        See Shakib and Hughes (1991), "A new finite element formulation for
        computational fluid dynamics" X and IX

        Args:
            u_conv (Function): convecting velocity, for example Adams-Bashforth
                        interpolated for BDF2

        Returns:
            forms_stab (dict):  streamline diffusion stabilization forms
        '''
        opt = self.options['fem']['stabilization']['streamline_diffusion']

        mu = self.mu
        rho = self.rho
        k = self.k

        sd = SDParameter(self.options, self.mesh, mu, rho, k,
                         self._logging_filehandler)
        if self._using_ale:
            w_conv = self.k*(self.d - self.d0)
            tau = sd.stabilization_parameter(u_conv, self.u_conv_assigned,
                                             F=self.F, w=w_conv)
        else:
            w_conv = None
            tau = sd.stabilization_parameter(u_conv, self.u_conv_assigned)
        self._sd_param = sd   # stored so solver can call sd.update_tau() each step

        ui = TrialFunction(self.Vi)
        vi = TestFunction(self.Vi)
        # ALE-consistent SUPG: relative velocity (u-w), deformed spatial gradient
        # F^-T grad, deformed volume measure J*dx. Fixed-mesh form otherwise.
        if self._using_ale:
            _Finv = inv(self.F)
            _c = u_conv - w_conv
            _gx = lambda _f: dot(grad(_f), _Finv)
            residual_convdiff = rho*dot(_c, _gx(ui))
        else:
            residual_convdiff = rho*dot(u_conv, grad(ui))

        if 'consistent' in opt and opt['consistent']:
            raise Exception('Consistent SUPG not implemented because dp/dxi'
                            ' needs to be reassembled for each component.')
            res_str = 'consistent (w/o pressure)'
            residual_time = k*rho*ui
            # residual_gradp = [self.p.dx(i) for i in range(self.ndim)]
            if self.V.ufl_element.degree > 1:
                residual_convdiff += -mu*div(grad(ui))
            a_supg_time = tau*dot(u_conv, grad(vi))*residual_time*dx
        else:
            res_str = 'minimal'
            a_supg_time = None

        self.logger.info('SD/SUPG residual: {r}'.format(r=res_str))

        if self._using_ale:
            # cap quadrature: the ALE SUPG integrand has inv(F)/J (rational) →
            # UFL auto-degree explodes when summed into the conv form (FFCx hang).
            _dx_supg = dx(metadata={'quadrature_degree': 6})
            a_supg_convdiff = tau*dot(_c, _gx(vi))*residual_convdiff*self.J*_dx_supg
        else:
            a_supg_convdiff = tau*dot(u_conv, grad(vi))*residual_convdiff*dx

        return {'supg_convdiff': a_supg_convdiff, 'supg_time': a_supg_time}


    # =========================================================================
    # Boundary conditions
    # =========================================================================

    def boundary_conditions(self):
        ''' Create boundary conditions '''
        # bc_lst = copy.deepcopy(self.options['boundary_conditions'])
        # self.bc_lst = bc_lst
        BC = BoundaryConditions(self)
        BC.process_bcs()
        self.bc_dict = BC.bc_dict
        self.ds = BC.ds

        self._using_wk = BC._using_wk
        self._using_mapdd = BC._using_mapdd
        self.wk = BC.wk

class BoundaryConditions(LoggerBase):
    ''' Boundary conditions class.

    Essential is the dictionary :code:`self.bc_dict`, which is used by the
    Problem and Solver classes.

    **Note on implementation**:
    The boundary conditions are stored in the format::

        bc_dict = {
            'u': {
                'dirichlet': {
                    1: [dbc1_u1, dbc2_u1],
                    2: [dbc1_u2],
                    3: [dbc1_u3, dbc2_u3]
                },
                'neumann': [ #TODO ],
            },
            'p': { ... }
        }
    '''

    # =========================================================================
    # Initialization
    # =========================================================================

    def __init__(self, problem):
        ''' Initialize.

        Args:
            problem: Problem object
        '''
        super().__init__()
        self._logging_filehandler = problem._logging_filehandler
        if self._logging_filehandler:
            self.logger.addHandler(self._logging_filehandler)

        self.logger.info('Processing boundary conditions')
        self.bcs = problem.options['boundary_conditions']
        self.options = problem.options
        self.Vi = problem.Vi
        self.V = problem.V
        self.Q = problem.Q
        self.k = problem.k
        self.rho = problem.rho
        self.mu = problem.mu
        self.mesh = problem.mesh
        self.facet_tags = problem.facet_tags
        self.bnds = problem.bnds   # alias kept for compatibility
        self.ndim = self.mesh.topology.dim
        self.ds = ufl.Measure('ds', domain=self.mesh, subdomain_data=self.facet_tags)

        self.u_lst = problem.u_lst
        self.u0_lst = problem.u0_lst
        self.u0_mapdd_lst = problem.u0_mapdd_lst
        
        self._using_wk = False
        self._using_mapdd = False
        self.wk = {
                'explicit': False,
                'implicit': False,
                }

        for bc in self.bcs:
            if bc['type'] == 'windkessel':
                self._using_wk = True
                self.wk['implicit'] = self.options['windkessel']['implicit']
                self.wk['condensed'] = self.options['windkessel'].get('condensed', False)
                if not self.wk['implicit']:
                    self.wk['explicit'] = True

                break
            
        self.bc_dict = {
            'u': {
                'dirichlet': {i: [] for i in range(self.ndim)},
                'neumann': {i: [] for i in range(self.ndim)},
                'inflow': {i: [] for i in range(self.ndim)},
                'mapdd': {i: [] for i in range(self.ndim)},
                'navierslip': dict([(i, []) for i in range(self.ndim)]
                                   + [('coef', []), ('id', [])]),
                'transpiration': dict([(i, []) for i in range(self.ndim)]
                                      + [('coef', []), ('id', [])]),
                # FSI Robin-Neumann interface (external coupling, JellyFSI):
                # set by _fsi_robin_velocity(), consumed in
                # form_velocity_tentative() and build_rhs_tentative_velocity()
                'fsi_robin': None,
                'fsi_nitsche': None,
                'dbc_expressions': {},
                'dbc_functions': {},
                # time-dependent Neumann tractions (see _neumann_velocity):
                # {bid: {'constant', 'waveform', 'id'}}
                'neumann_expressions': {},
                # same_dbc_boundaries is used in
                # Solver.solve_tentative_velocity() in order to check if DBCs
                # have to be re-applied and the solver/pc set up for each
                # component
                'same_dbc_boundaries': True,
            },
            'p': {
                'dirichlet': [],
                'neumann': [],
                'mapdd' : {
                    'params': {},
                    'dirichlet': [],
                    'neumann': [],
                    'dbc_params': {},
                },
                'windkessel': {
                    'dirichlet': [],
                    'dbc_params': {},
                    'params': {},
                },
                # list entrys of Robin BCs are forms_p, forms_u dictionaries
                'robin': {
                    'forms_u': [],
                    'forms_p': [],
                },
                'transpiration': {
                    'coef': [],
                    'dirichlet_functions': [],
                    'dirichlet_forms_u': [],
                    'dirichlet_forms_p': [],
                }
            },
        }

        self.ale = problem.ale
        self._using_ale = problem._using_ale
        self.F, self.J = problem.F, problem.J

        if self._using_ale:
            self.D = problem.D
            self.Di = problem.Di
            
            self.bc_dict.update({
                'd': {
                    'dirichlet': {i: [] for i in range(self.ndim)},
                    'neumann': [],
                    'dbc_expressions': {},
                    'dbc_functions': {},
                    'same_dbc_boundaries': True,
                }
            })
            # same_dbc_boundaries is problem dependent! 
            self.bc_dict['u'].update({'same_dbc_boundaries': True})

    def _C(self, val):
        ''' Shorthand: fem.Constant(self.mesh, val) '''
        return fem.Constant(self.mesh, np.array(val, dtype=PETSc.ScalarType))

    # =========================================================================
    # BC processing
    # =========================================================================

    def process_bcs(self):
        ''' Call functions to process boundary conditions corresponding to
        their type.
        '''
        if self._using_ale:
            if self.ale['lifting']['type'] in ('harmonic', 'elastic',
                                                'elastic_element'):
                deformations = self.ale.get('deformations', [])
                if deformations:
                    # Explicit list — process as declared
                    for bc in deformations:
                        if not ('id' in bc and 'type' in bc):
                            raise Exception('bc dict needs keys id & type')
                        elif bc['type'] == 'dirichlet':
                            self._dirichlet_displacement(bc)
                        elif bc['type'] == 'neumann':
                            self._neumann_displacement(bc)
                        else:
                            raise Exception('Unknown ALE deformation BC type '
                                            'at boundary {}'.format(bc['id']))
                else:
                    # Auto-zero: apply zero displacement to every boundary tag
                    # except FSI interface(s) (handled by type: fsi velocity BCs).
                    fsi_tags = {bc['id'] for bc in self.bcs
                                if bc.get('type') == 'fsi'}
                    import numpy as _np
                    # Boundary tags must be gathered GLOBALLY: on a
                    # partitioned mesh a rank may own no facets of a given
                    # tag (e.g. a localized inlet/outlet), so a local
                    # unique() yields a different tag set per rank. Since
                    # _dirichlet_displacement below calls the COLLECTIVE
                    # locate_dofs_topological, an uneven per-rank tag set
                    # desyncs the collectives and the run deadlocks (all
                    # ranks busy-spin). Allgather the tag set first.
                    _local_tags = _np.unique(self.facet_tags.values)
                    _gathered = self.mesh.comm.allgather(_local_tags)
                    all_tags = set(int(v) for _a in _gathered for v in _a)
                    zero = [0.0] * self.ndim
                    for tag in sorted(all_tags - fsi_tags):
                        self._dirichlet_displacement(
                            {'id': tag, 'type': 'dirichlet', 'value': zero})
                        self.logger.debug(
                            'ALE auto-zero displacement at boundary %d', tag)
            else:
                raise Exception('lifting: {} unknown'
                                .format(self.ale['lifting']['type']))

        for bc in self.bcs:
            if not ('preset' in bc or 'type' in bc):
                raise Exception('preset or type key required for BC.')
            if 'preset' in bc:
                self._preset_selector(bc)
            elif bc['type'] == 'dirichlet':
                self._dirichlet_velocity(bc)
                # pressure: Neumann zero, do-nothing
            elif bc['type'] == 'neumann':
                # Chorin-Temam: the tentative step carries NO pressure term
                # (form_velocity_tentative sets forms['u']['pres'] = None), so
                # the prescribed pressure must enter through the projection
                # ALONE. Adding the traction on top applies the same pressure
                # a second time by an inconsistent route -- see
                # _dirichlet_pressure for the measured numbers.
                if self._ct_scheme():
                    self._dirichlet_pressure(bc)
                else:
                    # IPCS / monolithic: the momentum form does contain the
                    # pressure volume term, so the boundary traction is the
                    # genuine natural condition. Left as-is (unverified: IPCS
                    # is independently broken in this solver).
                    self._neumann_velocity(bc)
                    self._dirichlet_pressure(bc)
            elif bc['type'] == 'windkessel':
                self._windkessel(bc)
            elif bc['type'] == 'inflow':
                self._inflow_profile(bc)
            elif bc['type'] == 'inflow_pinns':
                self._inflow_pinns_profile(bc)
            elif bc['type'] == 'mapdd':
                self._mapdd(bc)
            elif bc['type'] == 'parable':
                self._parable(bc)
            elif bc['type'] == 'fsi_robin':
                self._fsi_robin_velocity(bc)
                # pressure: Neumann zero, do-nothing (same as dirichlet)
            elif bc['type'] == 'fsi_nitsche':
                self._fsi_nitsche_velocity(bc)
                # pressure: Neumann zero, do-nothing (same as dirichlet)
            elif bc['type'] == 'fsi':
                # FSI interface velocity (v_s) + ALE displacement (d_s) in one entry.
                # No 'value' key — _dirichlet_velocity creates an external Function
                # from v_s when ale.type == 'external'; same for displacement.
                self._dirichlet_velocity(bc)
                if self._using_ale and self.ale['type'] == 'external':
                    self._dirichlet_displacement({'id': bc['id'], 'type': 'dirichlet'})
            else:
                raise Exception('Unknown velocity BC at boundary {}'.
                                format(bc['id']))

        if self.options['fem']['fix_pressure'] == 1:
            self._pressure_dirichlet_point_bc()

    # =========================================================================
    # Displacement BCs
    # =========================================================================

    def _dirichlet_displacement(self, bc):
        ''' Create displacement Dirichlet boundary condition at the interface
        from options adding them into :code:`self.bc_dict['d']['dirichlet']`.
        
        Args:
            bc (dict):  dict describing the interface boundary condition.
        '''
        if not ('value' in bc):
            if self.ale['type'] == 'external':
                self.logger.info('External displacement bc at bid: {}'
                                .format(bc['id']))
                func = Function(self.D)
                facets = self.facet_tags.find(bc['id'])
                dofs = locate_dofs_topological(self.D, self.mesh.topology.dim - 1, facets)
                dbc = dirichletbc(func, dofs)
                # storing dbc in the first component
                self.bc_dict['d']['dirichlet'][0].append(dbc)
                bc_key = (bc['id'], 0)
                self.bc_dict['d']['dbc_functions'][bc_key] = {
                    'function': func, 'id': bc['id']}
            else:
                raise KeyError('bc dict needs key value')
        
        elif not isinstance(bc['value'], (tuple, list)):
            raise Exception('Expected bc[\'value\'] of velocity Dirichlet BC '
                            'to be a list [bc_u1, bc_u2(, bc_u3)], but got '
                            'type:  {}'.format(type(bc['value'])))
        
        else:
            # The displacement components are handled separately, regardless
            # if the displcamenet space is treated component-wise or not.
            expr = None

            for i, val in enumerate(bc['value']):
                if val is None:
                    self.bc_dict['d']['same_dbc_boundaries'] = False
                    continue
                
                elif isinstance(val, (int, float)):
                    val = self._C(val)

                elif isinstance(val, str):
                    # TODO: replace string Expression with fem.Function + interpolate
                    deg = bc['degree'] if 'degree' in bc else 3
                    params = bc['parameters'] if 'parameters' in bc else dict()
                    expr = Function(self.Di)
                    self.logger.warning(
                        'String-based Expression for displacement BC '
                        '(bid={}) not yet ported — BC skipped.'.format(bc['id']))
                    continue

                elif isinstance(val, fem.Function):
                    expr = val

                elif isinstance(val, fem.Constant):
                    pass

                facets = self.facet_tags.find(bc['id'])
                if self.D.element.num_sub_elements > 0:
                    D_sub, _ = self.D.sub(i).collapse()
                    dofs_pair = locate_dofs_topological(
                        (self.D.sub(i), D_sub), self.mesh.topology.dim - 1, facets)
                    # DOLFINx 0.10 returns [dofs_in_sub, dofs_in_collapsed].
                    # dirichletbc(Constant, ...) needs a 1-D array (sub-space DOFs).
                    dofs = dofs_pair[0] if isinstance(dofs_pair, (list, tuple)) else dofs_pair
                    dbc = dirichletbc(val, dofs, self.D.sub(i))
                else:
                    dofs = locate_dofs_topological(
                        self.D, self.mesh.topology.dim - 1, facets)
                    dbc = dirichletbc(val, dofs, self.D)

                self.bc_dict['d']['dirichlet'][i].append(dbc)

                if expr:
                    bc_key = (bc['id'], i)
                    self.bc_dict['d']['dbc_expressions'][bc_key] = {
                        'expression': expr, 'id': bc['id']}

    def _neumann_displacement(self, bc):
        ''' Create weak form of Neumann boundary condition.

        Args:
            bc:     dict describing one boundary condition
        '''
        deg = bc['degree'] if 'degree' in bc else 3
        params = bc['parameters'] if 'parameters' in bc else dict()
        e = TestFunction(self.D)
        n = FacetNormal(self.mesh)

        # TODO: time-dependent neumann condition 
        if self.ale['type'] in ('manual', 'external'):
            if not ('value' in bc):
                raise KeyError('bc dict needs key value')

            # TODO: string Expression not yet ported; interpolate into fem.Function
            expr = Function(self.D)
            self.logger.warning(
                'String-based Neumann displacement Expression not yet ported; '
                'using zero displacement.')
            a_bc = inner(expr, e)*self.ds(bc['id'])
            self.bc_dict['d']['neumann'].append(a_bc)
        else:
            raise Exception('Only \'manual\' and \'external\' '
                            'types available as bc for lifting')

    # =========================================================================
    # Velocity BCs
    # =========================================================================

    def _dirichlet_velocity(self, bc):
        ''' Create velocity Dirichlet boundary condition from options and add
        into :code:`self.bc_dict['u']['dirichlet'][i]`.

        Args:
            bc (dict):  dict describing one boundary condition
        '''
        if not ('id' in bc):
            raise KeyError('bc dict needs key id')

        elif not ('value' in bc):
            if self.ale['type'] == 'external':
                # self.vel_bc = Function(self.V)
                facets = self.facet_tags.find(bc['id'])
                for i in range(self.ndim):
                    expr = Function(self.Vi)

                    if self.Vi.element.num_sub_elements > 0:
                        Vi_sub, _ = self.Vi.sub(i).collapse()
                        dofs_pair = locate_dofs_topological(
                            (self.Vi.sub(i), Vi_sub),
                            self.mesh.topology.dim - 1, facets)
                        # DOLFINx 0.10: Function BC needs Sequence[ndarray] + V
                        dbc = dirichletbc(expr, dofs_pair, self.Vi.sub(i))
                    else:
                        dofs = locate_dofs_topological(
                            self.Vi, self.mesh.topology.dim - 1, facets)
                        # DOLFINx 0.10: dirichletbc(Function, dofs) — no V argument
                        dbc = dirichletbc(expr, dofs)

                    self.bc_dict['u']['dirichlet'][i].append(dbc)
                    bc_key = (bc['id'], i)
                    self.bc_dict['u']['dbc_functions'][bc_key] = {
                        'function': expr, 'id': bc['id'], 'i': i}

            elif self.ale['type'] in ('default', 'manual'):
                raise KeyError('bc dict needs keys id & value')

            else:
                raise Exception('ale type: {} not recognized'
                                .format(self.ale['type']))

        elif not isinstance(bc['value'], (tuple, list)):
            raise Exception('Expected bc[\'value\'] of velocity Dirichlet BC '
                            'to be a list [bc_u1, bc_u2(, bc_u3)], but got '
                            'type:  {}'.format(type(bc['value'])))

        else:
            #
            # The velocity components are handled separately, regardless if the
            # velocity space is treated component-wise or not.
            # bc['value'] is a list of list 'ndim', containing a boundary
            # condition for each component, possibly 'None'. These values are cast
            # into a format compatible with DirichletBC().
            # self.bc_dict['u']['dirichlet'][i] (i in range(ndim)) is a list for
            # each velocity component, that holds all corresponding Dirichlet BCs
            # definined for component i.
            # This format is convenient for applying the BCs in
            # Solver.solve_tentative_velocity()
            #
            expr = None

            for i, val in enumerate(bc['value']):
                if val is None:
                    self.bc_dict['u']['same_dbc_boundaries'] = False
                    continue

                elif isinstance(val, (int, float)):
                    val = self._C(val)

                elif isinstance(val, str):
                    params = bc.get('parameters', {})
                    try:
                        val = self._C(float(val))
                    except ValueError:
                        if val in params:
                            val = self._C(float(params[val]))
                        else:
                            self.logger.warning(
                                'String BC value "{}" at bid={}, i={} has no matching '
                                'parameter — component BC skipped.'.format(
                                    val, bc['id'], i))
                            continue

                elif isinstance(val, fem.Function):
                    expr = val

                elif isinstance(val, fem.Constant):
                    pass

                facets = self.facet_tags.find(bc['id'])
                if self.Vi.element.num_sub_elements > 0:
                    Vi_sub, _ = self.Vi.sub(i).collapse()
                    dofs_pair = locate_dofs_topological(
                        (self.Vi.sub(i), Vi_sub),
                        self.mesh.topology.dim - 1, facets)
                    if isinstance(val, fem.Function):
                        dbc = dirichletbc(val, dofs_pair, self.Vi.sub(i))
                    else:
                        dofs = dofs_pair[0] if isinstance(dofs_pair, (list, tuple)) else dofs_pair
                        dbc = dirichletbc(val, dofs, self.Vi.sub(i))
                else:
                    dofs = locate_dofs_topological(
                        self.Vi, self.mesh.topology.dim - 1, facets)
                    if isinstance(val, fem.Function):
                        # DOLFINx 0.10: Function BC infers space from Function itself
                        dbc = dirichletbc(val, dofs)
                    else:
                        dbc = dirichletbc(val, dofs, self.Vi)

                self.bc_dict['u']['dirichlet'][i].append(dbc)

                if expr:
                    bc_key = (bc['id'], i)
                    self.bc_dict['u']['dbc_expressions'][bc_key] = {
                        'expression': expr, 'id': bc['id']}

    # =========================================================================
    # Pressure BCs
    # =========================================================================

    def _dirichlet_pressure(self, bc):
        ''' Create pressure Dirichlet boundary condition from options and add
        into :code:`self.bc_dict['p']['dirichlet']`.

        Under Chorin-Temam a `type: 'neumann'` BC reduces to exactly this
        Dirichlet condition, which alone is how a pressure is prescribed in
        CT: the projection imposes p = value, and the velocity update
        u^{n+1} = u* - (dt/rho)*grad(p) delivers the pressure-driven
        acceleration. The tentative step must get nothing extra -- it has no
        pressure term at all in CT (forms['u']['pres'] is None).

        Adding the traction on top applies the same physical pressure twice,
        by a route the projection cannot reconcile. Plane Poiseuille at true
        steady state, as a fraction of the analytic Q = H^3*dp/(12*mu*L):

            pressure Dirichlet alone .................  1.004   <- correct
            + traction +value*n (old 'neumann') ......  0.122
            + traction -value*n ...................... 15.969
            traction +value*n with p = -value ........ -1.235

        So this is NOT a sign error in the traction: flipping the sign makes
        it 16x worse, and making the traction and the Dirichlet value refer
        to the same pressure (last row) merely drives the flow backwards.
        The traction term simply does not belong in the CT tentative step.

        `value` may be a float or a waveform string in `t`, e.g.
            type: 'neumann'
            value: 'P0*sin(pi*t/Th)*(t < Th)'
            parameters: {P0: 8.0e3, Th: 0.3}

        Args:
            bc (dict):  dict describing one boundary condition
        '''
        if not ('id' in bc and 'value' in bc):
            raise KeyError('bc dict needs keys id & value')

        self._check_pressure_bc_scheme(bc)

        facets = self.facet_tags.find(bc['id'])
        dofs = locate_dofs_topological(self.Q, self.mesh.topology.dim - 1, facets)
        # shares the Constant with the velocity traction for 'neumann' BCs,
        # so a time-dependent waveform drives both consistently
        self.bc_dict['p']['dirichlet'].append(
            dirichletbc(self._bc_time_constant(bc), dofs, self.Q)
        )

    def _check_pressure_bc_scheme(self, bc):
        ''' Reject a NONZERO prescribed pressure under IPCS.

        IPCS solves for a pressure INCREMENT phi (Solver.pressure_increment
        does p += phi); CT solves for p itself (phi IS p). This Dirichlet
        condition is imposed on whatever the projection step solves for, so
        under IPCS a nonzero `value` is re-added to p on that boundary EVERY
        step and the pressure grows without bound -- measured p on the
        boundary for value=10, dt=0.005:

            CT   : n=1 -> 10   n=5 -> 10   n=10 -> 10   n=40 -> 10   (correct)
            IPCS : n=1 -> 10   n=5 -> 50   n=10 -> 100  n=40 -> 400  (n*value)

        value = 0 (a stress-free / do-nothing outlet) is the common case and
        is CORRECT under IPCS: phi = 0 leaves p untouched. That is why the
        Turek FSI benchmarks run fine -- and accurately -- with IPCS.

        Fixing this properly means imposing phi = value - p^n (so that p
        lands on `value` after the increment) rather than phi = value; until
        that is implemented, fail loudly instead of silently integrating a
        ramp. Note the explicit-Windkessel pressure Dirichlet takes a
        different code path and has the same limitation.

        Args:
            bc (dict):  dict describing one boundary condition

        Raises:
            NotImplementedError: nonzero pressure Dirichlet under IPCS
        '''
        tm = self.options.get('timemarching', {})
        if tm.get('velocity_pressure_coupling') != 'fractionalstep':
            return
        if tm.get('fractionalstep', {}).get('scheme') != 'IPCS':
            return

        value = bc.get('value')
        nonzero = isinstance(value, str) or (value is not None
                                             and float(value) != 0.0)
        if nonzero:
            raise NotImplementedError(
                'Boundary {}: a nonzero prescribed pressure (value = {!r}) is '
                'not supported by the IPCS scheme -- the projection solves '
                'for a pressure INCREMENT, so the value would be re-added to '
                'the pressure every time step. Use scheme: \'CT\' for '
                'pressure-driven flow, or value: 0 for a stress-free outlet '
                '(which is correct under IPCS).'.format(bc['id'], value))

    def _neumann_velocity(self, bc):
        ''' Create weak form of Neumann boundary condition: adds +value*n to
        the momentum RHS.

        NOT called under Chorin-Temam, where a 'neumann' BC reduces to its
        pressure Dirichlet part -- see process_bcs and _dirichlet_pressure for
        why (and for the measured Poiseuille errors). Still used by IPCS and
        the monolithic solver, whose momentum forms do contain the pressure
        volume term.

        A float `value` gives a constant traction. A string `value` is a
        waveform in `t` (plus any named constants from `parameters`), e.g.

            type: 'neumann'
            value: 'P*sin(pi*t/Th)*(t<Th)'
            parameters: {P: 1.0e3, Th: 0.3}

        Time-dependent tractions register the amplitude Constant in
        `bc_dict['u']['neumann_expressions']`; the solver re-evaluates it and
        re-assembles the Neumann RHS every step (see update_neumann_bcs).

        Args:
            bc (dict):  dict describing one boundary condition
        '''
        vi = TestFunction(self.Vi)
        n = FacetNormal(self.mesh)

        val = self._bc_time_constant(bc)

        val_ = val*self.J*inv(self.F).T*n
        for i in range(self.ndim):
            # a_bc = val*n[i]*vi*self.ds(bc['id'])
            a_bc = val_[i]*vi*self.ds(bc['id'])
            self.bc_dict['u']['neumann'][i].append(a_bc)

    def _ct_scheme(self):
        ''' True if this is the non-incremental Chorin-Temam fractional step,
        whose tentative velocity step carries no pressure term. '''
        tm = self.options.get('timemarching', {})
        if tm.get('velocity_pressure_coupling') != 'fractionalstep':
            return False
        return tm.get('fractionalstep', {}).get('scheme') == 'CT'

    def _bc_time_constant(self, bc):
        ''' Resolve a BC `value` to the fem.Constant holding its current
        magnitude, compiling a waveform string in `t` if given.

        A 'neumann' entry sets BOTH the velocity traction and the pressure
        Dirichlet value, so the two must share one Constant — otherwise a
        time-dependent traction and its pressure BC drift apart. The first
        caller creates and registers it; the second gets the same object.

        Args:
            bc (dict):  dict describing one boundary condition

        Returns:
            fem.Constant
        '''
        registry = self.bc_dict['u'].setdefault('neumann_expressions', {})

        if bc['id'] in registry:
            return registry[bc['id']]['constant']

        if not isinstance(bc['value'], str):
            return self._C(float(bc['value']))

        params = {k: float(v) for k, v in bc.get('parameters', {}).items()
                  if k != 't'}
        wave = _make_time_expression(bc['value'], params)
        val = self._C(wave(0.0))
        registry[bc['id']] = {'constant': val, 'waveform': wave,
                              'id': bc['id']}
        self.logger.info('Neumann BC bid=%d: time-dependent traction "%s"',
                         bc['id'], bc['value'])
        return val

    def _neumann_pressure(self, bc):
        ''' Homogeneous Neumann BC for pressure: do nothing.

        Args:
            bc (dict):  dict describing one boundary condition
        '''
        pass

    # =========================================================================
    # FSI Robin-Neumann interface BC  (external partitioned coupling)
    # =========================================================================

    def _fsi_robin_velocity(self, bc):
        ''' FSI Robin-Neumann interface condition (Badia-Nobile-Vergara).

        Replaces the interface velocity Dirichlet BC (u = v_s) with the
        Robin closure of the momentum boundary term:

            sigma(u,p)·n = alpha*(v_rb - u) + t_rb      on ds(bc['id'])

        v_rb : solid interface velocity   (fem.Function on V)
        t_rb : previous fluid-traction estimate sigma·n (fem.Function on V)
        Both are updated by the external coupler (JellyFSI, see
        jellyfsi/robin.py) every FSI sub-iteration.

        Substituting into -∫(sigma·n)·w ds of the momentum residual:

            LHS += alpha * u_i * w * js * ds(id)          [consumed in
                                                  form_velocity_tentative()]
            RHS += (alpha*v_rb_i + t_rb_i) * w * js * ds(id)   [consumed in
                                              build_rhs_tentative_velocity()]

        js = J*||F^-T·N|| is the Nanson surface scaling (deformed boundary
        area), the same geometric factor used in _neumann_velocity.

        alpha ~ rho_s*h_s/dt gives the fluid the solid's inertial surface
        impedance, removing the added-mass instability of plain
        Dirichlet-Neumann iterations. As alpha -> inf this recovers the
        Dirichlet BC; alpha -> 0 a pure Neumann BC with traction t_rb.

        YAML:
            -   id: 5
                type: 'fsi_robin'
                parameters:
                    alpha: 2.0e4    # ~ rho_s*h_s/dt

        Args:
            bc (dict):  dict describing the boundary condition
        '''
        ui = TrialFunction(self.Vi)
        vi = TestFunction(self.Vi)
        n = FacetNormal(self.mesh)
        nans = self.J*inv(self.F).T*n
        js = sqrt(dot(nans, nans))

        alpha = self._C(bc['parameters']['alpha'])
        v_rb = Function(self.V, name='fsi_robin_velocity')
        t_rb = Function(self.V, name='fsi_robin_traction')

        lhs = alpha*ui*vi*js*self.ds(bc['id'])
        rhs = {i: (alpha*v_rb[i] + t_rb[i])*vi*js*self.ds(bc['id'])
               for i in range(self.ndim)}

        self.bc_dict['u']['fsi_robin'] = {
            'id': bc['id'],
            'alpha': alpha,
            'v_rb': v_rb,
            't_rb': t_rb,
            'lhs': lhs,
            'rhs': rhs,
        }
        self.logger.info('FSI Robin-Neumann interface at boundary {} '
                         '(alpha={})'.format(bc['id'],
                                             bc['parameters']['alpha']))

    def _fsi_nitsche_velocity(self, bc):
        ''' FSI Nitsche weak imposition of the interface velocity u = v_rb
        (Burman-Fernandez consistent-penalty). Replaces the strong interface
        velocity Dirichlet BC by symmetric Nitsche terms on the tentative
        velocity, matching the ALE viscous operator's boundary flux
        mu*(grad(u).F^-1).nans (nans = J F^-T N, the Nanson vector):

            LHS += ( -mu*(grad(u).F^-1).nans * v          consistency
                     -mu*(grad(v).F^-1).nans * u          symmetry
                     + gamma*mu/h * u * v * js ) ds        penalty
            RHS += ( -mu*(grad(v).F^-1).nans * v_rb_i
                     + gamma*mu/h * v_rb_i * v * js ) ds

        Solid takes the standard Neumann traction (Nitsche-Dirichlet/Neumann).
        v_rb (solid interface velocity) is set by update_velocity_bcs each
        sub-iteration, exactly like the Robin path. gamma >= inverse-inequality
        constant for coercivity (default 100 for P1). '''
        if self.options['fem']['strain_symmetric']:
            raise NotImplementedError(
                'fsi_nitsche not implemented for strain_symmetric viscous form')
        ui = TrialFunction(self.Vi)
        vi = TestFunction(self.Vi)
        n = FacetNormal(self.mesh)
        nans = self.J*inv(self.F).T*n
        js = sqrt(dot(nans, nans))
        h = CellDiameter(self.mesh)
        mu = self.mu
        gamma = self._C(bc.get('parameters', {}).get('gamma', 100.0))
        v_rb = Function(self.V, name='fsi_nitsche_velocity')

        _gx = lambda f: dot(grad(f), inv(self.F))     # physical spatial grad
        flux_u = mu*dot(_gx(ui), nans)                # mu (grad u . F^-1) . nans
        flux_v = mu*dot(_gx(vi), nans)
        lhs = (-flux_u*vi - flux_v*ui
               + gamma*mu/h*ui*vi*js)*self.ds(bc['id'])
        rhs = {i: (-flux_v*v_rb[i]
                   + gamma*mu/h*v_rb[i]*vi*js)*self.ds(bc['id'])
               for i in range(self.ndim)}
        self.bc_dict['u']['fsi_nitsche'] = {
            'id': bc['id'], 'gamma': gamma, 'v_rb': v_rb,
            'lhs': lhs, 'rhs': rhs,
        }
        self.logger.info('FSI Nitsche interface at boundary {} (gamma={})'
                         .format(bc['id'],
                                 bc.get('parameters', {}).get('gamma', 100.0)))

    # =========================================================================
    # Inflow BCs
    # =========================================================================

    def _inflow_profile(self, bc):
        ''' Create Inflow Profile BC

        Args:
            bc (dict):  dict describing the boundary condition
        '''
        # TODO Add to documentation and exceptions to handle 
        # bdry options such as the profile/parameters/waveform, etc!
        if not 'profile' in bc:
            raise KeyError('profile option not found. '
                           'It has to specify the path to HDF5 file')

        gamma = bc['gamma']
        F, J = self.F, self.J

        # Reading the profile
        reading_csv = False
        n = FacetNormal(self.mesh)
        uprofile = Function(self.V)
        inout.read_HDF5_data(self.mesh.comm, bc['profile'], uprofile, 'u')

        elem = J*dot(inv(F).T*n, inv(F).T*n)
        area = self.mesh.comm.allreduce(assemble_scalar(fem_form(elem*self.ds(bc['id']))), op=MPI.SUM)
        Norm_fact = abs(self.mesh.comm.allreduce(assemble_scalar(fem_form(dot(uprofile,n)*self.ds(bc['id']))), op=MPI.SUM)/area)
        ui = TrialFunction(self.Vi)
        vi = TestFunction(self.Vi)
        
        self.bc_dict['u'].update({'inflow_lhs': None } )

        if '.csv' in bc['waveform']:
            reading_csv = True
            self.logger.info('taking inflow form csv file...')
            flow_init = self.mesh.comm.allreduce(
    assemble_scalar(fem_form(dot(uprofile,n)*self.ds(bc['id']))), op=MPI.SUM)
            flip = -1 if flow_init >0 else 1
            Norm_fact = flow_init*flip            
            time_data = []
            flow_data = []
            with open(bc['waveform']) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for row in csv_reader:
                    time_data.append(float(row[0]))
                    flow_data.append(float(row[1]))
            
            inflow_func = interp1d(time_data,flow_data, kind='cubic', fill_value='extrapolate')
            waveform = self._C(inflow_func(0.0))
        else:
            elem = J*dot(inv(F).T*n, inv(F).T*n) 
            area = self.mesh.comm.allreduce(
    assemble_scalar(fem_form(elem*self.ds(bc['id']))), op=MPI.SUM)
            Norm_fact = abs(self.mesh.comm.allreduce(
    assemble_scalar(fem_form(dot(uprofile,n)*self.ds(bc['id']))), op=MPI.SUM)/area)
            params = bc['parameters']
            # TODO: string-based waveform Expression not yet ported;
            # use a fem.Constant updated by solver via waveform_func each step
            waveform = self._C(0.0)
            self.logger.warning(
                'String-based waveform Expression not yet ported; '
                'waveform initialised to zero.')


        self.bc_dict['u']['dbc_expressions']['inflow'] = {'expression': waveform, 'id': bc['id']}
        if reading_csv:
            self.bc_dict['u']['dbc_expressions']['inflow'].update({'inflow_func': inflow_func} )

        for i in range(self.ndim):
            r_bc = gamma*(uprofile.sub(i)/Norm_fact*waveform)*vi*self.ds(bc['id'])
            self.bc_dict['u']['inflow'][i].append(r_bc)

        # The left hand side can be written only once for all the components
        self.bc_dict['u']['inflow_lhs'] = gamma*ui*vi*self.ds(bc['id'])
    
    def _inflow_pinns_profile(self, bc):
        ''' Create Inflow Profile BC based on PINNs solution

        Args:
            bc (dict):  dict describing the boundary condition
        '''

        gamma = bc['gamma']
        F, J = self.F, self.J

        # Reading the profile
        n = FacetNormal(self.mesh)
        uprofile = {0: Function(self.Vi), 1: Function(self.Vi), 2: Function(self.Vi)}

        ui = TrialFunction(self.Vi)
        vi = TestFunction(self.Vi)
        
        self.bc_dict['u'].update({'inflow_lhs': None } )

        u_data = load(bc['velocity_data'], allow_pickle = True)
        # loading initial time data
        uprofile[0].x.array[:] = u_data['ux'].item()[0][:,0]
        uprofile[1].x.array[:] = u_data['uy'].item()[0][:,0]
        uprofile[2].x.array[:] = u_data['uz'].item()[0][:,0]

        self.bc_dict['u']['dbc_expressions']['inflow'] = {'pinns_data': u_data, 'id': bc['id'], 'uprofile': uprofile}

        for i in range(self.ndim):
            r_bc = gamma*uprofile[i]*vi*self.ds(bc['id'])
            self.bc_dict['u']['inflow'][i].append(r_bc)

        # The left hand side can be written only once for all the components
        self.bc_dict['u']['inflow_lhs'] = gamma*ui*vi*self.ds(bc['id'])

    # =========================================================================
    # MAPDD / Windkessel
    # =========================================================================

    def _mapdd(self,bc):
        '''
            method of asymptotic partial decomposition of a domain (MAPDD) proposed
            in Bertoglio et al (2019).

        '''
        self._using_mapdd = True
    
        ui = TrialFunction(self.Vi)
        vi = TestFunction(self.Vi)
        n = FacetNormal(self.mesh)

        l = bc['parameters']['l']
        rho = self.rho
        mu = self.mu
        k = self.k
        bid = bc['id']

        self.bc_dict['p']['mapdd']['params'][bid] = {
            'fmass': self._C(rho*k),
            'fdiff': self._C(mu),
            'l': self._C(l),
        }

        self.bc_dict['u'].update({'mapdd_lhs': [] } )

        prm = self.bc_dict['p']['mapdd']['params'][bid]
        
        for i in range(self.ndim):
            a_rhs = prm['l']*rho*k*self.u0_mapdd_lst[i]*vi*self.ds(bc['id'])
            self.bc_dict['u']['mapdd'][i].append(a_rhs)

        # tangential component
        gradu_t = grad(ui) - dot(grad(ui),n)*n 
        gradv_t = grad(vi) - dot(grad(vi),n)*n
        a_lhs = 0
        a_lhs += prm['l']*mu*dot(gradu_t,gradv_t)*self.ds(bc['id'])
        a_lhs += prm['l']*rho*k*ui*vi*self.ds(bc['id'])
        self.bc_dict['u']['mapdd_lhs'].append(a_lhs)

    def _parable(self, bc):
        '''Parabolic (paraboloid) fully-Dirichlet inlet BC.

        The spatial shape is computed automatically from the inlet DOF
        positions via SVD — no geometry assumptions or manual radius needed.
        Works for any inlet orientation in 2-D and 3-D.

        Parameters in bc dict
        ---------------------
        id       : boundary tag ID
        U        : peak velocity at the centroid (float)
        waveform : (optional) path to a CSV file with two columns (time, Q)
                   where Q is the total volumetric flow rate at that time.
                   Omit for a constant-in-time profile at amplitude U.
        '''
        if 'U' not in bc['parameters']:
            raise KeyError("'parable' BC requires 'parameters: U' "
                           "(peak velocity at the centroid)")

        U = float(bc['parameters']['U'])
        _eps = 1e-12
        fdim = self.ndim - 1
        facets = self.facet_tags.find(bc['id'])

        # --- inlet DOF coordinates ---
        all_coords = self.Vi.tabulate_dof_coordinates()   # (total_dofs, 3)
        dofs_Vi = locate_dofs_topological(self.Vi, fdim, facets)
        inlet_coords = all_coords[dofs_Vi]                # (N_inlet, 3) LOCAL

        # The SVD frame (centroid, in-plane axes, radii) and the inward-
        # normal test must come from the GLOBAL inlet point cloud: on a
        # partitioned mesh the inlet usually lives on only a few ranks, so
        # a per-rank frame would be inconsistent (or undefined on ranks
        # owning no inlet dofs, which previously raised here). Gather the
        # inlet coords and the mesh-dof centroid across all ranks.
        comm = self.mesh.comm
        _parts = [a for a in comm.allgather(inlet_coords) if len(a)]
        global_inlet = (np.concatenate(_parts, axis=0)
                        if _parts else inlet_coords[:0])
        if len(global_inlet) == 0:
            raise RuntimeError(
                'Parabolic BC: no DOFs found on boundary id={}'.format(bc['id']))
        _gsum = comm.allreduce(all_coords.sum(axis=0), op=MPI.SUM)
        _gcnt = comm.allreduce(len(all_coords), op=MPI.SUM)
        mesh_centroid = _gsum / max(_gcnt, 1)

        # --- SVD frame from the GLOBAL inlet cloud ---
        centroid, t1, t2, n_hat, R1, R2 = \
            _compute_inlet_local_frame(global_inlet)

        # orient normal inward (towards mesh interior), global centroid
        if np.dot(mesh_centroid - centroid, n_hat) < 0:
            n_hat = -n_hat

        self.logger.info(
            'Parabolic BC bid={}: R1={:.4g}, R2={:.4g}, '
            'n=[{:.3g},{:.3g},{:.3g}]'.format(bc['id'], R1, R2, *n_hat))

        # --- spatial profile factory (returns scalar callable for Vi) ---
        def _make_interp(comp, scale):
            _c, _t1, _t2 = centroid, t1, t2
            _n, _R1, _R2 = n_hat, R1, R2
            _U, _s, _i = U, scale, comp
            def _f(x):
                pts = x.T - _c
                profile = 1.0 - (pts @ _t1 / _R1) ** 2
                if _R2 > _eps:
                    profile -= (pts @ _t2 / _R2) ** 2
                return _U * _s * np.clip(profile, 0.0, None) * _n[_i]
            return _f

        # --- temporal waveform → unified scale_func(t) → float ---
        # Reserved parameters not treated as expression variables:
        _reserved = {'U', 'waveform'}
        scale_func = None   # None means constant (no update needed)

        if 'waveform' in bc['parameters']:
            waveform = bc['parameters']['waveform']

            if isinstance(waveform, str) and waveform.endswith('.csv'):
                # CSV mode: columns (time, Q_total); normalise by unit-profile flow
                time_data, flow_data = [], []
                with open(waveform) as csv_file:
                    for row in csv.reader(csv_file, delimiter=','):
                        time_data.append(float(row[0]))
                        flow_data.append(float(row[1]))
                _interp_csv = interp1d(time_data, flow_data,
                                       kind='cubic', fill_value='extrapolate')

                # numerical Norm_fact: total flow of the unit parabolic profile
                n_facet = FacetNormal(self.mesh)
                u_tmp = Function(self.V)

                def _unit_profile(x,
                                  _c=centroid, _t1=t1, _t2=t2, _n=n_hat,
                                  _R1=R1, _R2=R2, _nd=self.ndim):
                    pts = x.T - _c
                    prof = 1.0 - (pts @ _t1 / _R1) ** 2
                    if _R2 > _eps:
                        prof -= (pts @ _t2 / _R2) ** 2
                    prof = np.clip(prof, 0.0, None)
                    vals = np.zeros((_nd, x.shape[1]))
                    for k in range(_nd):
                        vals[k] = prof * _n[k]
                    return vals

                u_tmp.interpolate(_unit_profile)
                Norm_fact = abs(self.mesh.comm.allreduce(
                    assemble_scalar(
                        fem_form(dot(u_tmp, n_facet) * self.ds(bc['id']))),
                    op=MPI.SUM))
                if Norm_fact < _eps:
                    raise RuntimeError(
                        'Parabolic BC bid={}: unit-flow Norm_fact ≈ 0 ({:.2e}). '
                        'Check boundary id.'.format(bc['id'], Norm_fact))

                scale_func = lambda t, _f=_interp_csv, _n=Norm_fact: float(_f(t)) / _n
                self.logger.info(
                    'Parabolic BC bid={}: CSV waveform, Norm_fact={:.4g}'
                    .format(bc['id'], Norm_fact))

            elif isinstance(waveform, str):
                # Expression mode: e.g. 'cos(w*t)'; extra bc parameters are
                # substituted as named variables. 't' is reserved for time.
                extra = {k: float(v) for k, v in bc['parameters'].items()
                         if k not in _reserved}
                if 't' in extra:
                    raise KeyError(
                        "'t' is reserved for simulation time and cannot be "
                        "used as a waveform parameter in parable BC "
                        "bid={}".format(bc['id']))
                _expr_str = waveform
                _math_ns = {
                    'sin': np.sin,  'cos': np.cos,  'tan': np.tan,
                    'exp': np.exp,  'log': np.log,  'sqrt': np.sqrt,
                    'abs': np.abs,  'tanh': np.tanh, 'pi': np.pi,
                    'min': min,     'max': max,
                }
                def scale_func(t, _e=_expr_str, _p=extra, _m=_math_ns):
                    ns = {'t': float(t)}
                    ns.update(_p)
                    ns.update(_m)
                    return float(eval(_e, {"__builtins__": {}}, ns))

                self.logger.info(
                    'Parabolic BC bid={}: expression waveform "{}", '
                    'params={}'.format(bc['id'], waveform, extra))

        # --- period / cycles: repeat scale_func over a fixed T_cycle ---
        # Two equivalent ways to declare a repeating waveform (BC top level):
        #   period: 0.5          T_cycle = 0.5 s (explicit; preferred — stable
        #                        if timemarching.T changes)
        #   cycles: 2            T_cycle = timemarching.T / 2  (derived)
        # period takes priority. Effective time: t_eff = t % T_cycle.
        _period  = bc.get('period', None)
        _ncycles = max(1, int(bc.get('cycles', 1)))
        if _period is not None:
            T_cycle = float(_period)
        elif _ncycles > 1:
            T_cycle = float(self.options['timemarching']['T']) / _ncycles
        else:
            T_cycle = None

        if T_cycle is not None and scale_func is not None:
            _orig = scale_func
            scale_func = lambda t, _f=_orig, _tc=T_cycle: _f(t % _tc)
            self.logger.info(
                'Parabolic BC bid={}: T_cycle={:.4g}s'.format(bc['id'], T_cycle))

        t0_scale = scale_func(0.0) if scale_func is not None else 1.0

        # --- per-component Functions and DirichletBCs ---
        # DOLFINx dirichletbc overloads for a Function:
        #   (Function, ndarray)                        – scalar/non-sub space
        #   (Function, Sequence[ndarray], SubSpace)    – enriched/sub space
        parable_funcs = []
        enriched = self.Vi.element.num_sub_elements > 0
        for i in range(self.ndim):
            if enriched:
                Vi_sub, _ = self.Vi.sub(i).collapse()
                func_i = Function(Vi_sub)
                func_i.interpolate(_make_interp(i, t0_scale))
                dofs_i = locate_dofs_topological(
                    (self.Vi.sub(i), Vi_sub), fdim, facets)
                dbc_i = dirichletbc(func_i, dofs_i, self.Vi.sub(i))
            else:
                func_i = Function(self.Vi)
                func_i.interpolate(_make_interp(i, t0_scale))
                dofs_i = locate_dofs_topological(self.Vi, fdim, facets)
                dbc_i = dirichletbc(func_i, dofs_i)

            self.bc_dict['u']['dirichlet'][i].append(dbc_i)
            parable_funcs.append(func_i)

        # --- register for time updates (any time-dependent waveform) ---
        if scale_func is not None:
            self.bc_dict['u']['dbc_expressions'][('parable', bc['id'])] = {
                'parable_funcs': parable_funcs,
                'parable_scale_func': scale_func,
                'centroid': centroid,
                't1': t1, 't2': t2, 'n': n_hat,
                'R1': R1, 'R2': R2, 'U': U,
                'id': bc['id'],
            }
    

    def _windkessel(self, bc):
        ''' Create Windkessel pressure BC.

        Args:
            bc (dict):  dict describing one boundary condition
        '''
        if not 'parameters' in bc:
            raise KeyError('Winkessel BCs require parameters')

        ds = self.ds
        F, J = self.F, self.J
        n = FacetNormal(self.mesh)

        dt = self.options['timemarching']['dt']
        bid = bc['id']
        R_p = bc['parameters']['R_p']
        R_d = bc['parameters']['R_d']
        C = bc['parameters']['C']
        eps = bc['parameters']['eps'] if 'eps' in bc['parameters'] else 0

        alpha = R_d*C/(R_d*C + dt)
        beta = R_d*(1 - alpha)
        gamma = R_p + beta

        # Setting initial values
        elem = J*dot(inv(F).T*n, inv(F).T*n)
        area = self.mesh.comm.allreduce(
    assemble_scalar(fem_form(elem*ds(bid))), op=MPI.SUM)
        self.logger.info(f'area (problem): {area}')
        Q = float(0)
        delta_r = alpha/gamma
        delta_l = 1/gamma

        # ---------------------------------------------------------------
        # p0 (YAML) == the RESERVOIR/CAPACITOR pressure at t=0, pi(0), for BOTH
        # schemes.  What each scheme stores internally differs, and that used to
        # be resolved by a SILENT OVERWRITE further down (the implicit branch
        # re-assigned prm['pi'] = pi0 after this block had set alpha*pi0), so
        # you could not tell from the YAML what pi(0) would actually be.
        # Resolved explicitly here instead:
        #   implicit: the pressure operator enforces Pl = gamma*Q + alpha*pi and
        #             the update is pi <- alpha*pi + beta*Q, whose steady state
        #             is pi = R_d*Q.  Hence pi(0) IS p0, used UNSCALED.
        #   explicit: Pl = R_p*Q + pi, seeded by one backward-Euler step from
        #             rest (Q=0), hence pi(0) = alpha*p0.
        # => never pre-divide p0 by alpha in the YAML.  The resolved value is
        #    logged below so the run record is unambiguous.
        # ---------------------------------------------------------------
        pi0 = bc['parameters']['p0']
        P0 = alpha*pi0
        pi = pi0 if self.wk.get('implicit') else P0
        
        self.bc_dict['p']['windkessel']['params'][bid] = {
            'eps': self._C(eps),
            'R_p': self._C(R_p),
            'R_d': self._C(R_d),
            'C': self._C(C),
            'pi0': self._C(pi0),
            'pi': self._C(pi),
            'Q': self._C(Q), # initial flow to zero
            'Pl': self._C(P0),
            'delta_l': self._C(delta_l),
            'delta_r': self._C(delta_r),
            'area': self._C(area),
            'alpha': self._C(alpha),
            'beta': self._C(beta),
            'gamma': self._C(gamma)
        }

        prm = self.bc_dict['p']['windkessel']['params'][bid]

        self.logger.info(
            'Windkessel bid %d (%s): R_p=%.6g R_d=%.6g C=%.6g  '
            'tau=R_d*C=%.4g s  alpha=%.8f  |  yaml p0=%.6g -> pi(0)=%.6g, '
            'Pl(0)=%.6g [dyn/cm2]',
            bid, 'implicit' if self.wk.get('implicit') else 'explicit',
            R_p, R_d, C, R_d*C, alpha, pi0, float(prm['pi'].value),
            float(prm['Pl'].value))

        if self.wk['explicit']:
            # Use prm['Pl'] directly as a fem.Constant; solver updates .value each step
            facets = self.facet_tags.find(bid)
            dofs = locate_dofs_topological(self.Q, self.mesh.topology.dim - 1, facets)
            dbc = dirichletbc(prm['Pl'], dofs, self.Q)
            self.bc_dict['p']['windkessel']['dirichlet'].append(dbc)
            bc_key = bid
            self.bc_dict['p']['windkessel']['dbc_params'][bc_key] = {
                    'Pl_const': prm['Pl'], 'bid': bid}
        elif self.wk['implicit']:
            # prm['pi'] was already initialised to p0 in the block above (see
            # the note there).  The scheme-dependent overwrite that used to sit
            # here is gone -- pi(0) is now decided in exactly one place.
            prm['rhs_p'] = prm['Q'] + prm['delta_r']*prm['pi']

    # =========================================================================
    # Robin BCs (Navier-slip and transpiration)
    # =========================================================================

    def _navierslip_velocity(self, bc):
        r''' Create weak forms of Navier-slip boundary condition, for each
        component, i,

        .. math::

            g\int_{\Gamma_i} u_i v_i (1- n_i^2) - v_i n_i (\sum_{j=1, j\neq
            i} u_j n_j)

        where :math:`\gamma` is the slip coefficient.

        For each i, creates the "diagonal" term :math:`(u_i,\ v_i)` and the
        ndim-1 cross terms. The resulting forms are stored in a (diag,
        cross)-dict within :code:`bc_dict`, need to be assembled into matrices
        (even in the explicit case) and multiplied by the corresponding
        velocity component vectors.

        Note that the coefficients are not included in the integrals. The
        assembled matrices have to be multiplied by the (possibly) varying
        coefficient.

        Args:
            bc (dict):  dict describing one boundary condition
        '''
        ui = TrialFunction(self.Vi)
        vi = TestFunction(self.Vi)
        n = FacetNormal(self.mesh)
        val = self._C(bc['value'])

        self.bc_dict['u']['navierslip']['coef'].append(val)
        self.bc_dict['u']['navierslip']['id'].append(bc['id'])
        method = (self.options['timemarching']['fractionalstep']
                  ['robin_bc_velocity_scheme'])

        # # START DEBUG
        # self.logger.warn('DEBUG NAVIERSLIP FOR 2D PIPE')
        # a_implicit = -ui*vi*self.ds(bc['id'])
        # # ui*n[1]*vi*n[0] == 0
        # a_explicit = {1: self._C(0)*ui*n[1]*vi*n[0]*self.ds(bc['id'])}
        # if bc['method'] == 'explicit':
        #     a_explicit.update({0: a_implicit})
        #     a_implicit = None

        # self.bc_dict['u']['navierslip'][0].append(
        #     {'semi-implicit': a_implicit, 'explicit': a_explicit}
        # )

        # a_implicit = -self._C(0)*ui*vi*self.ds(bc['id'])
        # # ui*n[1]*vi*n[0] == 0
        # a_explicit = {0: self._C(0)*ui*n[1]*vi*n[0]*self.ds(bc['id'])}
        # if bc['method'] == 'explicit':
        #     a_explicit.update({1: a_implicit})
        #     a_implicit = None
        # self.bc_dict['u']['navierslip'][1].append(
        #     {'semi-implicit': a_implicit, 'explicit': a_explicit}
        # )

        # return

        # # END DEBUG

        for i in range(self.ndim):
            # a_bc has sign of LHS
            a_implicit = ui*vi*(1 - n[i]**2)*self.ds(bc['id'])

            a_explicit = {}
            for j in range(self.ndim):
                if not i == j:
                    a_explicit.update({j: -ui*n[j]*vi*n[i]*self.ds(bc['id'])})

            if method == 'explicit':
                a_explicit.update({i: a_implicit})
                a_implicit = None

            self.bc_dict['u']['navierslip'][i].append(
                {'semi-implicit': a_implicit, 'explicit': a_explicit}
            )

    def _transpiration_velocity(self, bc):
        r''' Create weak forms of the transpiration boundary condition, for each
        component, i,

        .. math::

            \beta\int_{\Gamma_i} u_i v_i n_i^2 - v_i n_i (\sum_{j=1, j\neq i}
                u_j n_j)

        For each i, creates the "diagonal" term :math:`(u_i, v_i)` and the
        ndim-1 cross terms. The resulting forms are stored in a (diag,
        cross)-dict within :code:`bc_dict`, need to be assembled into matrices
        (even in the explicit case) and multiplied by the corresponding
        velocity component vectors.

        Note that the coefficients are not included in the integrals. The
        assembled matrices have to be multiplied by the (possibly) varying
        coefficient.

        Args:
            bc (dict):  dict describing one boundary condition
        '''
        ui = TrialFunction(self.Vi)
        vi = TestFunction(self.Vi)
        p = TrialFunction(self.Q)
        n = FacetNormal(self.mesh)
        val = self._C(bc['value'])
        self.bc_dict['u']['transpiration']['coef'].append(val)
        self.bc_dict['u']['transpiration']['id'].append(bc['id'])
        method = (self.options['timemarching']['fractionalstep']
                  ['robin_bc_velocity_scheme'])

        for i in range(self.ndim):
            # a_bc has sign of LHS
            a_implicit = ui*vi*n[i]**2*self.ds(bc['id'])

            a_explicit = {}
            for j in range(self.ndim):
                if not i == j:
                    a_explicit.update({
                        j: ui*n[j]*vi*n[i]*self.ds(bc['id'])
                    })

            # if self.options['scheme'] == 'CT':
            # for CT scheme
            a_pressure = -p*vi*n[i]*self.ds(bc['id'])

            if method == 'explicit':
                a_explicit.update({i: a_implicit})
                a_implicit = None

            self.bc_dict['u']['transpiration'][i].append(
                {'semi-implicit': a_implicit, 'explicit': a_explicit,
                 'pressure': a_pressure}
            )

    def _transpiration_pressure(self, bc):
        r''' Set up Transpiration BCs for the pressure projection step.
        Two approaches are available via the option
        :code:`timemarching: fractionalstep: transpiration_bc_projection`

        * dirichlet: Dirichlet BCs

          .. math::

                p = \beta u\cdot n

        * robin: Robin BCs

          .. math::

                \partial p/\partial n + \rho k/\beta p = \rho k u\cdot n

        Args:
            bc (dict):  dict describing one boundary condition
        '''
        if utils.is_enriched(self.V):
            raise Exception('Transpiration BCs currently not supported for '
                            'enriched elements.')

        vel_tbc_ids = self.bc_dict['u']['transpiration']['id']
        if bc['id'] not in vel_tbc_ids:
            raise Exception('Transpiration coefficient was not set for '
                            'boundary {}. Velocity boundary condition needs to'
                            ' be created first!'.format(bc['id']))
        assert bc['id'] == self.bc_dict['u']['transpiration']['id'][-1]

        coef_index = vel_tbc_ids.index(bc['id'])
        val = self.bc_dict['u']['transpiration']['coef'][coef_index]
        assert isinstance(val, Constant)
        self.bc_dict['p']['transpiration']['coef'].append(val)
        n = FacetNormal(self.mesh)

        method = (self.options['timemarching']['fractionalstep']
                  ['transpiration_bc_projection'])

        if method == 'dirichlet':
            # need to project $u\cdot n$ onto $Q$
            u = TrialFunction(self.V)
            p = TrialFunction(self.Q)
            q = TestFunction(self.Q)
            form_p = p*q*self.ds(bc['id'])
            form_u = dot(u, n)*q*self.ds(bc['id'])
            # a_proj_trans = dot(u, n)*q*self.ds(bc['id']) + \
            #     -1./1000*0.035*dot(dot(grad(u), n), n)*q*self.ds(bc['id'])
            # a_proj_trans = dot(u, n)*q*self.ds
            p_trans = Function(self.Q, name='p_trans')
            facets = self.facet_tags.find(bc['id'])
            dofs = locate_dofs_topological(self.Q, self.mesh.topology.dim - 1, facets)
            bc_p = dirichletbc(p_trans, dofs)
            self.bc_dict['p']['dirichlet'].append(bc_p)
            # boundary function is stored for convenience; DirichletBC detects
            # changes in function p_trans, does not need to be recreated.
            self.bc_dict['p']['transpiration']['dirichlet_functions'].append(
                p_trans)
            self.bc_dict['p']['transpiration']['dirichlet_forms_u'].append(
                form_u)
            self.bc_dict['p']['transpiration']['dirichlet_forms_p'].append(
                form_p)

        elif method == 'robin':
            q = TestFunction(self.Q)
            p = TrialFunction(self.Q)
            u = TrialFunction(self.V)
            rho, k = self.rho, self.k
            # signs wrt dp/dn + a*p = b*f(u)
            form_p = rho*k*p*q*self.ds(bc['id'])
            form_u = rho*k*dot(u, n)*q*self.ds(bc['id'])

            self.bc_dict['p']['robin']['forms_p'].append(form_p)
            self.bc_dict['p']['robin']['forms_u'].append(form_u)

        else:
            raise Exception('Transpiration pressure method "{}" unknown'.
                            format(method))

    # =========================================================================
    # Preset BCs
    # =========================================================================

    def _preset_selector(self, bc):
        ''' Prepare preset boundary condition.
        Create data sets and call :py:meth:`~.dirichlet_velocity`,
        :py:meth:`~.dirichlet_pressure`, and/or neumann as necessary.

        Recognized options:
         *   noslip
         *   parabola_inlet
         *   outlet
         *   symmetry
         *   navierslip
         *   transpiration
         *   sine_parabola_inlet

        Args:
            bc (dict):  dict describing one boundary condition
        '''
        preset = bc['preset']

        if preset == 'parabola_inlet':
            self._preset_parabola_inlet_bc(bc)
            # pressure: Neumann zero, do-nothing

        elif preset == 'sine_parabola_inlet':
            self._preset_sine_parabola_inlet_bc(bc)
            # pressure: Neumann zero, do-nothing

        elif preset == 'noslip':
            self._preset_noslip_bc(bc)
            # pressure: Neumann zero, do-nothing

        elif preset == 'outlet':
            self.logger.warn('BC type OUTLET deprecated (missleading), use '
                             'Neumann instead!')
            self._preset_outlet_bc(bc)
            # pressure: Neumann zero, do-nothing

        elif preset == 'symmetry':
            self._preset_symmetry_bc(bc)
            # pressure: Neumann zero, do-nothing

        elif preset == 'navierslip':
            self._preset_navierslip_bc(bc)
            # pressure: Neumann zero, do-nothing

        elif preset == 'transpiration':
            self._preset_transpiration_bc(bc)

        else:
            raise Exception('Velocity BC preset type \'{0}\' not recognized.'.
                            format(preset))

    def _preset_parabola_inlet_bc(self, bc):
        ''' Process preset inlet boundary condition, create format that
        function dirichlet understands.

        Args:
            bc (dict):  dict describing one boundary condition
        '''
        if 'method' in bc and bc['method'] == 'nitsche':
            self.logger.warn('Inlet BCs are imposed strongly. Ignoring '
                             'Nitsche setting.')

        if not isinstance(bc['value'], fem.Function):
            flow_direction = bc['flow_direction'] if 'flow_direction' in bc \
                else 0
            assert flow_direction in range(self.ndim), (
                'flow direction {} does not match problem dimension'
                .format(flow_direction))

            if 'bnd_interval' in bc['value'] and bc['value']['bnd_interval']:
                bnd_interval = bc['value']['bnd_interval']
                r0 = 0.5*(bnd_interval[1] - bnd_interval[0])
                x0 = 0.5*(bnd_interval[1] + bnd_interval[0])
                if 'R' in bc['value'] and bc['value']['R']:
                    self.logger.warning('BC {}: bnd_interval was set, ignoring'
                                        ' R'.format(bc['id']))
            else:
                r0 = bc['value']['R']
                x0 = 0.

            # get the indices orthogonal to flow direction (coordinates of
            # parabola)
            indices = list(range(self.ndim))
            indices.remove(flow_direction)

            if self.ndim == 2:
                inflow_str = ('U*(1 - pow(x[{0}] - x0, 2)/(R*R))'
                              .format(*indices))
            elif self.ndim == 3:
                self.logger.warn('3D paraboloidal inlet profile only valid for'
                                 ' circular cross sections')
                inflow_str = ('U/(R*R)*(R*R - pow(x[{0}] - x0, 2) - '
                              'pow(x[{1}] - x0, 2))'.format(*indices))

            U_val = float(bc['value']['U'])
            x0_val = float(x0)
            r0_val = float(r0)
            idx = indices  # radial axis indices

            inflow = []
            for comp in range(self.ndim):
                f = Function(self.Vi)
                if comp == flow_direction:
                    if self.ndim == 2:
                        f.interpolate(lambda x, _U=U_val, _x0=x0_val, _r=r0_val, _ax=idx[0]:
                            _U * (1.0 - (x[_ax] - _x0)**2 / _r**2))
                    else:
                        f.interpolate(lambda x, _U=U_val, _x0=x0_val, _r=r0_val, _a=idx[0], _b=idx[1]:
                            _U / _r**2 * (_r**2 - (x[_a] - _x0)**2 - (x[_b] - _x0)**2))
                else:
                    f.interpolate(lambda x: np.zeros(x.shape[1]))
                inflow.append(f)
            bc['value'] = inflow

        self._dirichlet_velocity(bc)

    def _preset_sine_parabola_inlet_bc(self, bc):
        ''' Process preset sine oscillation inlet boundary condition, create
        format that function dirichlet_velocity() understands.

        Args:
            bc (dict):  dict describing one boundary condition
        '''
        if 'method' in bc and bc['method'] == 'nitsche':
            self.logger.warn('Inlet BCs are imposed strongly. Ignoring '
                             'Nitsche setting.')

        if not isinstance(bc['value'], fem.Function):
            # shallow copy for ReynoldsContinuation
            flow_direction = bc['flow_direction'] if 'flow_direction' in bc \
                else 0
            assert flow_direction in range(self.ndim), (
                'flow direction {} does not match problem dimension'
                .format(flow_direction))

            if 'bnd_interval' in bc['value'] and bc['value']['bnd_interval']:
                bnd_interval = bc['value']['bnd_interval']
                r0 = 0.5*(bnd_interval[1] - bnd_interval[0])
                x0 = 0.5*(bnd_interval[1] + bnd_interval[0])
                if 'R' in bc['value'] and bc['value']['R']:
                    self.logger.warning('BC {}: bnd_interval was set, ignoring'
                                        ' R'.format(bc['id']))
            else:
                r0 = bc['value']['R']
                x0 = 0.

            # get the indices orthogonal to flow direction (coordinates of
            # parabola)
            indices = list(range(self.ndim))
            indices.remove(flow_direction)

            if self.ndim == 2:
                inflow_str = ('U*(1 - pow(x[{0}] - x0, 2)/(R*R))*'
                              'sin(a*DOLFIN_PI*t)'.format(*indices))
            elif self.ndim == 3:
                self.logger.warn('3D paraboloidal inlet profile only valid for'
                                 ' circular cross sections')
                inflow_str = ('U/(R*R)*(R*R - pow(x[{0}] - x0, 2) -'
                              'pow(x[{1}] - x0, 2))*sin(a*DOLFIN_PI*t)'
                              .format(*indices))

            U_val  = float(bc['value']['U'])
            x0_val = float(x0)
            r0_val = float(r0)
            a_val  = float(bc['value']['a'])
            idx    = indices
            # time_factor is updated by the solver each step
            time_factor = self._C(0.0)

            inflow = []
            for comp in range(self.ndim):
                f = Function(self.Vi)
                if comp == flow_direction:
                    if self.ndim == 2:
                        f.interpolate(lambda x, _U=U_val, _x0=x0_val, _r=r0_val, _ax=idx[0]:
                            _U * (1.0 - (x[_ax] - _x0)**2 / _r**2))
                    else:
                        f.interpolate(lambda x, _U=U_val, _x0=x0_val, _r=r0_val, _a=idx[0], _b=idx[1]:
                            _U / _r**2 * (_r**2 - (x[_a] - _x0)**2 - (x[_b] - _x0)**2))
                else:
                    f.interpolate(lambda x: np.zeros(x.shape[1]))
                inflow.append(f)

            # store time_factor + a so solver can update: time_factor.value = sin(a*pi*t)
            bc['time_factor'] = time_factor
            bc['a'] = a_val
            bc['value'] = [time_factor * f for f in inflow]

        # self.bc_dict['u']['time_bcs'].append(
        #     {'expression': bc['value'][flow_direction], 'id': bc['id'], 'i':
        #      flow_direction})
        # save id, i for enriched elements

        self._dirichlet_velocity(bc)

    def _preset_noslip_bc(self, bc):
        ''' Process preset no-slip boundary condition, create format that
        function dirichlet understands.

        Args:
            bc (dict):  dict describing one boundary condition
        '''
        self._dirichlet_velocity(bc)

    def _preset_outlet_bc(self, bc):
        ''' Process preset pressure outlet boundary condition. This is taken to
        be a value, g, for :code:`-(mu*grad(u)*n - p*n) = g*n`.
        In the fractional step solvers, g is imposed as a dirichlet BC on the
        pressure.
        Note that the same (outlet) BC or a corresponding Dirichlet BC needs to
        be defined for the pressure!

        Args:
            bc (dict):  dict describing one boundary condition
        '''
        self._neumann_velocity(bc)
        self._dirichlet_pressure(bc)

    def _preset_symmetry_bc(self, bc):
        r''' Process preset symmetry boundary condition.

        Create Dirichlet BC for :math:u\cdot n = 0` and Neumann for
        :math:`n\cdot\sigma t = 0`.
        For the pressure, a homogeneous Neumann BC should be set.

        Args:
            bc (dict):  dict describing one boundary condition
        '''
        # bc_n = {'id': bc['id'],
        #         'type': 'neumann',
        #         'value': [None, 0.]
        #         }
        bc_d = {'id': bc['id'],
                'method': bc['method'] if 'method' in bc else 'essential'
                }
        if bc_d['method'] == 'nitsche':
            bc_d['value'] = 0.
            raise Exception('Nitsche BCs currently not supported')
            # self._proc_nitsche_bc(bc_d)
        else:
            symm_normal_dir = bc['normal_dir']
            val = [None]*self.ndim
            val[symm_normal_dir] = 0.
            bc_d['value'] = val
            self._dirichlet_velocity(bc_d)

    def _preset_navierslip_bc(self, bc):
        ''' Process preset Navier-slip boundary condition, create format that
        function navierslip understands.

        Args:
            bc (dict):  dict describing one boundary condition
        '''
        if not isinstance(bc['value'], (int, float, fem.Function, fem.Constant)):
            raise Exception('value of navierslip BC needs to be of type (int,'
                            ' float, Constant, Function)')

        if isinstance(bc['value'], (int, float)):
            bc['value'] = self._C(bc['value'])

        self._navierslip_velocity(bc)
        # pressure do nothing

    def _preset_transpiration_bc(self, bc):
        ''' Process preset transpiration boundary condition, create format that
        function transpiration understands.

        Args:
            bc (dict):  dict describing one boundary condition
        '''
        if not isinstance(bc['value'], (int, float, fem.Function, fem.Constant)):
            raise Exception('value of transpiration BC needs to be of type '
                            '(int, float, Constant, Function)')

        if isinstance(bc['value'], (int, float)):
            bc['value'] = self._C(bc['value'])

        self._transpiration_velocity(bc)
        self._transpiration_pressure(bc)

    def _pressure_dirichlet_point_bc(self):
        ''' Prepare pressure point Dirichlet boundary condition.
        Set zero automatically.
        '''
        pt = self.options['fem']['fix_pressure_point']
        if len(pt) != self.mesh.topology.dim:
            raise Exception('Dimension of pressure BC point coordinates != '
                            'mesh dimension.')

        pt_arr = np.array(pt, dtype=float)
        tol = 1e-10
        if len(pt) == 2:
            def _at_point(x):
                return np.isclose(x[0], pt_arr[0], atol=tol) & \
                       np.isclose(x[1], pt_arr[1], atol=tol)
        else:
            def _at_point(x):
                return np.isclose(x[0], pt_arr[0], atol=tol) & \
                       np.isclose(x[1], pt_arr[1], atol=tol) & \
                       np.isclose(x[2], pt_arr[2], atol=tol)

        dofs = locate_dofs_geometrical(self.Q, _at_point)
        bc = dirichletbc(self._C(0.0), dofs, self.Q)
        self.logger.info('Setting pressure point BC')

        self.bc_dict['p']['dirichlet'].append(bc)

# TODO: Future work will be on unifying both problem classes
class ProblemCoupled(Problem):
    ''' NavierStokes problem with coupled velocity components. '''

    def __init__(self, inputfile=None):
        ''' Initialize.

        Args:
            inputfile (str):     path to YAML file
        '''
        super().__init__(inputfile)

    def create_functionspaces(self):
        r''' Create function spaces for velocity and pressure, namely:

        * D:   VectorFunctionSpace for the displacement
        * V:   VectorFunctionSpace for the velocity
        * Q:   FunctionSpace for the pressure

        and initialize functions :math:`d \in D`, :math:`u \in V`, :math:`p \in Q`.

        The elements are specified via :code:`ale: fem: displacement_space`,
        :code:`fem: velocity_space` and :code:`pressure_space`, where possible options are:

        *   displacement:   p1, p2
        *   velocity:       p1, p2, p1b/p1+ (bubble enriched)
        *   pressure:       p1, p0/dg0, p1-/dg1
        '''
        cell = self.mesh.basix_cell()

        if hasattr(self, 'mesh_ext'):
            s_space = self.ale['io']['fem_type'].lower()

            self.logger.info('Creating external displacement space: {}'
                            .format(s_space.capitalize()))

            deg = int(s_space[1])
            self.S = functionspace(self.mesh_ext, ("Lagrange", deg, (self.ndim,)))
            self.d_s = Function(self.S)
            self.v_s = Function(self.S)

        d_space = self.ale['fem']['displacement_space'].lower()
        u_space = self.options['fem']['velocity_space'].lower()
        p_space = self.options['fem']['pressure_space'].lower()

        if self._using_ale:
            self.logger.info('Creating displacement space: {}'.format(
                d_space.capitalize()))
            if d_space in ('p1', 'p2'):
                deg = int(d_space[1])
                self.D  = functionspace(self.mesh, ("Lagrange", deg, (self.ndim,)))
                self.DG = functionspace(self.mesh, ("DG", max(deg - 1, 0)))

            self.logger.info('Number of displacement DOFs: {}'.format(
                _dof_count(self.D)))
            self.d = Function(self.D, name='d')
            self.d0 = Function(self.D, name='d')

            self.F, self.J = self.F_(self.d), self.J_(self.d)
            self.F0, self.J0 = self.F_(self.d0), self.J_(self.d0)
        else:
            self.F, self.J = Identity(self.ndim), self._C(1.)
            self.F0, self.J0 = Identity(self.ndim), self._C(1.)

        self.logger.info('Creating velocity space: {}'.format(
            u_space.capitalize()))
        if u_space in ('p1', 'p2'):
            deg = int(u_space[1])
            self.V = functionspace(self.mesh, ("Lagrange", deg, (self.ndim,)))
        elif u_space in ('p1b', 'p1+'):
            deg = int(u_space[1])
            P1_v = bufl.element("Lagrange", cell, deg, shape=(self.ndim,))
            B_v  = bufl.element("Bubble",   cell, deg + self.ndim, shape=(self.ndim,))
            self.V = functionspace(self.mesh, bufl.enriched_element([P1_v, B_v]))

        self.logger.info('Creating pressure space: {}'.format(
            p_space.upper()))
        if p_space == 'p1':
            self.Q = functionspace(self.mesh, ("Lagrange", 1))
        elif p_space in ('p0', 'dg0'):
            self.Q = functionspace(self.mesh, ("DG", 0))
        elif p_space in ('p1-', 'dg1'):
            self.Q = functionspace(self.mesh, ("DG", 1))

        self.logger.info('Number of velocity DOFs: {}'.format(_dof_count(self.V)))
        self.logger.info('Number of pressure DOFs: {}'.format(_dof_count(self.Q)))

        self.u = Function(self.V, name='u')
        self.u0 = Function(self.V, name='u0')
        self.u0_mapdd = Function(self.V, name='u0_mapdd')
        self.p = Function(self.Q, name='p')

    def form_velocity_tentative(self):
        ''' Definition of forms of tentative velocity step.  '''
        rho = self.rho
        mu = self.mu
        k = self.k
        F, J = self.F, self.J
        F0, J0 = self.F0, self.J0

        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        p = TrialFunction(self.Q)

        def diff(u):
            # NOTE: The variationals forms are taken with $\star = n$ 
            # Classic formulation:
            #   inner(2*mu*grad(u), grad(v))*dx
            if self.options['fem']['strain_symmetric']:
                # Classic formulation:
                #   inner(2*mu*grad(u), grad(v))*dx
                return J*inner(2*mu*sym(dot(grad(u), inv(F)), 
                            sym(dot(grad(v), inv(F)))))*dx
            else:
                # Classic formulation:
                #   inner(mu*grad(u), grad(v))*dx
                return J*inner(mu*dot(grad(u), inv(F)), 
                            dot(grad(v), inv(F)))*dx

        def conv(u, u_conv):
            # NOTE: ALE formulation uses GCL stabilization
            # Classic formulation:
            #   rho*dot(grad(u)*u_conv, v)*dx
            a_conv = (rho*J*dot(dot(grad(u)*inv(F), u_conv - w_conv), v)*dx
                + 0.5*rho*div(J*inv(F)*w_conv)*inner(u, v)*dx 
                + rho*k*(J - J0)*inner(u, v)*dx)
            if self.options['fem']['convection_skew_symmetric']:
                # Classic formulation:
                #   0.5*rho*div(u_conv)*dot(u, v)*dx
                a_conv += 0.5*rho*div(J*dot(inv(F), u_conv))*dot(u, v)*dx

            return a_conv

        if self.options['timemarching']['fractionalstep']['scheme'] == 'IPCS':
            # BDF2 coefficients
            # u0_vec = as_vector(self.u0_lst)
            u_conv = 2.0*self.u - self.u0
            self.u_conv_assigned = Function(self.V)
        else:
            u_conv = self.u
            # TODO this still needed?
            self.u_conv_assigned = self.u

        if self._using_ale:
            # NOTE: $\circ\circ = n$ to obtain energy stable solutions 
            # check: https://doi.org/10.1016/j.aml.2020.106830
            d, d0 = self.d, self.d0
            w_conv = k*(d  - d0)

        else:
            w_conv = Function(self.V)
            
        a_mass = rho*J0*k*dot(u, v)*dx
        a_pres = -p*J0*inner(grad(v), inv(F0))*dx
        a_diff = diff(u)
        a_conv = conv(u, u_conv)

        self.forms['u'].update({'mass': a_mass,
                                'diff': a_diff,
                                'conv': a_conv})

        if self.options['timemarching']['fractionalstep']['scheme'] == 'CT':
            self.forms['u'].update({'pres': None})
        else:
            self.forms['u'].update({'pres': a_pres})
            # TODO: HANDLE THIS
        
        if self._using_mapdd:
            self.forms['u'].update({'pres': a_pres})

        self.forms['u'].update({
            'neumann': sum(self.bc_dict['u']['neumann']),
            'navierslip': self.bc_dict['u']['navierslip'],
            'transpiration': self.bc_dict['u']['transpiration'],
            'inflow_rhs': self.bc_dict['u']['inflow_rhs'] ,
            'inflow_lhs': self.bc_dict['u']['inflow_lhs'] ,
            'mapdd_rhs': sum(self.bc_dict['u']['mapdd_rhs']),
        })


        if self.bc_dict['u']['mapdd_lhs']:
            self.forms['u']['conv'] += sum(self.bc_dict['u']['mapdd_lhs'])

        forms_stab = self.stabilization(u_conv)

        # TODO maybe do this in self.stabilization()?
        if 'bfs' in forms_stab:
            self.forms['u']['conv'] += forms_stab['bfs']
        if 'fnv' in forms_stab:
            self.forms['u']['conv'] += forms_stab['fnv']
        if 'supg_convdiff' in forms_stab:
            self.forms['u']['conv'] += forms_stab['supg_convdiff']
        if 'supg_time' in forms_stab:
            self.forms['u'].update({'supg_time': forms_stab['supg_time']})
        # if 'supg_gradp' in forms_stab:
        #     self.forms['u'].update({'supg_gradp': forms_stab['supg_gradp']})
            
    def form_velocity_update(self):
        ''' Definition of forms of velocity update. '''
        p = TrialFunction(self.Q)
        v = TestFunction(self.V)

        F0, J0 = self.F0, self.J0
        # FIXME: which one to use: J or J0 in gradp ?
        # Classic formulation:
        #   -dot(grad(p), v)*dx
        a_gradp = -J0*dot(inv(F0)*grad(p), v)*dx

        # mass matrices already defined in form_tentative_velocity()
        self.forms['u'].update({'gradp': a_gradp})

    def form_pressure(self):
        ''' Definition of forms of pressure projection step. '''
        k = self.k
        rho = self.rho
        F0, J0 = self.F0, self.J0

        p = TrialFunction(self.Q)
        q = TestFunction(self.Q)
        u = TrialFunction(self.V)

        # Classic formulation:
        #   a_lap = inner(grad(p), grad(q))*dx
        #   a_divu = -k*rho*div(u)*q*dx
        a_lap = J0*inner(dot(grad(p), inv(F0)),
                    dot(grad(q), inv(F0)))*dx
        a_divu = -k*rho*div(J0*dot(inv(F0), u))*q*dx

        self.forms['p'].update({
            'laplacian': a_lap,
            'rhs_u': a_divu + sum(self.bc_dict['p']['robin']['forms_u']),
            'neumann': sum(self.bc_dict['p']['neumann']),
            'robin': self.bc_dict['p']['robin']['forms_p'],
            'transpiration_dirichlet_u': self.bc_dict['p']['transpiration']
            ['dirichlet_forms_u'],
            'transpiration_dirichlet_p': self.bc_dict['p']['transpiration']
            ['dirichlet_forms_p'],
        })


        if self._using_mapdd:
            #n = FacetNormal(self.mesh)
            #N = J*inv(F).T*n
            #t_p = dot(grad(p), inv(F)) - dot(dot(grad(p), inv(F)),N)*N
            #t_q = dot(grad(q), inv(F)) - dot(dot(grad(q), inv(F)),N)*N
            
            for bid, prm in self.bc_dict['p']['mapdd']['params'].items():
                #const = self._C(prm['eps_gradp'])
                #self.forms['p']['laplacian'] += const*inner(t_p, t_q)*ds(bid)
                self.forms['p']['laplacian'] += (1/prm['l'])*p*q*self.ds(bid)
            
    def forced_normal(self, uprev = None):
        ''' Forms of forced normal velocities using a semi-implicit implementation as:
                
                Ai = gamma*(ui - {sum_[i!=j](u0j*nj) + ui*ni }*ni )*vi

            In this way, a coupling between the components is avoided since in the dot() product between u and n,
            only the "solving" i-component is used implicitly, while the rest are taken from the velocity of the
            previous time-step u0.

        Args:
            uprev:      Velocity of the previous time-step used in the dot product

        Returns:
            forms dict: forced normal form
        '''

        term_dict = self.options['fem']['stabilization']['forced_normal']
        gamma = self._C(term_dict['gamma'])
        bind_lst = (term_dict['boundaries'])
        self.logger.info('Adding normal forced velocity at '
                         'boundary id {}'.format(bind_lst))

        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        F, J = self.F, self.J
        n = FacetNormal(self.mesh)
        # Using the reference system ds = dS J |inv(F).T N|
        #elem = sqrt(dot(J*inv(F).T*n, J*inv(F).T*n))

        ut = u - dot(u,n)*n
        vt = v - dot(v,n)*n
        a_fnv = sum([gamma*dot(ut,vt)*self.ds(i) for i in bind_lst])

        return {'fnv': a_fnv}

    def stab_backflow(self, u_conv=None):
        ''' Backflow stabilization.

        Args:
            u_conv:      convecting velocity, for example Adams-Bashforth
                        interpolated for BDF2
        '''
        def abs_n(x):
            return 0.5*(x - abs(x))

        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        n = FacetNormal(self.mesh)

        ind = self.options['fem']['stabilization']['backflow_boundaries']
        self.logger.info('adding backflow stabilization on boundaries {}'.
                         format(ind))
        ds = ufl.Measure('ds', domain=self.mesh, subdomain_data=self.facet_tags)
        k, rho = self.k, self.rho
        F, J = self.F, self.J

        if self._using_ale:
            d, d0 = self.d, self.d0
            w_conv = k*(d - d0)
        else:
            w_conv = Function(self.V)
        # Classic formulation:
        #   sum([-0.5*rho*abs_n(dot(u_conv, n))*dot(u, v)*ds(i) for i in ind])
        a_bfs = sum([-0.5*rho*abs_n(J*dot(inv(F)*(u_conv - w_conv), n))*dot(u, v)*ds(i) for i in ind])

        return {'bfs': a_bfs}

    def streamline_diffusion(self, u_conv):
        ''' Streamline Diffusion stabilization.
        There are different definitions for the stabilization parameter tau if
        'length_scale' is 'metric' and otherwise (average or max).

        See Shakib and Hughes (1991), "A new finite element formulation for
        computational fluid dynamics" X and IX

        Args:
            u_conv:      convecting velocity, for example Adams-Bashforth
                        interpolated for BDF2
        '''
        opt = self.options['fem']['stabilization']['streamline_diffusion']

        mu = self.mu
        rho = self.rho
        k = self.k

        sd = SDParameter(self.options, self.mesh, mu, rho, k,
                         self._logging_filehandler)
        tau = sd.stabilization_parameter(u_conv, self.u_conv_assigned)

        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        residual_convdiff = rho*grad(u)*u_conv

        if 'consistent' in opt and opt['consistent']:
            raise Exception('Consistent SUPG not implemented because dp/dxi'
                            ' needs to be reassembled for each component.')
            res_str = 'consistent (w/o pressure)'
            residual_time = k*rho*u
            # residual_gradp = [self.p.dx(i) for i in range(self.ndim)]
            if self.V.ufl_element.degree > 1:
                residual_convdiff += -mu*div(grad(u))
            a_supg_time = tau*rho*dot(grad(v)*u_conv, residual_time)*dx
        else:
            res_str = 'minimal'
            a_supg_time = None

        self.logger.info('SD/SUPG residual: {r}'.format(r=res_str))

        a_supg_convdiff = tau*rho*dot(grad(v)*u_conv, residual_convdiff)*dx

        return {'supg_convdiff': a_supg_convdiff, 'supg_time': a_supg_time}

    def boundary_conditions(self):
        # bc_lst = copy.deepcopy(self.options['boundary_conditions'])
        # self.bc_lst = bc_lst
        BC = BoundaryConditionsCoupled(self)
        BC.process_bcs()
        self.bc_dict = BC.bc_dict
        self.ds = BC.ds

        self._using_wk = BC._using_wk
        self.wk = BC.wk
        self._using_mapdd = BC._using_mapdd

        # TODO: clean boundary conditions inheritance!
        if self._using_ale:
            self.disp_bc = BC.disp_bc
            self.vel_bc = BC.vel_bc
            self.vel_bc_lst = BC.vel_bc_lst


class BoundaryConditionsCoupled(BoundaryConditions):
    ''' Boundary Conditions class for ProblemCoupled '''

    def __init__(self, problem):
        ''' Constructor.

        Args:
            options     options dictionary
            V           velocity FunctionSpace (vector if coupled components,
                        scalar if split components are used)
            Q           pressure FunctionSpace
            logging_filehandler (optional)      filehandler of log file
                                                (run.log)
        '''
        self.init_logging()
        self._logging_filehandler = problem._logging_filehandler
        if self._logging_filehandler:
            self.logger.addHandler(self._logging_filehandler)

        self.logger.info('Processing boundary conditions')
        self.bcs = problem.options['boundary_conditions']
        self.options = problem.options
        self.V = problem.V
        self.Q = problem.Q
        self.k = problem.k
        self.u0_mapdd = problem.u0_mapdd
        self.rho = problem.rho
        self.mu = problem.mu
        self.bnds = problem.bnds
        self.mesh = problem.mesh
        self.facet_tags = problem.facet_tags
        self.ndim = self.mesh.topology.dim
        self.ds = ufl.Measure('ds', domain=self.mesh, subdomain_data=self.facet_tags)

        self._using_wk = False
        self._using_mapdd = False

        self.wk = {
                'implicit': False,
                'LRC': False,
                }

        for bc in self.options['boundary_conditions']:
            if 'type' in bc.keys():
                if bc['type'] == 'windkessel':
                    self._using_wk = True

        if self._using_wk:
            self.wk['implicit'] = self.options['windkessel']['implicit']
            if 'low_rank_update' in self.options['windkessel']:
                self.wk['LRC'] = self.options['windkessel']['low_rank_update']

        if self._using_wk:
            raise Exception('Windkessel BC its not implemented on coupled problem yet!')

        self.bc_dict = {
            'u': {
                'dirichlet': [],
                'neumann': [],
                'navierslip': dict([('forms', []), ('coef', []), ('id', [])]),
                'transpiration': dict([('forms', []), ('coef', []),
                                       ('id', [])]),
                'inflow_rhs': None,
                'inflow_lhs': None,
                'mapdd_lhs': [],
                'mapdd_rhs': [],
                'dbc_expressions': {},
                # # store time dependent BCs for time-updates during time
                # # stepping
                # 'time_bcs': [],
                # same_dbc_boundaries is used in
                # Solver.solve_tentative_velocity() in order to check if DBCs
                # have to be re-applied and the solver/pc set up for each
                # component
                'same_dbc_boundaries': True,
            },
            'p': {
                'dirichlet': [],
                'neumann': [],
                'mapdd' : {
                    'params': {},
                    'dirichlet': [],
                    'neumann': [],
                    'dbc_params': {},
                },
                'windkessel': {
                    'dirichlet': [],
                    'id': [],
                    'ds': [],
                    'normal': [],
                    'Qspace': [],
                    'R_p': {},
                    'C': {},
                    'R_d': {},
                    'pii0': {},
                    'pii': {},
                    'alpha': {},
                    'beta': {},
                    'gamma': {}
                },
                # list entrys of Robin BCs are forms_p, forms_u dictionaries
                'robin': {
                    'forms_u': [],
                    'forms_p': [],
                },
                'transpiration': {
                    'coef': [],
                    'dirichlet_functions': [],
                    'dirichlet_forms_u': [],
                    'dirichlet_forms_p': [],
                }
            },
        }

        self.ale = problem.ale
        self._using_ale = problem._using_ale
        self.F, self.J = problem.F, problem.J

        if self._using_ale:
            self.D = problem.D
            # NOTE: disp_bc and vel_bc allocate external bcs.
            self.disp_bc = None
            self.vel_bc = None
            self.vel_bc_lst = [None for i in range(self.ndim)]

            self.bc_dict.update({
                'd':{
                    'dirichlet': [],
                    'neumann': [],
                    'dbc_expressions': {}
                }
            })

    def _dirichlet_velocity(self, bc):
        ''' Create velocity Dirichlet boundary condition from options and add
        into self.bc_dict['u']['dirichlet'].

        Args:
            bc (dict):  dict describing one boundary condition
        '''
        if self.ale['type'] in ('default', 'manual'):
            if not ('id' in bc and 'value' in bc):
                raise KeyError('bc dict needs keys id & value')
        else:
            if not ('id' in bc):
                raise KeyError('bc dict needs key id')

        #
        # The velocity components are handled separately, regardless if the
        # velocity space is treated component-wise or not.
        # bc['value'] is a list of list 'ndim', containing a boundary
        # condition for each component, possibly 'None'. These values are cast
        # into a format compatible with DirichletBC().
        # self.bc_dict['u']['dirichlet'][i] (i in range(ndim)) is a list for
        # each velocity component, that holds all corresponding Dirichlet BCs
        # definined for component i.
        # This format is convenient for applying the BCs in
        # Solver.solve_tentative_velocity()
        #
        if self.ale['type'] in ('default', 'manual'):
            val = bc['value']
            if not len(val) == self.ndim:
                raise Exception('Dimension of Dirichlet BC does not match geometry'
                                ': {} != {}'.format(len(val), self.ndim))

            expr = None

            if None not in val:
                if all(isinstance(x, (int, float)) for x in val):
                    val = self._C(val)

                elif any(isinstance(x, str) for x in val):
                    params = bc.get('parameters', {})
                    resolved = []
                    for x in val:
                        if isinstance(x, (int, float)):
                            resolved.append(float(x))
                        elif isinstance(x, str):
                            try:
                                resolved.append(float(x))
                            except ValueError:
                                if x in params:
                                    resolved.append(float(params[x]))
                                else:
                                    self.logger.warning(
                                        'String BC value "{}" at bid={} has no matching '
                                        'parameter — BC skipped.'.format(x, bc['id']))
                                    return
                    val = self._C(resolved)

                elif isinstance(val, fem.Function):
                    expr = val

                elif isinstance(val, fem.Constant):
                    pass

                else:
                    raise Exception('Inconsistent Dirichlet value types at '
                                    'boundary {}'.format(bc['id']))

                if utils.is_enriched(self.V):
                    V_collapsed, _ = self.V.collapse()
                    val = _project(val, V_collapsed)

                facets = self.facet_tags.find(bc['id'])
                dofs = locate_dofs_topological(self.V, self.mesh.topology.dim - 1, facets)
                dbc = dirichletbc(val, dofs, self.V)
                self.bc_dict['u']['dirichlet'].append(dbc)

                if expr:
                    bc_key = bc['id']
                    self.bc_dict['u']['dbc_expressions'][bc_key] = {
                        'expression': expr, 'id': bc['id']}

            else:
                for i, val in enumerate(bc['value']):
                    if val is None:
                        self.bc_dict['u']['same_dbc_boundaries'] = False
                        continue

                    elif isinstance(val, (int, float)):
                        val = self._C(val)

                    elif isinstance(val, str):
                        params = bc.get('parameters', {})
                        if val in params:
                            val = self._C(float(params[val]))
                        else:
                            self.logger.warning(
                                'String BC value "{}" at bid={}, i={} has no matching '
                                'parameter — component BC skipped.'.format(
                                    val, bc['id'], i))
                            continue

                    elif isinstance(val, fem.Function):
                        expr = val

                    elif isinstance(val, fem.Constant):
                        pass

                    facets = self.facet_tags.find(bc['id'])
                    V_sub, _ = self.V.sub(i).collapse()
                    if utils.is_enriched(V_sub):
                        val = _project(val, V_sub)

                    dofs = locate_dofs_topological(
                        (self.V.sub(i), V_sub), self.mesh.topology.dim - 1, facets)
                    dbc = dirichletbc(val, dofs, self.V.sub(i))
                    self.bc_dict['u']['dirichlet'].append(dbc)

                    if expr:
                        bc_key = (bc['id'], i)
                        self.bc_dict['u']['dbc_expressions'][bc_key] = {
                            'expression': expr, 'id': bc['id']}

        elif self.ale['type'] == 'external':
            self.vel_bc = Function(self.V)
            facets = self.facet_tags.find(bc['id'])

            for i in range(self.ndim):
                Vi_sub, _ = self.V.sub(i).collapse()
                self.vel_bc_lst[i] = Function(Vi_sub)

                dofs = locate_dofs_topological(
                    (self.V.sub(i), Vi_sub), self.mesh.topology.dim - 1, facets)
                dbc = dirichletbc(self.vel_bc_lst[i], dofs, self.V.sub(i))
                self.bc_dict['u']['dirichlet'].append(dbc)

    def _neumann_velocity(self, bc):
        ''' Create weak form of Neumann boundary condition '''
        v = TestFunction(self.V)
        n = FacetNormal(self.mesh)
        val = self._C(bc['value'])
        val_ = val*self.J*inv(self.F).T*n
        # Classic formulation:
        #   val*dot(n, v)*self.ds(bc['id'])
        a_bc = dot(val_, v)*self.ds(bc['id'])
        self.bc_dict['u']['neumann'].append(a_bc)

    def _inflow_profile(self, bc):
        ''' Create Inflow Profile BC

        Args:
            bc (dict):  dict describing the boundary condition
        '''
        # TODO Add to documentation and exceptions to handle 
        # bdry options such as the profile/parameters/waveform, etc!
        if not 'profile' in bc:
            raise KeyError('profile option not found. '
                           'It has to specify the path to HDF5 file')

        gamma = bc['gamma']
        F, J = self.F, self.J

        # Reading the profile
        reading_csv = False
        n = FacetNormal(self.mesh)
        uprofile = Function(self.V)
        inout.read_HDF5_data(self.mesh.comm, bc['profile'], uprofile, 'u')
        elem = J*dot(inv(F).T*n, inv(F).T*n)
        area = self.mesh.comm.allreduce(
    assemble_scalar(fem_form(elem*self.ds(bc['id']))), op=MPI.SUM)
        Norm_fact = abs(self.mesh.comm.allreduce(
    assemble_scalar(fem_form(dot(uprofile,n)*self.ds(bc['id']))), op=MPI.SUM)/area)
        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        
        self.bc_dict['u'].update({'inflow_lhs': None } )

        if '.csv' in bc['waveform']:
            reading_csv = True
            self.logger.info('taking inflow form csv file...')
            flow_init = self.mesh.comm.allreduce(
    assemble_scalar(fem_form(dot(uprofile,n)*self.ds(bc['id']))), op=MPI.SUM)
            flip = -1 if flow_init >0 else 1
            Norm_fact = flow_init*flip            
            time_data = []
            flow_data = []
            with open(bc['waveform']) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for row in csv_reader:
                    time_data.append(float(row[0]))
                    flow_data.append(float(row[1]))
            
            inflow_func = interp1d(time_data,flow_data, kind='cubic', fill_value='extrapolate')
            waveform = self._C(inflow_func(0.0))
        else:
            elem = J*dot(inv(F).T*n, inv(F).T*n) 
            area = self.mesh.comm.allreduce(
    assemble_scalar(fem_form(elem*self.ds(bc['id']))), op=MPI.SUM)
            Norm_fact = abs(self.mesh.comm.allreduce(
    assemble_scalar(fem_form(dot(uprofile,n)*self.ds(bc['id']))), op=MPI.SUM)/area)
            params = bc['parameters']
            # TODO: string-based waveform Expression not yet ported;
            # use a fem.Constant updated by solver via waveform_func each step
            waveform = self._C(0.0)
            self.logger.warning(
                'String-based waveform Expression not yet ported; '
                'waveform initialised to zero.')


        self.bc_dict['u']['dbc_expressions']['inflow'] = {'expression': waveform, 'id': bc['id']}
        if reading_csv:
            self.bc_dict['u']['dbc_expressions']['inflow'].update({'inflow_func': inflow_func} )

        self.bc_dict['u']['inflow_rhs'] = gamma*(1/Norm_fact)*waveform*dot(uprofile,v)*self.ds(bc['id'])
        # The left hand side can be written only once for all the components
        self.bc_dict['u']['inflow_lhs'] = gamma*dot(u,v)*self.ds(bc['id'])

    def _mapdd(self,bc):
        '''
            method of asymptotic partial decomposition of a domain (MAPDD) proposed
            in Bertoglio et al (2019).

        '''
        self._using_mapdd = True
    
        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        n = FacetNormal(self.mesh)

        l = bc['parameters']['l']
        rho = self.rho
        mu = self.mu
        k = self.k
        bid = bc['id']

        self.bc_dict['p']['mapdd']['params'][bid] = {
            'fmass': self._C(rho*k),
            'fdiff': self._C(mu),
            'l': self._C(l),
        }

        prm = self.bc_dict['p']['mapdd']['params'][bid]
        a_rhs = prm['l']*rho*k*dot(self.u0_mapdd,v)*self.ds(bc['id'])
        self.bc_dict['u']['mapdd_rhs'].append(a_rhs)

        # tangential component
        un = dot(u,n)
        vn = dot(v,n)
        gradtan_un = grad(un) - dot(grad(un),n)*n
        gradtan_vn = grad(vn) - dot(grad(vn),n)*n
        a_lhs = 0
        a_lhs += prm['l']*mu*dot(gradtan_un, gradtan_vn)*self.ds(bc['id'])
        a_lhs += prm['l']*rho*k*dot(u,v)*self.ds(bc['id'])
        self.bc_dict['u']['mapdd_lhs'].append(a_lhs)

    def _navierslip_velocity(self, bc):
        ''' Create weak forms of Navier-slip boundary condition, for each
        component, i,

            val\int_{\Gamma_i} u_i v_i (1- n_i^2) - v_i n_i (\Sigma_{j=1, j\neq
            i} u_j n_j) ds

        For each i, creates the "diagonal" term (u_i, v_i) and the ndim-1 cross
        terms. The resulting forms are stored in a (diag, cross)-dict within
        bc_dict, need to be assembled into matrices (even in the explicit case)
        and multiplied by the corresponding velocity component vectors.

        Note that the coefficients are not included in the integrals. The
        assembled matrices have to be multiplied by the (possibly) varying
        coefficient.

        Args:
            bc (dict)       bc dict
        '''
        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        n = FacetNormal(self.mesh)
        val = self._C(bc['value'])

        self.bc_dict['u']['navierslip']['coef'].append(val)
        self.bc_dict['u']['navierslip']['id'].append(bc['id'])
        method = (self.options['timemarching']['fractionalstep']
                  ['robin_bc_velocity_scheme'])

        if method == 'implicit':
            a = (dot(u, v) - dot(u, n)*dot(v, n))*self.ds(bc['id'])
        else:
            raise NotImplementedError('Fully coupled FS only works with '
                                      'implicit Robin BC')

        self.bc_dict['u']['navierslip']['forms'].append({'implicit': a})

    def _transpiration_velocity(self, bc):
        ''' Create weak forms of the transpiration boundary condition, for each
        component, i,

            \int_{\Gamma_i} u_i v_i n_i^2 - v_i n_i (\Sigma_{j=1, j\neq i}
                u_j n_j) ds

        For each i, creates the "diagonal" term (u_i, v_i) and the ndim-1 cross
        terms. The resulting forms are stored in a (diag, cross)-dict within
        bc_dict, need to be assembled into matrices (even in the explicit case)
        and multiplied by the corresponding velocity component vectors.

        Note that the coefficients are not included in the integrals. The
        assembled matrices have to be multiplied by the (possibly) varying
        coefficient.

        Args:
            bc (dict)       bc dict
        '''
        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        p = TrialFunction(self.Q)
        n = FacetNormal(self.mesh)
        val = self._C(bc['value'])
        self.bc_dict['u']['transpiration']['coef'].append(val)
        self.bc_dict['u']['transpiration']['id'].append(bc['id'])
        method = (self.options['timemarching']['fractionalstep']
                  ['robin_bc_velocity_scheme'])

        if method == 'implicit':
            a = dot(u, n)*dot(v, n)*self.ds(bc['id'])
        else:
            raise NotImplementedError('Fully coupled FS only works with '
                                      'implicit Robin BC')

        a_pressure = -p*dot(v, n)*self.ds(bc['id'])

        self.bc_dict['u']['transpiration']['forms'].append(
            {'implicit': a, 'pressure': a_pressure}
        )

    def _preset_parabola_inlet_bc(self, bc):
        ''' Process preset inlet boundary condition, create format that
        function dirichlet understands.

        Args:
            bc      dictionary describing one boundary condition
        '''
        if 'method' in bc and bc['method'] == 'nitsche':
            self.logger.warn('Inlet BCs are imposed strongly. Ignoring '
                             'Nitsche setting.')

        if not isinstance(bc['value'], fem.Function):
            flow_direction = bc['flow_direction'] if 'flow_direction' in bc \
                else 0
            assert flow_direction in range(self.ndim), (
                'flow direction {} does not match problem dimension'
                .format(flow_direction))

            if 'bnd_interval' in bc['value'] and bc['value']['bnd_interval']:
                bnd_interval = bc['value']['bnd_interval']
                r0 = 0.5*(bnd_interval[1] - bnd_interval[0])
                x0 = 0.5*(bnd_interval[1] + bnd_interval[0])
                if 'R' in bc['value'] and bc['value']['R']:
                    self.logger.warning('BC {}: bnd_interval was set, ignoring'
                                        ' R'.format(bc['id']))
            else:
                r0 = bc['value']['R']
                x0 = 0.

            # get the indices orthogonal to flow direction (coordinates of
            # parabola)
            indices = list(range(self.ndim))
            indices.remove(flow_direction)

            if self.ndim == 2:
                inflow_str = ('U*(1 - pow(x[{0}] - x0, 2)/(R*R))'
                              .format(*indices))
            elif self.ndim == 3:
                self.logger.warn('3D paraboloidal inlet profile only valid for'
                                 ' circular cross sections')
                inflow_str = ('U/(R*R)*(R*R - pow(x[{0}] - x0, 2) - '
                              'pow(x[{1}] - x0, 2))'.format(*indices))

            U_val  = float(bc['value']['U'])
            x0_val = float(x0)
            r0_val = float(r0)
            idx    = indices

            inflow = Function(self.V)
            def _parabola_vec(x, _U=U_val, _x0=x0_val, _r=r0_val,
                              _fd=flow_direction, _idx=idx, _ndim=self.ndim):
                vals = np.zeros((_ndim, x.shape[1]))
                if _ndim == 2:
                    vals[_fd] = _U * (1.0 - (x[_idx[0]] - _x0)**2 / _r**2)
                else:
                    vals[_fd] = _U / _r**2 * (
                        _r**2 - (x[_idx[0]] - _x0)**2 - (x[_idx[1]] - _x0)**2)
                return vals
            inflow.interpolate(_parabola_vec)
            bc['value'] = inflow

        self._dirichlet_velocity(bc)

    def _preset_sine_parabola_inlet_bc(self, bc):
        ''' Process preset sine oscillation inlet boundary condition, create
        format that function dirichlet_velocity() understands.

        Args:
            bc      dictionary describing one boundary condition
        '''
        if 'method' in bc and bc['method'] == 'nitsche':
            self.logger.warn('Inlet BCs are imposed strongly. Ignoring '
                             'Nitsche setting.')

        if not isinstance(bc['value'], fem.Function):
            flow_direction = bc['flow_direction'] if 'flow_direction' in bc \
                else 0
            assert flow_direction in range(self.ndim), (
                'flow direction {} does not match problem dimension'
                .format(flow_direction))

            if 'bnd_interval' in bc['value'] and bc['value']['bnd_interval']:
                bnd_interval = bc['value']['bnd_interval']
                r0 = 0.5*(bnd_interval[1] - bnd_interval[0])
                x0 = 0.5*(bnd_interval[1] + bnd_interval[0])
                if 'R' in bc['value'] and bc['value']['R']:
                    self.logger.warning('BC {}: bnd_interval was set, ignoring'
                                        ' R'.format(bc['id']))
            else:
                r0 = bc['value']['R']
                x0 = 0.

            indices = list(range(self.ndim))
            indices.remove(flow_direction)

            U_val  = float(bc['value']['U'])
            x0_val = float(x0)
            r0_val = float(r0)
            a_val  = float(bc['value']['a'])
            idx    = indices
            time_factor = self._C(0.0)

            spatial = Function(self.V)
            def _parabola_vec(x, _U=U_val, _x0=x0_val, _r=r0_val,
                              _fd=flow_direction, _idx=idx, _ndim=self.ndim):
                vals = np.zeros((_ndim, x.shape[1]))
                if _ndim == 2:
                    vals[_fd] = _U * (1.0 - (x[_idx[0]] - _x0)**2 / _r**2)
                else:
                    vals[_fd] = _U / _r**2 * (
                        _r**2 - (x[_idx[0]] - _x0)**2 - (x[_idx[1]] - _x0)**2)
                return vals
            spatial.interpolate(_parabola_vec)

            # time_factor updated each step: time_factor.value = sin(a*pi*t)
            bc['time_factor'] = time_factor
            bc['a'] = a_val
            bc['value'] = time_factor * spatial

        # self.bc_dict['u']['time_bcs'].append(
        #     {'expression': bc['value'], 'id': bc['id']})
        # save id, i for enriched elements

        self._dirichlet_velocity(bc)


def problem(inputfile):
    opt = inout.read_parameters(inputfile)['timemarching']['fractionalstep']
    if 'coupled_velocity' in opt and opt['coupled_velocity']:
        return ProblemCoupled(inputfile)
    else:
        return Problem(inputfile)
