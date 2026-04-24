''' Monolithic Navier-Stokes problem module

Author: David Nolte (dnolte@dim.uchile.cl)
Date:   2018-03-19
'''
from dolfin import *
import dolfin
import ufl
from common import geom, inout, utils
from pathlib import Path
import numpy as np
import copy
from abc import ABC, abstractmethod
from ..logger.logger import LoggerBase


if not dolfin.__version__ >= '2018':
    class MPI(MPI):
        comm_world = mpi_comm_world()


class ProblemBase(LoggerBase, ABC):
    ''' Class that sets up the Navier-Stokes problem to be solved.
    '''
    # TODO: simplify boundary conditions. Use-cases are basically known, assume
    #   that only the 'preset' BCs will be used in the feature and remove the
    #   switch!

    def __init__(self, inputfile=None):
        ''' Initialize Navier-Stokes problem. Initalizes all instance variables
        with None and loads parameter file.

        Args:
            inputfile (str):     path to YAML input file

        Attributes:
            self.options:   options dictionary
            self.nls_form:  list variational forms of residual F and jacobian J
                            of the nonlinear problem
            self.ls_form:   list variational forms of residual F and jacobian J
                            of the bilinear (linearized) problem
            self.qnls_form: ls_form + newton linearization terms (qnewton)
            self.bcs:       list of boundary conditions
            self.W:         function space
            self.w:         solution Function(W)
            self.bnds       boundary MeshFunction
        '''
        self.check_version()
        super().__init__()

        self.options = None
        self.inputfile = inputfile
        if inputfile:
            self.get_parameters(inputfile)

        self.setup_logger()
        self.logger.info('Initializing')
        self.logger.info('Number of parallel tasks: {}'.format(
            MPI.size(MPI.comm_world)))
        self.logger.info('Write out path: {}'.format(
            self.options['io']['write_path']))

        # mesh, boundaries
        self.mesh = None
        self.bnds = None

        self.dt = 0

        # needed for nsestimator
        self.bc_lst = None

        # linear form
        self.ls_form = None
        # nonlinear form
        self.nls_form = None
        # "quasi" nonlinear form (TODO deprecate!)
        self.qnls_form = None

        self.bcs = []
        self.bcs_neumann = []
        self.bcs_nitsche = []
        self.bcs_navierslip = []
        self.bcs_transpiration = []

        self.w = None
        self.W = None

        self._bid_outlet = None

    @abstractmethod
    def variational_form(self):
        ''' Define variational forms of the problem.
        Store nonlinear forms in self.nls_form (F, J), and bilinear forms in
        self.ls_form (F, J). self.qnls_form is used for quasi-Newton methods.
        '''
        pass

    def check_version(self):
        ''' Check if compatible dolfin version is installed '''
        if dolfin.__version__ >= '2018':
            pass
        elif (DOLFIN_VERSION_MAJOR < 2017 or (DOLFIN_VERSION_MAJOR == 2017
                                              and DOLFIN_VERSION_MINOR < 2)):
            raise Exception('DOLFIN version >= 2017.2.0 required')

    def init(self):
        ''' Initialize problem:
            1. function spaces
            2. boundary conditions
            3. variational forms
        '''
        if self.options:
            self.set_constants()
            self.init_mesh()
            self.mixed_functionspace()
            self.boundary_conditions()
            self.variational_form()
        else:
            raise Exception('Options not set. call get_parameters(optfile) '
                            'first!')

        self.t = 0

        # self.read_checkpoint()

        return self

    def setup_logger(self):
        ''' Create logging File Handler '''
        MPI.barrier(MPI.comm_world)
        path = Path(self.options['io']['write_path']).joinpath('run.log')
        if MPI.rank(MPI.comm_world) == 0:
            utils.trymkdir(str(path.parent))
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        MPI.barrier(MPI.comm_world)
        self.set_log_filehandler(str(path))

    def set_constants(self):
        ''' Set Constants from options file as instance attributes '''
        if 'fluid' not in self.options:
            raise DeprecationWarning('Fluid properties should be defined in '
                                     'the input file as items of a \'fluid\' '
                                     'section, e.g.,\n fluid:\n\tdensity: 1.\n'
                                     '\tdynamic_viscosity: 0.001')

            self.mu = Constant(self.options['dynamic_viscosity'])
            self.rho = Constant(self.options['density'])

        else:
            self.mu = Constant(self.options['fluid']['dynamic_viscosity'])
            self.rho = Constant(self.options['fluid']['density'])

        if (self.options['timemarching']['monolithic']['timescheme'] !=
                'steady'):
            self.dt = Constant(self.options['timemarching']['dt'])
        else:
            self.dt = Constant(0)
        self.dt0 = Constant(0)

    def get_parameters(self, inputfile):
        ''' Reads parameters from YAML input file into options dictionary

        Args:
            inputfile (str):     path to YAML file
        '''
        self.options = inout.read_parameters(inputfile)
        self.logger.debug('Parameters:')
        self.logger.debug(inout.dump_parameters(self.options))
        return self

    def init_mesh(self):
        ''' Read mesh and boundary information. '''
        self.logger.info('Reading mesh {}'.format(self.options['mesh']))
        self.mesh, _, self.bnds = inout.read_mesh(self.options['mesh'])
        self.ndim = self.mesh.topology().dim()

    def mixed_functionspace(self):
        ''' Create mixed function space
                W = V x Q
        on given mesh and define the solution function, w \in W.

        Implemented options, set by options['elements']:
            'TH', 'P2P1':     Taylor-Hood P2/P1
            'Mini', 'P1bP1':  Mini P1+Bubble/P1
            'P1', 'P1P1':     P1/P1

        '''
        u_space = self.options['fem']['velocity_space']
        p_space = self.options['fem']['pressure_space']
        self.logger.info('Creating {}/{} function space'.format(u_space,
                                                                p_space))
        if u_space in ('p1', 'p2'):
            deg = int(u_space[1])
            U = VectorElement('P', self.mesh.ufl_cell(), deg)
        elif u_space in ('p1b', 'p1+'):
            deg = int(u_space[1])
            P = FiniteElement('P', self.mesh.ufl_cell(), deg)
            B = FiniteElement('Bubble', self.mesh.ufl_cell(), 1 + self.ndim)
            U = VectorElement(P + B)
        else:
            raise Exception('Velocity space "{}" not supported!'
                            .format(u_space))

        if p_space == 'p1':
            P = FiniteElement('P', self.mesh.ufl_cell(), 1)
        elif p_space in ('p0', 'dg0'):
            P = FiniteElement('DG', self.mesh.ufl_cell(), 0)
        elif p_space in ('p1-', 'dg1'):
            P = FiniteElement('DG', self.mesh.ufl_cell(), 1)
        else:
            raise Exception('Pressure space "{}" not supported!'
                            .format(p_space))

        W = FunctionSpace(self.mesh, MixedElement([U, P]))

        self.logger.info('Number of DOFs: {}'.format(W.dim()))
        w = Function(W)
        w.vector().zero()

        self.w = w
        self.W = W

    def boundary_conditions(self):
        ''' Process boundary conditions.
        Boundary conditions are defined by means of a list of dictionaries of
        the form
            [id1, { settings... },
             id2: { settings... },
                ...],
        where id1 and id2 are boundary indicators, matching self.bnds.
        **NOTE**: id=0 is reserved for INTERIOR EDGES. All boundaries need
        indicators id > 0!

        The settings of each boundary condition is a dictionary with keys

            'preset':  (Optionally) selects a predefined boundary condition.
                Each preset requires corresponding parameters set via the key
                'value': Possible options:

                    'noslip': No-slip BC with
                        'value' = float/Expression, or tuple with len == ndim
                    'driven_lid': regularized driven lid,
                        'value' = U
                    'inflow': parabolic inflow BC, requires
                        'value' = {'R': radius(optional), 'U': u_max,
                                    'symmetric': boolean}
                            Set 'symmetric' = True for half-parabola with u_max
                            at bottom, and False for full parabolic profile
                    'outflow': stress outlet with
                        'value' = pn
                    'symmetry': symmetry boundary condition

            'method': Method of imposing Dirichlet BCs. Nitsche's weak method
                is applied wrt normal and tangential compontents of the
                velocity vector, whereas standard essential BCs by constraining
                the function space (default) are set wrt to (x, y, z)
                components. If a single value is found as the 'value' item, it
                is applied on all components.
                options:
                    'nitsche':  requires 'value' = (normal, tangential)
                    'essential': requires 'value' = (ux, uy, uz)
                        or 'value' = u_all.
                    Compatible with 'preset'.

            'type': Set type of boundary condition. Self-explaining items:
                'dirichlet', 'neumann'

            'value': The boundary value to be applied by the method
                specified above. Can be a single value or tuple (n, t) or (x,
                y, (z)) of numbers, DOLFIN Constants or Expressions.
                If combined with 'preset', value needs to be passed
                accordingly.
                Neumann accepts scalar for normal component
                Dirichlet essential (x, y, (z))
                Dirichlet Nitsche (n, )   # TODO extend for cartesian coords?
                                           # e.g. for 3D driven cavity lid
                                           # TODO: add flag 'cartesian'
                Navier-Slip: tba

        Attributes:
            bcs:        list of DirichletBC objects
            bcs_weak    list of weak Neumann, Navier-Slip or Nitsche BCs

        '''
        self.bcs = []
        self.bcs_neumann = []
        self.bcs_nitsche = []
        self.bcs_navierslip = []
        self.bcs_transpiration = []
        bc_lst = copy.deepcopy(self.options['boundary_conditions'])
        #  bc_lst = self.options['boundary_conditions']
        self.bc_lst = bc_lst

        self.check_boundary_conditions()

        for bc in bc_lst:
            if 'preset' in bc and bc['preset']:
                self._preset_bc_selector(bc)
            elif bc['type'] == 'dirichlet':
                if 'method' in bc and bc['method'] == 'nitsche':
                    self._proc_nitsche_bc(bc)
                else:
                    self._proc_dirichlet_bc(bc)
            elif bc['type'] == 'neumann':
                self._proc_neumann_bc(bc)
            elif bc['type'] == 'navierslip':
                raise Exception('This should never happen! Specify navier-slip'
                                ' BCs via preset interface')
                self._proc_navierslip_bc(bc)

        if self.options['fem']['fix_pressure'] == 1:
            self._proc_pressure_point_bc()

        return self

    def _form_weak_bcs(self, w):
        ''' Add weak boundary conditions (Neumann, Nitsche, Navier-Slip) to
        LHS & RHS of variational forms.
        'w' is passed (TrialFunction for bilinear_, Function self.w for
        nonlinear_) so that the same functions can be used for linearized and
        nonlinear solvers.

        Args:
            w       1. Function (nonlinear case) or TrialFunction (lin)
                        or
                    2. TUPLE of functions w = (u, p)

        Return:
            a, L    lists of BC contributions to a, L
        '''
        self.logger.info('Creating weak forms of weak boundary conditions')
        Llist_neu = self._create_neumann_form()
        alist_nit, Llist_nit = self._create_nitsche_form(w)
        alist_nav = self._create_navierslip_form(w)
        alist_nav += self._create_transpiration_form(w)

        return alist_nit + alist_nav, Llist_neu + Llist_nit

    def _create_neumann_form(self):
        ''' Create contributions to (a, L) due to Neumann BCs.

        Return:
            a       list of bilinear form terms (natural boundary integral)
            L       list of RHS contributions
        '''

        (v, q) = TestFunctions(self.W)
        # (u, p) = split(w)

        ds = Measure('ds', domain=self.mesh, subdomain_data=self.bnds)

        n = FacetNormal(self.mesh)
        # a = []
        L = []

        for bc in self.bcs_neumann:
            bid = bc[0]
            val = bc[1]
            assert self.is_Constant(val) or self.is_Expression(val)
            assert not val.ufl_shape, 'Neumann data required as scalar.'
            L.append(val*dot(v, n)*ds(bid))

        return L

    def _create_nitsche_form(self, w):
        ''' Create contributions to (a, L) due to weak Dirichlet BCs via the
        Nitsche method.
        According to the options, the positivity/stability terms are
        skew-symmetric, e.g.,
            -<u-g, mu*grad(v)*n + q*n>
        (pressure cancels out when (v,q) = (u,p)) or "positivity" ensuring
        (TODO: check this!), e.g.,
            +<u-g, mu*grad(v)*n - q*n>,
        (full integral cancels out for (v,q) = (u,p)).

        Attributes:
            self.bcs_nitsche    list of shape [bid, bcval] where bcval is a
                                dolfin Constant or Expression of len 1 (normal
                                component imposed) or ndim (full cartesian
                                components)

        Args:
            w       1. Function (nonlin) or TrialFunction (lin)
                        or
                    2. TUPLE of (u, p)   (i.e. (u_mid, p))

        Return:
            a       list of bilinear form terms (natural boundary integral)
            L       list of RHS contributions
        '''
        a = []
        L = []

        if not self.bcs_nitsche:
            return a, L

        if isinstance(w, dolfin.function.function.Function):
            (u, p) = split(w)
            self.logger.debug('_create_nitsche_form: received FUNCTION w')
        elif isinstance(w, list) or isinstance(w, tuple):
            u, p = w
            self.logger.debug('_create_nitsche_form: received TUPLE w=(u, p)')
        else:
            raise Exception('Argument w invalid.')

        (v, q) = TestFunctions(self.W)
        n = FacetNormal(self.mesh)
        h = CellDiameter(self.mesh)
        ds = Measure('ds', domain=self.mesh, subdomain_data=self.bnds)

        beta1 = Constant(self.options['nitsche']['beta1'])
        beta2 = Constant(self.options['nitsche']['beta2'])
        met = self.options['nitsche']['method']
        mu = self.mu

        for bc in self.bcs_nitsche:
            bid = bc[0]
            val = bc[1]
            if not val.ufl_shape:
                # normal component only
                a.append((
                    -dot(mu*grad(u)*n - p*n, n)*dot(v, n)
                    + beta1/h*mu*dot(u, n)*dot(v, n)
                    + beta2/h*dot(u, n)*dot(v, n)
                )*ds(bid))
                L.append(
                    beta1/h*mu*val*dot(v, n)*ds(bid)
                    + beta2/h*val*dot(v, n)*ds(bid)
                )
                if met == 0:
                    a.append(-dot(mu*grad(v)*n + q*n, n)*dot(u, n)*ds(bid))
                    L.append(-dot(mu*grad(v)*n + q*n, n)*val*ds(bid))
                elif met == 1:
                    a.append(+dot(mu*grad(v)*n - q*n, n)*dot(u, n)*ds(bid))
                    L.append(+dot(mu*grad(v)*n - q*n, n)*val*ds(bid))

            elif len(val) == self.ndim:
                a.append((
                    -dot(mu*grad(u)*n - p*n, v)
                    + beta1/h*mu*dot(u, v)
                    + beta2/h*dot(u, n)*dot(v, n)
                    # beta2/h*dot(u, v)
                )*ds(bid))
                L.append(
                    beta1/h*mu*dot(val, v)*ds(bid)
                    + beta2/h*dot(val, v)*ds(bid)
                )
                if met == 0:
                    a.append(-dot(mu*grad(v)*n + q*n, u)*ds(bid))
                    L.append(-dot(mu*grad(v)*n + q*n, val)*ds(bid))
                elif met == 1:
                    a.append(+dot(mu*grad(v)*n - q*n, u)*ds(bid))
                    L.append(+dot(mu*grad(v)*n - q*n, val)*ds(bid))
            else:
                raise Exception('Invalid shape of BC value')

        return a, L

    def _create_navierslip_form(self, w):
        ''' Create contributions to (a, L) due to Navier-slip BCs.

        Attributes:
            self.bcs_navierslip     list of shape [bid, bcval] where bcval is a
                                    scalar dolfin Constant or Expression

        Args:
            w       1. Function (nonlin) or TrialFunction (lin)
                        or
                    2. TUPLE of functions w = (u, p)

        Return:
            a       list of bilinear form terms (natural boundary integral)
        '''
        a = []

        if not self.bcs_navierslip:
            return a

        if isinstance(w, dolfin.function.function.Function):
            (u, p) = split(w)
            self.logger.debug('_create_navierslip_form: received FUNCTION w')
        elif isinstance(w, list) or isinstance(w, tuple):
            u, p = w
            self.logger.debug('_create_navierslip_form: received TUPLE '
                              'w=(u, p)')
        else:
            print(type(w))
            raise Exception('Argument w invalid.')

        (v, q) = TestFunctions(self.W)
        n = FacetNormal(self.mesh)
        ds = Measure('ds', domain=self.mesh, subdomain_data=self.bnds)

        for bc in self.bcs_navierslip:
            bid = bc[0]
            val = bc[1]
            assert self.is_Constant(val) or self.is_Expression(val), (
                'Gamma (val) expected to be of type Constant or Expression')

            a.append(val*(dot(u, v) - dot(u, n)*dot(v, n))*ds(bid))

        return a

    def _create_transpiration_form(self, w):
        ''' Create contributions to (a, L) due to transpiration BCs.

        Attributes:
            self.bcs_transpiration  list of shape [bid, bcval] where bcval is a
                                    scalar dolfin Constant or Expression

        Args:
            w       1. Function (nonlin) or TrialFunction (lin)
                        or
                    2. TUPLE of functions w = (u, p)

        Return:
            a       list of bilinear form terms (natural boundary integral)
        '''
        a = []

        if not self.bcs_transpiration:
            return a

        if isinstance(w, dolfin.function.function.Function):
            (u, p) = split(w)
            self.logger.debug('_create_nitsche_form: received FUNCTION w')
        elif isinstance(w, list) or isinstance(w, tuple):
            u, p = w
            self.logger.debug('_create_nitsche_form: received TUPLE w=(u, p)')
        else:
            raise Exception('Argument w invalid.')

        (v, q) = TestFunctions(self.W)
        n = FacetNormal(self.mesh)
        ds = Measure('ds', domain=self.mesh, subdomain_data=self.bnds)

        for bc in self.bcs_transpiration:
            bid = bc[0]
            val = bc[1]
            assert self.is_Constant(val) or self.is_Expression(val), \
                'Gamma (val) expected to be of type Constant or Expression'

            a.append(val*dot(u, n)*dot(v, n)*ds(bid))

        return a

    def _backflowstab_lin(self):
        ''' Build backflow stabilization terms for outlet and for weak
        Dirichlet BCs.

        Args:

        Return:
            alist       contribution to variational form (Picard terms)
            alist_newt  Newton extra terms
        '''
        self.logger.warning('linear backflow stabilization method should '
                            'only be used for STEADY problems.')
        ds = Measure('ds', domain=self.mesh, subdomain_data=self.bnds)
        # (u, p) = TrialFunctions(self.W)
        w = Function(self.W)
        (u, p) = split(w)
        (v, q) = TestFunctions(self.W)
        (u0, p0) = split(self.w)

        n = FacetNormal(self.mesh)
        alist = []
        alist_newt = []
        if self.options['backflowstab']['outlet']:
            if self._bid_outlet:
                bid = self._bid_outlet
                alist.append(-0.5*self.rho*0.5 *
                             (dot(u0, n) - abs(dot(u0, n)))*dot(u, v)*ds(bid)
                             )
                # -0.5*self.rho*self.abs_n(dot(u0, n))*dot(u, v)*ds(bid)
                alist_newt.append(
                    -0.5*self.rho*0.5*(
                        # dot(u, n)*dot(u, v) - abs(dot(u, n))*dot(u, v)
                        dot(u, n)*dot(u0, v) -
                        sign(dot(u0, n))*dot(u, n)*dot(u0, v)
                        # + dot(u0, n)*dot(u, v)  # included in Picard term
                        # -abs(dot(u0, n))*dot(u, v) # included in Picard term
                        # last line equals
                        # sign(dot(u0, n))*dot(u0, n)*dot(u, v)
                    )*ds(bid)
                )
                # print('outlet stab: bid {0}'.format(bid))

        # XXX: hacked?? loop through nitsche AND TRANSPIRATION
        for bc in self.bcs_nitsche + self.bcs_transpiration:
            assert not bc[0] == self._bid_outlet, (
                'Outlet ID == Nitsche ID. This should never happen.')

            bid = bc[0]
            if self.options['backflowstab']['nitsche'] == 0:
                # no backflow stabilization on Nitsche boundaries
                pass
            elif self.options['backflowstab']['nitsche'] == 1:
                alist.append(
                    -0.5*self.rho*dot(u0, n)*dot(u, v)*ds(bid)
                )
                alist_newt.append(
                    -0.5*self.rho*dot(u, n)*dot(u0, v)*ds(bid)
                )

            elif self.options['backflowstab']['nitsche'] == 2:
                alist.append(
                    -0.5*self.rho*0.5 *
                    (dot(u0, n) - abs(dot(u0, n)))*dot(u, v)*ds(bid)
                )
                alist_newt.append(
                    -0.5*self.rho*0.5*(
                        dot(u, n)*dot(u0, v) -
                        sign(dot(u0, n))*dot(u, n)*dot(u0, v)
                    )*ds(bid)
                )

        return alist, alist_newt

    def _backflowstab_nonlin(self):
        ''' Build backflow stabilization terms for outlet and for weak
        Dirichlet BCs.

        Return:
            alist   list of contributions to variational form
        '''
        (v, q) = TestFunctions(self.W)
        (u, p) = split(self.w)
        ds = Measure('ds', domain=self.mesh, subdomain_data=self.bnds)

        n = FacetNormal(self.mesh)
        # h = CellDiameter(self.mesh)
        alist = []
        if self.options['backflowstab']['outlet']:
            if self._bid_outlet:
                bid = self._bid_outlet
                alist.append(
                    -0.5*self.rho*self.abs_n(dot(u, n))*dot(u, v)*ds(bid)
                )

        # XXX: hacked?? loop through nitsche AND TRANSPIRATION
        for bc in self.bcs_nitsche + self.bcs_transpiration:
            assert not bc[0] == self._bid_outlet, (
                'Outlet ID == Nitsche ID. This should never happen.')
            bid = bc[0]

            if self.options['backflowstab']['nitsche'] == 0:
                # no backflow stabilization at Nitsche boundaries
                pass
            elif self.options['backflowstab']['nitsche'] == 1:
                alist.append(
                    -0.5*self.rho*dot(u, n)*dot(u, v)*ds(bid)
                )
            elif self.options['backflowstab']['nitsche'] == 2:
                alist.append(
                    -0.5*self.rho*self.abs_n(dot(u, n))*dot(u, v)*ds(bid)
                )

        return alist

    def _backflowstab(self, w, w_):
        ''' Build backflow stabilization terms for outlet and for weak
        Dirichlet BCs.
        Generalized version for linear and nonlinear problems.
        The unknown function (w) and a second function, w_, has to be given,
        as appears in the convection term:
            For nonlinear problems (to be solved with the Newton method), w=w_.
            For linearized problems, w_ is the known iteration (self.w).

        Args:
            w:  unknown Function or TUPLE (u, p)
            w_: Function used for convection velocity or TUPLE (u, p)

        Return:
            alist   list of contributions to variational form
        '''
        # check if a Function or a Tuple of split function was given
        if dolfin.__version__ >= '2019':
            type_function = dolfin.function.function.Function
        else:
            type_function = dolfin.function.function.Function

        if isinstance(w, type_function):
            (u, p) = split(w)
            self.logger.debug('_backflowstab: received FUNCTION w')
        elif isinstance(w, list) or isinstance(w, tuple):
            u, p = w
            self.logger.debug('_backflowstab: received TUPLE w=(u, p)')
        else:
            raise Exception('Argument w invalid.')

        if isinstance(w_, type_function):
            (u_, p_) = split(w_)
            self.logger.debug('_backflowstab: received FUNCTION w')
        elif isinstance(w, list) or isinstance(w, tuple):
            u_, p_ = w_
            self.logger.debug('_backflowstab: received TUPLE w_=(u_, p_)')
        else:
            raise Exception('Argument w_ invalid.')

        (v, q) = TestFunctions(self.W)
        # (u, p) = split(w)
        # (u_, p_) = split(w_)
        ds = Measure('ds', domain=self.mesh, subdomain_data=self.bnds)

        n = FacetNormal(self.mesh)
        # h = CellDiameter(self.mesh)
        alist = []
        # FIX: 04/03/2018: bow all backflow stab boundaries need to be given
        # explicitly in fem:stabilization:backflow_boundaries

        for bid in self.options['fem']['stabilization']['backflow_boundaries']:
            alist.append(
                -0.5*self.rho*self.abs_n(dot(u_, n))*dot(u, v)*ds(bid)
                # -0.5*self.rho*dot(u_, n)*dot(u, v)*ds(bid)
            )

        return alist

    def _preset_bc_selector(self, bc):
        ''' Prepare preset boundary condition.
        Create data sets and call _proc_dirichlet_bc and/or
        _proc_neumann_bc as necessary.

        Recognized options:
            noslip
            inlet
            driven_lid
            outlet
            symmetry
            navierslip
            navierslip_transpiration

        time-dependent: sine_parabola_inlet

        Args:
            bc     list of BCs {'id': bid, 'method', 'value',...}
        '''
        preset = bc['preset']

        if preset == 'parabola_inlet':
            # get direction of boundary, assuming it is (more or less) straight
            self._preset_parabola_inlet_bc(bc)

        elif preset == 'driven_lid':
            self._preset_driven_lid_bc(bc)

        elif preset == 'noslip':
            self._preset_noslip_bc(bc)

        elif preset == 'outlet':
            self._preset_outlet_bc(bc)

        elif preset == 'symmetry':
            self._preset_symmetry_bc(bc)

        elif preset == 'navierslip':
            self._preset_navierslip_bc(bc)

        elif preset == 'no_penetration':
            self._preset_no_penetration_bc(bc)

        elif preset == 'navierslip_transpiration':
            self._preset_navierslip_bc(bc)
            self._preset_transpiration_bc(bc)

        elif preset == 'transpiration':
            self._preset_transpiration_bc(bc)

        elif preset == 'sine_parabola_inlet':
            self._preset_sine_inlet_bc(bc)

        else:
            raise ValueError('BC type \'{0}\' not recognized.'.format(preset))

        pass

    def _preset_parabola_inlet_bc(self, bc):
        ''' Process preset inlet boundary condition, create format that
        function _proc_dirichlet_bc understands.

        Args:
            bc      dictionary describing one boundary condition
        '''
        if 'method' in bc and bc['method'] == 'nitsche':
            self.logger.warn('Inlet BCs are imposed strongly. Ignoring '
                             'Nitsche setting.')

        if not self.is_Expression(bc['value']):
            # shallow copy for ReynoldsContinuation
            # if 'symmetric' in bc['symmetric'] and bc['symmetric']:
            #     # assume that symm_normal_dir indicates the normal direction
            #     # on the symmetry plane and that the symmetry axis/plane goes
            #     # through zero
            #     symm_normal_dir = bc['symmetry_normal_dir']

            flow_direction = bc['flow_direction'] if 'flow_direction' in bc \
                else 0
            assert flow_direction in range(self.ndim), (
                'flow direction {} does not match problem dimension'
                .format(flow_direction))

            if 'bnd_interval' in bc['value'] and bc['value']['bnd_interval']:
                bnd_interval = bc['value']['bnd_interval']
                r0 = 0.5*abs(bnd_interval[1] - bnd_interval[0])
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

            inflow_lst = ['0.0']*self.ndim
            inflow_lst[flow_direction] = inflow_str
            inflow = Expression(inflow_lst,
                                U=bc['value']['U'],
                                x0=x0,
                                R=r0,
                                degree=2)
            bc['value'] = inflow

        if 'method' in bc and bc['method'] == 'nitsche':
            raise Exception('Inlet only possible via strong Dirichlet BCs.')
        else:
            self._proc_dirichlet_bc(bc)
        pass

    def _preset_driven_lid_bc(self, bc):
        ''' Process preset driven lid boundary condition, create format that
        function _proc_dirichlet_bc understands.
        Define a 2D or 3D Expression for a *regularized* velocity
            u = U/R^4*(R^4 - (x - x0)^4), v = 0
        or (assuming cube),
            u = U/R^4*(R^4 - (x - x0)^4 - (z - x0)^4, v = 0, w = 0
        where x0 is the center point of the boundary and R its straight line
        distance to the bounds of the surface or line.

        Args:
            bc      dictionary describing one boundary condition
        '''
        self.logger.warn(('Driven lid assumed to be top boundary, moving in '
                          'the x direction'))

        lid_interval = bc['value']['lid_interval']
        x0 = 0.5*(lid_interval[1] + lid_interval[0])
        r0 = 0.5*(lid_interval[1] - lid_interval[0])

        if self.ndim == 2:
            lid_str = 'U/R4*(R4 - pow(x[0] - x0, 4))'
        elif self.ndim == 3:
            lid_str = ('U/R4*(R4 - pow(x[0] - x0, 4) - pow(x[2] - x0, 4))')
            self.logger.warn('check 3D driven lid expression. UNTESTED.')

        lid_lst = ['0.0']*self.ndim
        lid_lst[0] = lid_str

        lid = Expression(lid_lst, U=bc['value']['U'], x0=x0, R4=r0**4,
                         degree=4)
        bc['value'] = lid

        if 'method' in bc and bc['method'] == 'nitsche':
            self._proc_nitsche_bc(bc)
        else:
            self._proc_dirichlet_bc(bc)
        pass

    def _preset_noslip_bc(self, bc):
        ''' Process preset no-slip boundary condition, create format that
        function _proc_dirichlet_bc understands.

        Args:
            bc      dictionary describing one boundary condition
        '''
        if 'method' in bc and bc['method'] == 'nitsche':
            self._proc_nitsche_bc(bc)
        else:
            self._proc_dirichlet_bc(bc)

        pass

    def _preset_outlet_bc(self, bc):
        ''' Process preset pressure outlet boundary condition, create format
        that function _proc_neumann_bc understands. (Just call function)

        Args:
            bc      dictionary describing one boundary condition
        '''
        self._proc_neumann_bc(bc)
        self._bid_outlet = bc['id']
        pass

    def _preset_symmetry_bc(self, bc):
        ''' Process preset symmetry boundary condition, create format that
        functions _proc_dirichlet_bc and _proc_neumann_bc understand.

        Create Dirichlet BC for u_n = 0 and Neumann for n.sigma.t = 0.

        Args:
            bc      dictionary describing one boundary condition
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
            self._proc_nitsche_bc(bc_d)
        else:
            symm_normal_dir = bc['normal_dir']
            val = [None]*self.ndim
            val[symm_normal_dir] = 0.
            bc_d['value'] = val
            self._proc_dirichlet_bc(bc_d)

        # XXX do nothing: normal component is taken care of by the Dirichlet BC
        #       and tangential component is zero.
        # self._proc_neumann_bc(bc_n)
        pass

    def _preset_navierslip_bc(self, bc):
        ''' Process preset Navier-slip boundary condition, create format that
        function _proc_navierslip_bc understands.

        Args:
            bc      dictionary describing one boundary condition
        '''

        bid = bc['id']

        # gm = bc['value']['gm']
        # R_inn = bc['value']['R'] if 'R' in bc['value'] else 0
        # #  R_out = R_inn + bc['value']['dR']
        # dR = bc['value']['dR']
        if isinstance(bc['value'], dict) and 'gamma' in bc['value']:
            gamma = Constant(bc['value']['gamma'])
        elif isinstance(bc['value'], (int, float)):
            gamma = Constant(bc['value'])
        else:
            raise ValueError('Navier-Slip BC value type not understood ({})'.
                             format(type(bc['value'])))

        # gm_pois = 2.0*self.options['dynamic_viscosity']*R_inn/(R_inn**2 - R_out**2)
        # # gm_const = Constant(gm_pois*gm)
        # gm_expr = Expression('G0*a', G0=gm_pois, a=gm)

        # XXX define gm_expr as Expression of variables: mu, Ri, Ro, a=factor
        # Now gm_expr has a starting value, but each variable can be modified
        # separately.
        #  gm_expr = Expression('2.0*a*mu*R_i/(R_i*R_i - R_o*R_o)',
        #                    a=gm, mu=self.options['dynamic_viscosity'], R_i=R_inn, R_o=R_out)
        # XXX 22/09/16      write gamma in terms of R_i, dR only!

        # if R_inn > 0:
        #     gm_expr = Expression('2.0*a*mu*R_i/(-2*R_i*dR - dR*dR)',
        #                          a=gm, mu=self.options['dynamic_viscosity'], R_i=R_inn, dR=dR,
        #                          degree=1)
        # else:
        #     gm_expr = Expression('2.0*a*mu*x[1]/(-2*x[1]*dR - dR*dR)',
        #                          a=gm, mu=self.options['dynamic_viscosity'], dR=dR, degree=1)

        self._proc_navierslip_bc([bid, gamma])
        pass

    def _preset_no_penetration_bc(self, bc):
        ''' Process no-penetration boundary condition.

        Args:
            bc      dictionary describing one boundary condition
        '''

        bc_d = {'id': bc['id'],
                'method': bc['method'] if 'method' in bc else 'nitsche'
                }
        if bc_d['method'] == 'nitsche':
            bc_d['value'] = 0.
            self._proc_nitsche_bc(bc_d)
        else:
            self.logger.warn('Using strong DBC for non-penetration BC')
            val = [None]*self.ndim
            val[bc['normal_dir']] = 0.
            bc_d['value'] = val
            self._proc_dirichlet_bc(bc_d)

        pass

    def _preset_transpiration_bc(self, bc):
        ''' Process preset transpiration boundary condition, create format that
        function _proc_transpiration_bc understands.

        Args:
            bc      dictionary describing one boundary condition
        '''
        bid = bc['id']
        if isinstance(bc['value'], dict) and 'beta' in bc['value']:
            beta = Constant(bc['value']['beta'])
        elif isinstance(bc['value'], (int, float)):
            beta = Constant(bc['value'])
        else:
            raise ValueError('Transpiration BC value type not understood ({})'.
                             format(type(bc['value'])))

        self._proc_transpiration_bc([bid, beta])

        pass

    def _preset_sine_inlet_bc(self, bc):
        ''' Process preset sine oscillation inlet boundary condition, create
        format that function _proc_dirichlet_bc understands.

        Args:
            bc      dictionary describing one boundary condition
        '''
        if 'method' in bc and bc['method'] == 'nitsche':
            self.logger.warn('Inlet BCs are imposed strongly. Ignoring '
                             'Nitsche setting.')

        if not self.is_Expression(bc['value']):
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

            inflow_lst = ['0.0']*self.ndim
            inflow_lst[flow_direction] = inflow_str
            inflow = Expression(inflow_lst,
                                U=bc['value']['U'],
                                x0=x0,
                                R=r0,
                                a=bc['value']['a'],
                                t=0.0,
                                degree=2)
            bc['value'] = inflow

            self.time_expr.append(bc['value'])

        if 'method' in bc and bc['method'] == 'nitsche':
            raise Exception('Inlet only possible via strong Dirichlet BCs.')
        else:
            self._proc_dirichlet_bc(bc)
        pass

    def _get_boundary_orientation(self, bid):
        ''' Get boundary orientation from boundary  MeshFunction 'bnds'.
        Extracts coordinates of boundary points and and finds constant
        cartesian coordinate direction, if any. This direction equals the
        normal direction. Limited to horizontal and vertical boundaries.

        Args:
            bid     boundary indicator

        Returns:
            imin    component index with minimum coordinate standard deviation

        Example:
            _get_boundary_orientation(1) == 0 means, that along boundary 1, the
            x (i=0) coordinate is constant; i.e. the normal vector is n = e_x.
            A return value imin = 1 means that y = const, the boundary is
            parallel to the X-Z plane.
        '''
        # http://fenicsproject.org/qa/9135/obtain-coordinates-defined-by-mesh-function?show=9135#q9135
        raise Exception('_get_boundary_orientation dropped!')

        It_facet = SubsetIterator(self.bnds, bid)
        self.logger.warn('SubsetIterator deprecated! avoid this for parallel')
        pts = []
        for c in It_facet:
            pts.append([c.midpoint().x(), c.midpoint().y(), c.midpoint().z()])
        pts_std = np.array(pts).std(axis=0)
        if self.ndim == 2:
            pts_std = pts_std[:self.ndim]
        imin = np.argmin(pts_std)
        assert pts_std[imin] <= 10*DOLFIN_EPS, (
            'Found minimal std = {0} in coordinate direction i = {1}, does '
            'not seem to be a straight line'.format(pts_std[imin], imin))

        return imin

    def _get_inlet_parabola_coef(self, bid, bnd_dir, symmetric=False):
        ''' Get radius and center point for parabolic inflow profile.
        3D: assume circular inlet!
        Symmetric: always assume symmetry axis is (0, 0) center line
        Also require that R > 0!

        Args:
            bid (int)       boundary indicator
            bnd_dir (int)   normal direction (x,y,z) = (0,1,2) at boundary

        Returns:
            R               Radius w.r.t. midpoint
            x0              center/mid point
        '''
        raise Exception('_get_inlet_parabola_coef was dropped!')

        It_facet = SubsetIterator(self.bnds, bid)
        pts = []
        for c in It_facet:
            for v in vertices(c):
                ptsi = [v.point().x(), v.point().y(), v.point().z()]
                ptsi.pop(bnd_dir)
                pts.append(ptsi)

        pts = np.array(pts)
        assert pts.shape[1] == 2, (
            'Something wrong with dimensions. Index bnd_dir not deleted?')

        pts1 = pts[:, 0]
        if symmetric:
            x0 = pts1.min()
            assert x0 == 0.0, 'Symmetric: x0 expected to be 0.0'
            R = pts1.max()
        else:
            x0 = 0.5*(pts1.max() + pts1.min())
            R = 0.5*(pts1.max() - pts1.min())

        if self.ndim == 3 and symmetric:
            assert pts[:, 0].max() - pts[:, 1].max() <= DOLFIN_EPS, (
                'Symmetric 3D section does not seem circular AND with'
                ' *positive radius*')

        return x0, R

    def _proc_navierslip_bc(self, bc):
        ''' Prepare Navier-Slip Gamma boundary condition '''
        bid = bc[0]
        bcval = bc[1]

        if self.bcs_navierslip is None:
            self.bcs_navierslip = []

        if isinstance(bcval, (int, float)):
            bcval = Constant(bcval)
        elif not (self.is_Constant(bcval) or self.is_Expression(bcval)):
            raise Exception('Navier-Slip friction coefficient must be a number'
                            ' or a dolfin Constant/Expression.')

        self.bcs_navierslip.append([bid, bcval])
        pass

    def _proc_transpiration_bc(self, bc):
        ''' Prepare transpiration boundary condition '''
        bid = bc[0]
        bcval = bc[1]

        if self.bcs_transpiration is None:
            self.bcs_transpiration = []

        if isinstance(bcval, (int, float)):
            bcval = Constant(bcval)
        elif not (self.is_Constant(bcval) or self.is_Expression(bcval)):
            raise Exception('Transpiration resistance coefficient must be a '
                            'number or a dolfin Constant/Expression.')

        self.bcs_transpiration.append([bid, bcval])
        pass

    def _proc_dirichlet_bc(self, bc):
        ''' Create Dirichlet boundary condition and appends to instance
        self.bcs list.
        Check if bc value is given as
            1. list with len == ndim
                - of int/float
                - containing Nones AND int/float/Expression/Constant
        or  2. Constant with len == ndim
        or  3. Expression with optional dict of parameters
        and treat appropriately.

        Args:
            bc          BC dict, with keys {'id', 'method', 'type', 'value'}

        Attributes:
            self.bcs    appends BC to instance self.bcs list
        '''

        bid = bc['id']
        bcval = bc['value']
        if isinstance(bcval, tuple):
            bcval = [b for b in bcval]

        assert len(bcval) == self.ndim, (
            'BC {} does not match geometric dimensions. Use '
            '(val, None, None) if no BC should be set.'.format(bid))
        # Values given as numbers or list of numbers => convert

        if isinstance(bcval, list):
            # check if all list entries are numbers
            if all(isinstance(b, (int, float)) for b in bcval):
                bcval = Constant(bcval)
                V = self.W.sub(0)
                if self.is_enriched(V):
                    bcval = project(bcval, V.collapse(), solver_type='mumps')

                self.bcs.append(DirichletBC(V, bcval, self.bnds, bid))

            # check if all entries are strings
            elif all(isinstance(b, str) for b in bcval):
                # if all items in the bc list are strings, create ONE
                # Expression from all list entries by unpacking the list with
                # *kwarg. The parameter dict, if given under the 'parameters'
                # key, are passed, by unpacking the dict with **kwargs.
                deg = bc['degree'] if 'degree' in bc else 3
                bcval = Expression(bcval, degree=deg,
                                   **bc['parameters'])
                if 't' in bc['parameters']:
                    if (self.options['timemarching']['monolithic']
                            ['timescheme'] != 'steady'):
                        self.time_expr.append(bcval)
                V = self.W.sub(0)
                if self.is_enriched(V):
                    bcval = project(bcval, V.collapse(), solver_type='mumps')
                self.bcs.append(DirichletBC(V, bcval, self.bnds, bid))

            elif None in bcval:
                # get indices where not None
                inone = [i for i, x in enumerate(bcval) if x is not None]
                for i in inone:
                    if isinstance(bcval[i], (int, float)):
                        bcval[i] = Constant(bcval[i])
                    elif isinstance(bcval[i], str):
                        deg = bc['degree'] if 'degree' in bc else 3
                        bcval[i] = Expression(bcval[i], degree=deg,
                                              **bc['parameters'])
                    if (self.is_Constant(bcval[i]) or
                            self.is_Expression(bcval[i])):
                        Vi = self.W.sub(0).sub(i)
                        if self.is_enriched(Vi):
                            bcval[i] = project(bcval[i], Vi.collapse(),
                                               solver_type='mumps')
                        self.bcs.append(DirichletBC(Vi, bcval[i], self.bnds,
                                                    bid))
            else:
                raise Exception('Type in BC value array not recognized.')
        else:
            # not a list nor a number
            if self.is_Expression(bcval) or self.is_Constant(bcval):
                if bcval.ufl_shape and len(bcval) == self.ndim:
                    V = self.W.sub(0)
                    if self.is_enriched(V):
                        bcval = project(bcval, V.collapse(),
                                        solver_type='mumps')

                    self.bcs.append(DirichletBC(V, bcval, self.bnds, bid))

                else:
                    raise Exception('len(bcval) == ndim required!')
            else:
                raise Exception('bcval was expected to be dolfin Expression '
                                'or Constant')
        pass

    def _proc_nitsche_bc(self, bc):
        ''' Prepare Nitsche Dirichlet boundary condition.
        Assume that BC value is given as a list or Constant/Expression of the
        cartesian vector components [u1, u2, (u3)].
        The normal component can be specified by passing a scalar number value.

        Possible values:
            set full velocity vector
                [u1, u2, (u3)]
                Expression(u1, u2, u3)
                Constant(u1, u2, u3)
            only normal component
                u_n (scalar)

        Args:
            bc          BC dict, with keys {'id', 'method', 'value'}
        '''
        # FIXME: MAKE CLEAR NORMAL/TANGENT INDICATOR!
        if self.bcs_nitsche is None:
            self.bcs_nitsche = []

        bid = bc['id']

        bcval = bc['value']
        if isinstance(bcval, tuple):
            bcval = [bc for bc in bcval]

        if isinstance(bcval, list):
            assert len(bcval) == self.ndim, 'Dimension mismatch of Nitsche BC'
            if all(isinstance(b, (int, float)) for b in bcval):
                bcval = Constant(bcval)
            else:
                raise Exception('Vectorial nitsche BC expects list of numbers')

        elif isinstance(bcval, (int, float)):
            # interpret scalar value as normal component
            bcval = Constant(bcval)

        elif self.is_Constant(bcval) or self.is_Expression(bcval):
            if bcval.ufl_shape and len(bcval) == self.ndim:
                pass
            else:
                raise Exception('Nitsche BC of type Const/Expr expected to '
                                'have len=ndim')

        else:
            raise Exception('Nitsche BC format: list [u_1,..,u_d] or normal '
                            'component as scalar.')

        self.bcs_nitsche.append([bid, bcval])

        # OLD
        # if self.is_Constant(bcval) or self.is_Expression(bcval):
        #     if bcval.ufl_shape and len(bcval) == self.ndim:
        #         self.bcs_nitsche.append([bid, bcval])
        #     else:
        #         raise Exception('Nitsche BC of type Const/Expr expected to '
        #                         'have len=ndim')

        # elif type(bcval) is list:
        #     if (all(type(b) in (int, float) for b in bcval) and len(bcval) ==
        #             self.ndim):
        #         self.bcs_nitsche.append([bid, Constant(bcval)])
        #     elif bcval[1] is None and type(bcval[0]) in (int, float):
        #         self.bcs_nitsche.append([bid, Constant(bcval[0])])
        #     else:
        #         raise Exception('Nitsche BCs of list type need to be given '
        #                   'in form [num1, num2, (num3)] or [num1, None].')
        # else:
        #     raise Exception('Type of bc value needs to be list or '
        #                'Const/Expr.')

        pass

    def _proc_pressure_point_bc(self):
        ''' Prepare pressure point Dirichlet boundary condition.
        Set zero automatically.
        '''
        pt = self.options['fem']['fix_pressure_point']
        if len(pt) == self.mesh.topology().dim():
            bc = DirichletBC(self.W.sub(1), 0.0, geom.Point(pt),
                             method='pointwise')
            self.logger.info('applying pressure point BC')
            self.bcs.append(bc)
        else:
            raise Exception('Dimension of pressure BC point coordinates != '
                            'mesh dimension.')
        pass

    def _proc_neumann_bc(self, bc):
        ''' Prepare Neumann boundary condition.
            Assume a scalar value is given for the normal component of the
            stress vector, the tangential part is zero.

            Possible values/combinations:
                scalar p0:
                    'p0*dot(v, n)*ds(bid)'

            Args:
                bc     boundary condition dict
        '''
        if self.bcs_neumann is None:
            self.bcs_neumann = []

        bid = bc['id']

        bcval = bc['value']

        if isinstance(bcval, (int, float)):
            bcval = Constant(bcval)
            self.bcs_neumann.append([bid, bcval])
        else:
            raise Exception('Neumann BC requires a scalar value (float)')

        # OLD CODE. Too complex and general for specific needs.
        # elif type(bcval) is list and len(bcval) == 2:
        #     if all(type(b) in (int, float) for b in bcval):
        #         bcval = Constant(bcval)
        #         self.bcs_neumann.append([bid, bcval])
        #     elif None in bcval:
        #         # get indices where not None
        #         # NOTE: knowing that len == 2, this can be done much simpler
        #         #   but possible extension to cartesian 3D formulation ...
        #         inone = [i for i, x in enumerate(bcval) if x is not None]
        #         for i in inone:
        #             if isinstance(bcval[i], (int, float)):
        #                 bcval[i] = Constant(bcval[i])
        #             if (self.is_Constant(bcval[i]) or
        #                     self.is_Expression(bcval[i])):
        #                 self.bcs_neumann.append([bid, i, bcval[i]])
        #             else:
        #                 raise Exception('Value type not recognized.')
        #     else:
        #         raise Exception('Type in BC value array not recognized. '
        #                         'Maybe mixed numbers with Const/Expr?')
        # else:
        #     raise Exception('Expected list of length 2,
        # [normal, tangential]')

        # assert len(self.bcs_neumann) == count + 1, (
        #     'No or more than 1 Neumann BCs set. Should be exactly 1')

        pass

    def check_boundary_conditions(self):
        ''' Check consistency of boundary conditions. '''
        if not self.W:
            raise Exception('Function space needs to be created prior to'
                            ' creating boundary conditions.')
        if not self.bnds:
            raise Exception('Boundary indicator MeshFunction not set!')

        bcs = self.options['boundary_conditions']
        bc_id = [bc['id'] for bc in bcs]

        indicators = np.unique(self.bnds.array())
        indicators = indicators[indicators > 0]
        if not np.unique(bc_id).sort() == indicators.sort():
            raise Exception('Mesh boundary indicators do not match boundary '
                            ' conditions IDs.')

        return self

    def is_enriched(self, V):
        ''' Check if the given (sub) function space has enriched elements. '''
        if V.num_sub_spaces():
            V = V.sub(0)
        return isinstance(V.ufl_element(),
                          ufl.finiteelement.enrichedelement.EnrichedElement)

    def is_Expression(self, obj):
        ''' Check if object has type dolfin Expression '''
        if dolfin.__version__ >= '2018':
            return isinstance(obj, dolfin.function.expression.Expression)
        else:
            return isinstance(obj, dolfin.function.expression.Expression)

    def is_Constant(self, obj):
        ''' Check if object has type dolfin Constant '''
        if dolfin.__version__ >= '2018':
            return isinstance(obj, dolfin.function.constant.Constant)
        else:
            return isinstance(obj, dolfin.function.constant.Constant)

    def abs_n(self, x):
        return 0.5*(x - abs(x))


class NSSteadyProblem(ProblemBase):
    def __init__(self, inputfile=None):
        super().__init__(inputfile)

    def variational_form(self):
        ''' Caller for variational form functions.
        Build forms of bilinear (i.e., linearized) and nonlinear problem.
        Store in self.nls_form (F, J) and self.ls_form (F, J). ls_form is
        extended by Newton linearization terms (for quasi-Newton methods) and
        stored in self.qnls_form.
        '''

        if not self.W:
            self.mixed_functionspace()
        assert self.W and self.w, ('Function space W and function w not'
                                   'initialized.')

        self.logger.info('Create bilinear variational form')
        self.bilinear_form()

        self.logger.info('Create nonlinear variational form')
        self.nonlinear_form()

        return self

    def nonlinear_form(self):
        ''' Build residual and residual Jacobian of the nonlinear variational
        form.

        Attributes:
            nls_form        tuple (F, J)
        '''
        mu = self.mu
        rho = self.rho
        stemam = Constant(self.options['fem']['convection_skew_symmetric'])

        zero = Constant((0.,)*self.ndim)

        z = TestFunction(self.W)
        (v, q) = split(z)
        (u, p) = split(self.w)

        a = inner(mu*grad(u), grad(v))*dx + rho*dot(grad(u)*u, v)*dx - \
            p*div(v)*dx + q*div(u)*dx
        a += stemam*0.5*rho*div(u)*dot(u, v)*dx
        L = dot(zero, v)*dx

        # temporary! dirty hack!
        n = FacetNormal(self.mesh)
        ds = Measure('ds', domain=self.mesh, subdomain_data=self.bnds)
        self.energy_form = (inner(mu*grad(u), grad(u))*dx
                            + rho*dot(grad(u)*u, u)*dx
                            + stemam*0.5*rho*div(u)*dot(u, u)*dx
                            - 0.5*rho*0.5*(dot(u, n) -
                                           abs(dot(u, n)))*dot(u, u)*ds(2)
                            )

        a_bc, L_bc = self._form_weak_bcs(self.w)
        a_bfs = self._backflowstab_nonlin()

        a = sum([a] + a_bc + a_bfs)
        L = sum([L] + L_bc)

        F = a - L
        J = derivative(F, self.w)

        self.nls_form = [F, J]

        return self

    def bilinear_form(self):
        ''' Build residual F and residual Jacobian J of the *linearized*
        variational problem, by means of the Picard and the Newton method.

        Attributes:
            ls_form         tuple (F, J), Picard linearization
            qnls_form       tuple (F, J), Newton linearization
        '''
        mu = self.mu
        rho = self.rho
        stemam = Constant(self.options['fem']['convection_skew_symmetric'])
        # sbf = Constant(self.options['use_backflowstab'])

        zero = Constant((0.,)*self.ndim)
        # n = FacetNormal(self.mesh)

        # ds = Measure('ds', domain=self.mesh, subdomain_data=self.bnds)

        w = Function(self.W)
        (u, p) = split(w)
        (v, q) = TestFunctions(self.W)

        (u0, p0) = split(self.w)

        a = (inner(mu*grad(u), grad(v))*dx + rho*dot(grad(u)*u0, v)*dx -
             p*div(v)*dx + q*div(u)*dx)
        a += stemam*0.5*rho*div(u0)*dot(u, v)*dx

        L = dot(zero, v)*dx

        # Newton extra term for convection and Temam stabilization
        nc = (rho*dot(grad(u0)*u, v)*dx + stemam*0.5*rho*div(u)*dot(u0, v)*dx)

        a_bc, L_bc = self._form_weak_bcs(w)
        a_bfs, a_bfs_nl = self._backflowstab_lin()

        a = sum([a] + a_bc + a_bfs)
        L = sum([L] + L_bc)

        # F = action(a - L, self.w)
        # J = a

        J = derivative(a, w)
        F = ufl.replace(a - L, {w: self.w})
        Jnc_ = derivative(nc, w)

        Jnc = sum([J + Jnc_] + a_bfs_nl)
        self.ls_form = [F, J]
        self.qnls_form = [F, Jnc]

        return self


class NSUnsteadyProblem(ProblemBase):
    ''' Derived problem class for unsteady problems. Straight forward extension
    to NSSteadyProblem, highly experimental and to be considered a temporary
    solution (hack).
    '''
    def __init__(self, inputfile=None):
        super().__init__(inputfile)

        self.time_expr = []
        self.w0 = None
        self.w1 = None
        self.fs_theta = None

    def variational_form(self):
        ''' Create variational form, depending on time discretization scheme.
        '''

        if not self.W:
            self.mixed_functionspace()
        assert self.W and self.w, ('Function space W and function w not'
                                   'initialized.')

        meth = self.options['timemarching']['monolithic']['timescheme'].lower()
        if meth in ('generalized_midpoint', 'gmp'):
            self.form_generalized_midpoint()
        elif meth in ('fs', 'fractionalstep'):
            self.form_fracstep()

        return self

    def form_fracstep(self):
        ''' Fractional step-Theta second order time stepping scheme, written in
        terms of increments dw=(du, dp).
        Has better stability than the midpoint rule (theta=0.5, like TR=CN).
        GLS CURRENTLY NOT SUPPORTED!
        See [Ran99]_, [Tur99]_, [JMR06]_.

        .. [Ran99] Rannacher, R. (1999). Finite element methods for the
           incompressible Navier-Stokes equations [Lecture notes]. University
           of Heidelberg.
           url: http://numerik.iwr.uni-heidelberg.de/Paper/Preprint1999-14.pdf

        .. [Tur99] Turek, S. (1999). Efficient solvers for incompressible flow
           problems: An algorithmic and computational approach. Springer.

        .. [JMR06] John, V., Matthies, G., & Rang, J. (2006). A comparison of
           time-discretization/linearization approaches for the incompressible
           Navier–Stokes equations. Computer Methods in Applied Mechanics and
           Engineering, 195(44–47), 5995–6010.
           url: https://doi.org/10.1016/j.cma.2005.10.007


        This should work in ALL cases: semi-implicit, fully implicit,
        in terms of the increment dw in w.
        For Newton, Picard, linear extrapolation: select u0, u_
        correspondingly before J=derivative() and ufl.replace() afterwards for
        the correct residuals, F.
        In doing so, ONE form `a` is defined that is used for J and F.
        u: Function w.r.t. which the form is differentiated when building
            the Jacobian (TrialFunction)
        u_: linearization (if any) or nonlinear velocity (u_ = u, in the
            latter case)
        u0: velocity from the previous time step

        1.  linear extrapolation:
            Jacobian:
                u:   new Function(W)
                u_:  from self.w (equal to u0)
            Residual:
                u:   from self.w (current time step)
                u_:  from self.w
                u0:  from self.w
            Note: this is redundant/overhead, as (u-u0) on RHS cancels and
            there is no advantage in considering the increment in u. The
            only reason is compatibility with the nonlinear iteration
            methods in order to have an unified approach for all
            linearization methods.
        2.  Newton (fully implicit):
            Jacobian:
                u:  self.w  (current iteration)
                u_: self.w  (u_ == u)
            Residual:
                u:  self.w
                u_: self.w
                u0: previous solution, self.w0
        3.  Picard (fixed point)
            Jacobian:
                u:  new local Function(W)
                u_: from self.w (current iteration)
            Residual:
                u:  from self.w
                u_: from self.w
                u0: previous solution, self.w0
        '''
        nonlinear_method = (self.options['timemarching']['monolithic']
                            ['nonlinear']['method'])
        if nonlinear_method in ('snes', 'snes_manual'):
            nonlinear_method = 'newton'

        self._timescheme = 'fractionalstep'

        rho = self.rho
        mu = self.mu
        # dt = Constant(self.options['timemarching']['dt'])
        dt = self.dt

        (v, q) = TestFunctions(self.W)
        if nonlinear_method in ('constant_extrapolation',
                                'linear_extrapolation'):
            # linear extrapolation
            w = Function(self.W)
            w_ = self.w
            w0 = self.w
            replace_w = {w: w_}
        elif nonlinear_method == 'newton':
            # Newton's method
            w = self.w
            w_ = self.w
            w0 = Function(self.W)
            self.w0 = w0
            replace_w = dict()
        elif nonlinear_method == 'picard':
            # Picard iteration
            w = Function(self.W)
            w_ = self.w
            w0 = Function(self.W)
            self.w0 = w0
            replace_w = {w: w_}
        (u, p) = split(w)
        (u_, p_) = split(w_)
        (u0, p0) = split(w0)

        if nonlinear_method == 'linear_extrapolation':
            # w1/u1: solution of t_{n-1}; w0/u0: t_n
            self.w1 = w0.copy(deepcopy=True)
            u1, p1 = split(self.w1)
            dt0 = Constant(dt)
            self.dt0 = dt0
            # u_c = 2*u0 - u1
            # u_c = 1./dt*((dt + dt0)*u0 - dt*u1)
            u_ = (1. + self.dt/dt0)*u0 - self.dt/dt0*u1

        def eps(u):
            return sym(grad(u))

        def a_diff(u, v):
            if self.options['fem']['strain_symmetric']:
                return inner(2*mu*eps(u), eps(v))*dx
            else:
                return inner(mu*grad(u), grad(v))*dx

        def a_conv(u, u_, v):
            ac = rho*dot(grad(u)*u_, v)*dx
            if self.options['fem']['convection_skew_symmetric']:
                ac += 0.5*rho*div(u_)*dot(u, v)*dx
            return ac

        theta = 1 - np.sqrt(2)/2
        alpha = (1 - 2*theta)/(1 - theta)
        # stage 1:
        theta1 = Constant(alpha*theta)
        theta2 = Constant(theta)
        theta3 = Constant((1 - alpha)*theta)
        self.fs_theta = [theta1, theta2, theta3]

        residual = (
            rho/dt*(u - u0) + theta1*(grad(u)*u_ - div(mu*grad(u))) +
            theta2*grad(p) + theta3*(grad(u0)*u0 - div(mu*grad(u0)))
        )

        # NOTE: stab.form is modified to use theta2/theta1*grad(c) and tau_m
        #   uses 1/(dt*theta1)
        a_stab = sum(self.stabilization(u, u, u_, u0, p, residual=residual))
        a = (
            rho/dt*dot(u, v)*dx
            + theta1*(a_diff(u, v) + a_conv(u, u_, v) + a_stab)
            - theta2*p*div(v)*dx
            + q*div(u)*dx
        )
        L = rho/dt*dot(u0, v)*dx - theta3*(a_diff(u0, v) + a_conv(u0, u0, v))

        a_bc, L_bc = self._form_weak_bcs(w)
        a_bfs = self._backflowstab((u, p), (u_, p_))

        a += sum(a_bc + a_bfs)
        L += sum(L_bc)
        a = a - L

        J = derivative(a, w)
        F = ufl.replace(a, replace_w)

        self.ls_form = [F, J]

        return self

    def form_generalized_midpoint(self):
        r''' Generalized midpoint scheme [DP03]_, [Lay08]_. Similar to what is
        usually called the one step-theta schemes, but the convection term is
        considered as (y \in [0,1])

            u(t_{n+y}).grad(u(t_{n+y})),                                   (1)
            with u(t_{n+y}) ~ y*u(t_{n+1}) + (1-y)*u(t_n)

        instead of

            [y*u(t_n).grad(u(t_n)) + (1-y)*u(t_{n+1}).grad(u(t_{n+1}))]    (2)

        For y=1/2, (2) gives the standard ("two leg") trapezoidal rule (TR)
        (Crank-Nicolson). (1) is known as the midpoint rule, also "one leg"
        TR [Lay08]_ or Leapfrog. For y=1/2, both methods are of second order.

        .. [DP03] Dettmer, W., & Perić, D. (2003). An analysis of the time
           integration algorithms for the finite element solutions of
           incompressible Navier–Stokes equations based on a stabilised
           formulation. Computer Methods in Applied Mechanics and Engineering,
           192(9–10), 1177–1226. https://doi.org/10.1016/S0045-7825(02)00603-5

        .. [Lay08] Layton, W. (2008). Introduction to the numerical analysis of
           incompressible viscous flows. SIAM.

        The Jacobian of the variational form (in the linear and the nonlinear
        case) is obtained by means of the derivative() DOLFIN function. The
        solver methods are written in terms of the *incremental* dw = (du, dp),
        rather than for the new iteration (w_{n+1}^{k+1}) itself. This
        procedure results in a straight forward implementation of the Newton's
        method and in a slightly more cumbersome---but fully compatible---
        algorithm for the linearized solvers. See [Kay+10]_ for examples of the
        TR/AB2 method written in terms of the increment 'du'.

        .. [Kay+10] Kay, D. A., Gresho, P. M., Griffiths, D. F., & Silvester,
           D. J. (2010). Adaptive time-stepping for incompressible flow Part
           II: Navier–Stokes equations. SIAM Journal on Scientific Computing,
           32(1), 111–128.

        The following treatment should work in ALL cases: semi-implicit, fully
        implicit, in terms of the increment dw in w.
        For Newton, Picard, linear extrapolation: select u0, u_
        correspondingly before J=derivative() and ufl.replace() afterwards for
        the correct residuals, F.
        In doing so, ONE form `a` is defined that is used for J and F.
        u: Function w.r.t. which the form is differentiated when building
            the Jacobian (TrialFunction)
        u_: linearization (if any) or nonlinear velocity (u_ = u, in the
            latter case)
        u0: velocity from the previous time step
        [u_mid:  theta*u_{n+1} + (1-theta)*u_n, i.e.: theta*u_ + (1-theta)*u0]

        1.  linear extrapolation:
            Jacobian:
                u:   new Function(W)
                u_:  from self.w (equal to u0)
            Residual:
                u:   from self.w (current time step)
                u_:  from self.w
                u0:  from self.w
            Note: this is redundant/overhead, as (u-u0) on RHS cancels and
            there is no advantage in considering the increment in u. The
            only reason is compatibility with the nonlinear iteration
            methods in order to have an unified approach for all
            linearization methods.
        2.  Newton (fully implicit):
            Jacobian:
                u:  self.w  (current iteration)
                u_: self.w  (u_ == u)
            Residual:
                u:  self.w
                u_: self.w
                u0: previous solution, self.w0
        3.  Picard (fixed point)
            Jacobian:
                u:  new local Function(W)
                u_: from self.w (current iteration)
            Residual:
                u:  from self.w
                u_: from self.w
                u0: previous solution, self.w0
        '''
        theta = self.options['timemarching']['monolithic']['theta']
        if not (theta >= 0 and theta <= 1):
            raise Exception('GMP Theta allowed interval: [0, 1], is: {}'
                            .format(theta))

        self._timescheme = 'generalized_midpoint'

        nonlinear_method = (self.options['timemarching']['monolithic']
                            ['nonlinear']['method'])
        if nonlinear_method in ('snes', 'snes_manual'):
            nonlinear_method = 'newton'

        #
        if nonlinear_method in ('constant_extrapolation',
                                'linear_extrapolation'):
            # linear/constant extrapolation
            w = Function(self.W)
            w_ = self.w
            w0 = self.w
            replace_w = {w: w_}
        elif nonlinear_method == 'newton':
            # Newton's method
            w = self.w
            w_ = self.w
            w0 = Function(self.W)
            self.w0 = w0
            replace_w = dict()
        elif nonlinear_method == 'picard':
            # Picard iteration
            w = Function(self.W)
            w_ = self.w
            w0 = Function(self.W)
            self.w0 = w0
            replace_w = {w: w_}
        else:
            raise Exception('Linearization \'{}\' not available'.format(
                nonlinear_method))

        (u, p) = split(w)
        (u_, p_) = split(w_)
        (u0, p0) = split(w0)

        # define midpoint velocity
        # "implicit" Newton iteration k+1
        u_mid = theta*u + (1 - theta)*u0
        # p_mid = theta*p + (1 - theta)*p0
        # possibly linearized: (Newton iteration k)
        u_mid_ = theta*u_ + (1 - theta)*u0

        dt = self.dt
        if nonlinear_method == 'linear_extrapolation':
            # w1/u1: solution of t_{n-1}; w0/u0: t_n
            self.w1 = w0.copy(deepcopy=True)
            u1, p1 = split(self.w1)
            dt0 = Constant(dt)
            self.dt0 = dt0
            # Newton backward polynomial linear extrapolation (cf.
            # Kay2010_Adaptive, Turek1999, ...)
            # # u_c = 2*u0 - u1
            # u_c = (1. + dt/dt0)*u0 - dt/dt0*u1
            # Adams-Bashforth LE (Simo & Armero 1994, Layton 2008, ...)
            u_c = 1.5*u0 - 0.5*u1
            u_mid_ = u_c

        (v, q) = TestFunctions(self.W)
        rho = self.rho
        mu = self.mu

        def eps(u):
            return sym(grad(u))

        # linear LHS Galerkin terms
        def a_diff(u, v):
            if self.options['fem']['strain_symmetric']:
                return inner(2*mu*eps(u), eps(v))*dx
            else:
                return inner(mu*grad(u), grad(v))*dx

        def a_conv(u, u_, v):
            ac = rho*dot(grad(u)*u_, v)*dx
            if self.options['fem']['convection_skew_symmetric']:
                ac += 0.5*rho*div(u_)*dot(u, v)*dx
            return ac

        a_stab = sum(self.stabilization(u, u_mid, u_mid_, u0, p))

        # NOTE: in the mass balance q*div(u)*dx, use the NEW velocity u_{n+1}!
        a = (
            rho/dt*dot(u - u0, v)*dx
            + a_diff(u_mid, v) + a_conv(u_mid, u_mid_, v) + a_stab
            - p*div(v)*dx + q*div(u)*dx
        )

        a_bc, L_bc = self._form_weak_bcs((u_mid, p))
        a_bfs = self._backflowstab((u_mid, p), (u_mid_, p))
        # a_bc, L_bc = self._form_weak_bcs(w)
        # a_bfs = self._backflowstab(w, w_)

        a += sum(a_bc + a_bfs)
        L = sum(L_bc)
        a -= L

        J = derivative(a, w)
        F = ufl.replace(a, replace_w)

        self.ls_form = [F, J]

        return self

    def stabilization(self, u_new, u, u_, u0, p, residual=None):
        ''' Generate stabilization terms. This function is a factory for
        instances of the stabilization classes.
        Supported options (input.yaml file): combinations of
            - infsup: pspg, pressure-stabilization
            - streamline_diffusion/SUPG
            - GradDiv

        Args:
            u_new    u_{n+1}, unknown velocity
            u0       u_n, previous time step velocity
            u, p     (possibly) implicit solution in residual evaluation
            u_       nonlin or linearized advection velocity
            residual (optional) for non-standard NS residual (e.g. FracStep)

        Returns:
            forms  tuble of forms with stabilization terms
        '''
        forms = []
        infsup = self.options['fem']['stabilization']['monolithic']['infsup']
        if (isinstance(infsup, str) and infsup.lower() ==
                'pressure-stabilization'):
            stab = PressStab(self, residual=residual)
            forms.append(stab.form(u_new, u, u_, u0, p))

        if ((isinstance(infsup, str) and infsup.lower() == 'pspg') or
                self.options['fem']['stabilization']['streamline_diffusion']
                ['enabled']):
            stab = SUPGPSPG(self, residual=residual)
            forms.append(stab.form(u_new, u, u_, u0, p, w=self.w))
            # self._u_supg = stab._u_supg

        return forms


class Stabilization(LoggerBase, ABC):
    ''' Abstract Stabilization base class. '''
    def __init__(self, nsproblem):
        super().__init__()

        self._logging_filehandler = nsproblem._logging_filehandler
        self.logger.addHandler(self._logging_filehandler)
        self.logger.info('Initializing stabilization')

    @abstractmethod
    def form(self, u_new, u, u_, u0, p):
        ''' Abstract method for variational form to be defined by every
        stabilization method. '''
        pass


class SUPGPSPG(Stabilization):
    ''' SUPG/PSPG stabilization class '''
    def __init__(self, nsproblem, residual=None):
        ''' Initializes SUPG/PSPG class.

        Args:
            nsproblem   instance of "nsproblem" (NSUnsteadyProblem)
            residual    (optional) non-standard NS residual (FracStep)
        '''
        super().__init__(nsproblem)

        self.residual = residual
        self.rho = nsproblem.rho
        self.mu = nsproblem.mu
        self.dt = nsproblem.dt
        self.dt0 = nsproblem.dt0
        self.mesh = nsproblem.mesh
        self.W = nsproblem.W

        self._tau_info_printed = False

        self.opt = nsproblem.options
        self.setup(nsproblem)

    def form(self, u_new, u, u_, u0, p, w=None):
        ''' Weak formulation of the SUPG/PSPG stabilization. GradDiv is added
        if corresponding option is set. SUPG, PSPG, GradDiv with residual =
        full gives GLS.

        Args:
            u_new    u_{n+1}, unknown velocity
            u0       u_n, previous time step velocity
            u, p     (possibly) implicit solution in residual evaluation
            u_       nonlin or linearized advection velocity
            w        current solution (for cpp Expressions)
        '''
        v, q = TestFunctions(self.W)
        v_pg = 0
        logstr = 'Stabilization type:'
        if self.opt['fem']['stabilization']['streamline_diffusion']['enabled']:
            v_pg = grad(v)*u_
            logstr += ' SUPG'
        if self.opt['fem']['stabilization']['monolithic']['infsup'] == 'pspg':
            c = self.theta2/self.theta1
            v_pg += c*grad(q)
            logstr += ' PSPG'
        a = self.tau_m(u_, w=w)*dot(v_pg, self.res_m(u_new, u, u_, u0, p))*dx
        if self.opt['fem']['stabilization']['monolithic']['graddiv']:
            a += self.tau_c(u_, w=w)*self.res_c(u)*div(v)*dx
            logstr += ' GradDiv'

        self.logger.info(logstr)
        return a

    def setup(self, nsproblem):
        ''' Set up stabilization methods according to options:
            Select correct form of residual, stabilization parameters, etc.
        '''
        if nsproblem._timescheme == 'fractionalstep':
            self.theta1 = nsproblem.fs_theta[0]
            self.theta2 = nsproblem.fs_theta[1]
            self.theta3 = nsproblem.fs_theta[2]
        elif hasattr(nsproblem, 'theta'):
            self.theta1 = nsproblem.theta
            self.theta2 = Constant(1)
            self.theta3 = Constant(1)
        else:
            self.theta1 = 1
            self.theta2 = Constant(1)
            self.theta3 = Constant(1)

        if self.opt['timemarching']['monolithic']['timescheme'] == 'steady':
            self.kt = 0
        else:
            self.kt = Constant(1./(self.dt*self.theta1))

    def res_m(self, u_new, u, u_, u0, p):
        ''' Residual of the momentum equation.
        Args:
            u_new    u_{n+1}, unknown velocity
            u0       u_n, previous time step velocity
            u, p     (possibly) implicit solution in residual evaluation
            u_       nonlin or linearized advection velocity
        '''
        # opt = self.opt['fem']['stabilization']['streamline_diffusion']
        self.logger.warning('res_m')
        if self.residual:
            self.logger.debug('Using residual given by variational_form()')
            res = self.residual
        else:
            if self.opt['fem']['stabilization']['monolithic']['consistent']:
                self.logger.info('Residual form: consistent')
                res = (self.rho/self.dt*(u_new - u0) + self.rho*grad(u)*u_ +
                       grad(p) - self.mu*div(grad(u)))
            # elif 'residual' in opt and opt['residual'] == 'steady':
            #     self.logger.info('Residual form: steady')
            #     res = self.rho*grad(u)*u_ + grad(p) - self.mu*div(grad(u))
            else:
                self.logger.info('Residual form: convection')
                res = 0
                if (self.opt['fem']['stabilization']['streamline_diffusion']
                        ['enabled']):
                    res = self.rho*grad(u)*u_
                if (self.opt['fem']['stabilization']['monolithic']['infsup']
                        == 'pspg'):
                    res += grad(p)
        return res

    def res_c(self, u):
        ''' Residual of the continuity equation '''
        return div(u)

    def tau_sd(self, w):
        ''' Standard doubly asymptotic streamline diffusion formula, cf.
        [BH82]_.

        .. [BH82] Brooks, A. N., & Hughes, T. J. (1982). Streamline
           upwind/Petrov-Galerkin formulations for convection dominated flows
           with particular emphasis on the incompressible Navier-Stokes
           equations.  Computer Methods in Applied Mechanics and Engineering,
           32(1), 199–259.
        '''
        opt = self.opt['fem']['stabilization']['streamline_diffusion']
        lc = opt['length_scale']

        if not (self.opt['timemarching']['monolithic']['timescheme'] in
                ('gmp', 'generalized_midpoint') and
                self.opt['timemarching']['monolithic']['theta'] == 1 and
                self.opt['timemarching']['monolithic']['nonlinear']['method']
                == 'constant_extrapolation'):
            raise Exception('SD parameter "standard" only supported for '
                            'gmp(theta=1) scheme with constant extrapolation!')

        if lc == 'metric':
            if dolfin.__version__ >= '2018' or (hasattr(dolfin, 'has_pybind11')
                                                and has_pybind11()):
                tau_cpp_code = '''
#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <dolfin/function/GenericFunction.h>
#include <dolfin/function/Expression.h>
#include <dolfin/common/Array.h>

class tau : public dolfin::Expression
{
public:
    std::shared_ptr<dolfin::GenericFunction> viscosity, density;
    std::shared_ptr<dolfin::GenericFunction> u;
    std::shared_ptr<dolfin::GenericFunction> G;

    tau() : dolfin::Expression() { }

    void eval(dolfin::Array<double>& values, const dolfin::Array<double>& x,
              const ufc::cell& c) const
    {
        // Evaluate viscosity at given coordinates
        dolfin::Array<double> mu(viscosity->value_size());
        dolfin::Array<double> rho(density->value_size());
        viscosity->eval(mu, x, c);
        density->eval(rho, x, c);

        double u_norm2 = 0.;
        double u_inner_metric = 0.;
        double v = 0.;
        dolfin::Array<double> w(u->value_size());
        dolfin::Array<double> g(G->value_size());
        u->eval(w, x, c);
        G->eval(g, x, c);
        for (uint i = 0; i < w.size(); ++i)
        {
            v = 0.;
            u_norm2 += w[i]*w[i];
            for (uint j = 0; j < w.size(); ++j)
                v += g[i*w.size()+j]*w[j];
                // u_inner_metric += w[i]*g[i][j]*w[j];
            u_inner_metric += w[i]*v;
        }
        u_inner_metric = sqrt(u_inner_metric);

        // Compute Peclet number and evaluate stabilization parameter
        double Pe = u_norm2/u_inner_metric*rho[0]/mu[0];
        // "critical" formula
        // values[0] = (Pe > 1.0) ? 0.5*h*(1.0 - 1.0/Pe)/u_norm : 0.0;
        // "doubly asymptotic" formula
        double xi = (Pe < 3.0) ? Pe/3. : 1.0;
        // avoid division by zero if norm(u) = 0
        values[0] = (u_inner_metric > 0) ? 1./u_inner_metric*xi : 0;
    }
};

PYBIND11_MODULE(SIGNATURE, m)
{
    py::class_<tau, std::shared_ptr<tau>, dolfin::Expression>
    (m, "tau")
    .def(py::init<>())
    .def_readwrite("G", &tau::G)
    .def_readwrite("u", &tau::u)
    .def_readwrite("density", &tau::density)
    .def_readwrite("viscosity", &tau::viscosity);
}
'''
                tau = CompiledExpression(compile_cpp_code(tau_cpp_code).tau(),
                                         element=FiniteElement(
                                             'DG', self.mesh.ufl_cell(), 0),
                                         domain=self.mesh
                                         )
                tau.viscosity = self.mu.cpp_object()
                tau.density = self.rho.cpp_object()
                tau.u = w.sub(0).cpp_object()
                if not hasattr(self, 'G'):
                    self.G = self._metric()
                tau.G = self.G.cpp_object()

            else:
                tau_cpp_code = '''
class tau : public Expression
{
public:
  std::shared_ptr<GenericFunction> viscosity, density;
  std::shared_ptr<GenericFunction> u;
  std::shared_ptr<GenericFunction> G;

  tau() : Expression() { }

  void eval(Array<double>& values, const Array<double>& x,
            const ufc::cell& c) const
  {
    const std::shared_ptr<const Mesh> mesh = u->function_space()->mesh();
    // Evaluate viscosity at given coordinates
    Array<double> mu(viscosity->value_size());
    Array<double> rho(density->value_size());
    viscosity->eval(mu, x, c);
    density->eval(rho, x, c);

    double u_inner_metric = 0.;
    double u_norm2 = 0.;
    double v = 0.;
    Array<double> w(u->value_size());
    Array<double> g(G->value_size());
    u->eval(w, x, c);
    G->eval(g, x, c);
    for (uint i = 0; i < w.size(); ++i)
    {
        v = 0.;
        u_norm2 += w[i]*w[i];
        for (uint j = 0; j < w.size(); ++j)
            v += g[i*w.size()+j]*w[j];
            // u_inner_metric += w[i]*g[i][j]*w[j];
        u_inner_metric += w[i]*v;
    }
    u_inner_metric = sqrt(u_inner_metric);

    // Note: norm(u)^2/sqrt(<u,Gu>)  =^=  2*norm(u)*h

    // Compute Peclet number and evaluate stabilization parameter
    double Pe = u_norm2/u_inner_metric*rho[0]/mu[0];
    // "critical" formula
    // values[0] = (Pe > 1.0) ? 0.5*h*(1.0 - 1.0/Pe)/u_norm : 0.0;
    // "doubly asymptotic" formula
    double xi = (Pe < 3.0) ? Pe/3. : 1.0;
    // avoid division by zero if norm(u) = 0
    values[0] = (u_inner_metric > 0) ? 1./u_inner_metric*xi : 0;
  }
};
'''
                tau = Expression(tau_cpp_code, element=FiniteElement(
                                 'DG', self.mesh.ufl_cell(), 0),
                                 domain=self.mesh,
                                 mpi_comm=self.mesh.mpi_comm())
                tau.viscosity = self.mu
                tau.density = self.rho
                tau.u = w.sub(0)
                if not hasattr(self, 'G'):
                    self.G = self._metric()
                tau.G = self.G

        elif lc == 'max':
            raise NotImplementedError()
        elif lc == 'average':
            if dolfin.__version__ >= '2018' or (hasattr(dolfin, 'has_pybind11')
                                                and has_pybind11()):

                tau_cpp_code = '''
#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <dolfin/function/GenericFunction.h>
#include <dolfin/function/Expression.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/common/Array.h>
#include <dolfin/function/FunctionSpace.h>

class tau : public dolfin::Expression
{
public:
    std::shared_ptr<dolfin::GenericFunction> viscosity, density;
    std::shared_ptr<dolfin::GenericFunction> u;

    tau() : dolfin::Expression() { }

    void eval(dolfin::Array<double>& values, const dolfin::Array<double>& x,
               const ufc::cell& c) const
    {
        // Get dolfin cell and its diameter
        const std::shared_ptr<const dolfin::Mesh> mesh = u->function_space()->mesh();
        const dolfin::Cell cell(*mesh, c.index);
        double h = cell.h();
        // Evaluate viscosity at given coordinates
        dolfin::Array<double> mu(viscosity->value_size());
        dolfin::Array<double> rho(density->value_size());
        viscosity->eval(mu, x, c);
        density->eval(rho, x, c);

        double u_norm = 0.0;
        dolfin::Array<double> w(u->value_size());
        u->eval(w, x, c);
        for (uint i = 0; i < w.size(); ++i)
            u_norm += w[i]*w[i];
        u_norm = sqrt(u_norm);

        // Compute Peclet number and evaluate stabilization parameter
        double Pe = 0.5*u_norm*h*rho[0]/mu[0];
        // "critical" formula
        // values[0] = (Pe > 1.0) ? 0.5*h*(1.0 - 1.0/Pe)/u_norm : 0.0;
        // "doubly asymptotic" formula
        double xi = (Pe < 3.0) ? Pe/3. : 1.0;
        values[0] = (u_norm > 0) ? 0.5*h/u_norm*xi : 0;
    }
};

PYBIND11_MODULE(SIGNATURE, m)
{
    py::class_<tau, std::shared_ptr<tau>, dolfin::Expression>
    (m, "tau")
    .def(py::init<>())
    .def_readwrite("u", &tau::u)
    .def_readwrite("density", &tau::density)
    .def_readwrite("viscosity", &tau::viscosity);
}
'''

                tau = CompiledExpression(compile_cpp_code(tau_cpp_code).tau(),
                                         element=FiniteElement(
                                             'DG', self.mesh.ufl_cell(), 0),
                                         domain=self.mesh
                                         )
                tau.viscosity = self.mu.cpp_object()
                tau.density = self.rho.cpp_object()
                tau.u = w.sub(0).cpp_object()

            else:
                tau_cpp_code = '''
class tau : public Expression
{
public:
  std::shared_ptr<GenericFunction> viscosity, density;
  std::shared_ptr<GenericFunction> u;

  tau() : Expression() { }

  void eval(Array<double>& values, const Array<double>& x,
            const ufc::cell& c) const
  {
    // Get dolfin cell and its diameter
    // FIXME: Avoid dynamical allocation
    const std::shared_ptr<const Mesh> mesh = u->function_space()->mesh();
    const Cell cell(*mesh, c.index);
    double h = cell.h();
    // Evaluate viscosity at given coordinates
    // FIXME: Avoid dynamical allocation
    Array<double> mu(viscosity->value_size());
    Array<double> rho(density->value_size());
    viscosity->eval(mu, x, c);
    density->eval(rho, x, c);
    // Compute l2 norm of velocity
    double u_norm = 0.0;
    // FIXME: Avoid dynamical allocation
    Array<double> w(u->value_size());
    u->eval(w, x, c);
    for (uint i = 0; i < w.size(); ++i)
        u_norm += w[i]*w[i];
    u_norm = sqrt(u_norm);

    // Compute Peclet number and evaluate stabilization parameter
    double Pe = 0.5*u_norm*h*rho[0]/mu[0];
    // "critical" formula
    // values[0] = (Pe > 1.0) ? 0.5*h*(1.0 - 1.0/Pe)/u_norm : 0.0;
    // "doubly asymptotic" formula
    double xi = (Pe < 3.0) ? Pe/3. : 1.0;
    values[0] = (u_norm > 0) ? 0.5*h/u_norm*xi : 0.;
  }
};
'''
                tau = Expression(tau_cpp_code,
                                 element=FiniteElement(
                                     'DG', self.mesh.ufl_cell(), 0),
                                 domain=self.mesh,
                                 mpi_comm=self.mesh.mpi_comm())
                tau.viscosity = self.mu
                tau.density = self.rho
                # self._u_supg = Function(self.W.sub(0).collapse())
                tau.u = w.sub(0)

        else:
            raise Exception('Streamline diffusion length scale "{}" not '
                            'recognized'.format(lc))

        if self._tau_info_printed is False:
            self.logger.info('SUPG parameter: doubly asymptotic, lc = {}'
                             .format(lc))
            self._tau_info_printed = True

        return tau

    def tau_shakib(self, u):
        ''' SUPG/GLS Stabilization parameter, after [SH91]_.
        Anisotropic/metric based version, see [Baz+07]_, [FD15]_.

        .. [SH91] Shakib, F., & Hughes, T. J. R. (1991). A new finite element
           formulation for computational fluid dynamics: IX. Fourier analysis
           of space-time Galerkin/least-squares algorithms. Computer Methods in
           Applied Mechanics and Engineering, 87(1), 35–58.
           https://doi.org/10.1016/0045-7825(91)90145-V

        .. [Baz+07] Bazilevs, Y., Calo, V. M., Cottrell, J. A., Hughes, T. J.
           R., Reali, A., & Scovazzi, G. (2007). Variational multiscale
           residual-based turbulence modeling for large eddy simulation of
           incompressible flows.  Computer Methods in Applied Mechanics and
           Engineering, 197(1–4), 173–201.
           https://doi.org/10.1016/j.cma.2007.07.016

        .. [FD15] Forti, D., & Dedè, L. (2015). Semi-implicit BDF time
           discretization of the Navier–Stokes equations with VMS-LES modeling
           in a High Performance Computing framework. Computers & Fluids, 117,
           168–182.
        '''
        opt = self.opt['fem']['stabilization']['streamline_diffusion']
        lc = opt['length_scale']

        rho, mu = self.rho, self.mu
        Cinv = opt['Cinv']

        if lc == 'metric':

            Cinv = Constant(30) if not Cinv else Constant(Cinv)

            if not hasattr(self, 'G'):
                self.G = self._metric()
            G = self.G

        elif lc in ('max', 'average'):

            if lc == 'max':
                h = MaxCellEdgeLength(self.mesh)
            elif lc == 'average':
                h = CellDiameter(self.mesh)
            Cinv = Constant(12) if not Cinv else Constant(Cinv)

        else:
            raise Exception('Streamline diffusion length scale "{}" not '
                            'recognized'.format(lc))

        if not ('parameter_element_constant' in opt and
                opt['parameter_element_constant']):

            if lc == 'metric':
                tau = (self.kt**2 + dot(u, G*u) +
                       Cinv*(mu/rho)**2*inner(G, G))**(-0.5)

            else:
                tau = ((self.kt)**2 + (2/h)**2*dot(u, u) +
                       (Cinv*mu/rho/h**2)**2)**(-0.5)

        else:
            if dolfin.__version__ >= '2018' or (hasattr(dolfin, 'has_pybind11')
                                                and has_pybind11()):
                if lc == 'metric':
                    tau_cpp_code = '''
#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <dolfin/function/GenericFunction.h>
#include <dolfin/function/Expression.h>
#include <dolfin/common/Array.h>

class tau : public dolfin::Expression
{
public:
    std::shared_ptr<dolfin::GenericFunction> viscosity, density, k, cinv;
    std::shared_ptr<dolfin::GenericFunction> u;
    std::shared_ptr<dolfin::GenericFunction> G;

    tau() : dolfin::Expression() { }

    void eval(dolfin::Array<double>& values, const dolfin::Array<double>& x,
              const ufc::cell& c) const
    {
//        const std::shared_ptr<const Mesh> mesh = u->function_space()->mesh();
        dolfin::Array<double> mu(viscosity->value_size());
        dolfin::Array<double> rho(density->value_size());
        dolfin::Array<double> dt_inv(k->value_size());
        dolfin::Array<double> Cinv(cinv->value_size());
        viscosity->eval(mu, x, c);
        density->eval(rho, x, c);
        k->eval(dt_inv, x, c);
        cinv->eval(Cinv, x, c);

        // Compute l2 norm of velocity
        double u_inner_metric = 0.;
        double v = 0.;
        dolfin::Array<double> w(u->value_size());
        dolfin::Array<double> g(G->value_size());
        u->eval(w, x, c);
        G->eval(g, x, c);
        for (uint i = 0; i < w.size(); ++i)
        {
            v = 0.;
            for (uint j = 0; j < w.size(); ++j)
                v += g[i*w.size()+j]*w[j];
                // u_inner += w[i]*g[i][j]*w[j];
            u_inner_metric += w[i]*v;
        }

        double gg = 0.;
        for (uint i = 0; i < g.size(); ++i)
            gg += g[i]*g[i];

        values[0] = 1./sqrt(pow(dt_inv[0], 2) + \
                            u_inner_metric + \
                            Cinv[0]*pow(mu[0]/rho[0], 2)*gg);
    }
};

PYBIND11_MODULE(SIGNATURE, m)
{
    py::class_<tau, std::shared_ptr<tau>, dolfin::Expression>
    (m, "tau")
    .def(py::init<>())
    .def_readwrite("G", &tau::G)
    .def_readwrite("u", &tau::u)
    .def_readwrite("density", &tau::density)
    .def_readwrite("viscosity", &tau::viscosity)
    .def_readwrite("k", &tau::k)
    .def_readwrite("cinv", &tau::cinv);
}
'''
                elif lc == 'average':
                    tau_cpp_code = '''
#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <dolfin/function/GenericFunction.h>
#include <dolfin/function/Expression.h>
#include <dolfin/common/Array.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Cell.h>

class tau : public dolfin::Expression
{
public:
    std::shared_ptr<dolfin::GenericFunction> viscosity, density, k, cinv;
    std::shared_ptr<dolfin::GenericFunction> u;

    tau() : dolfin::Expression() { }

    void eval(dolfin::Array<double>& values, const dolfin::Array<double>& x,
              const ufc::cell& c) const
    {
    const std::shared_ptr<const dolfin::Mesh> mesh = u->function_space()->mesh();
        const dolfin::Cell cell(*mesh, c.index);
        double h = cell.h();
        dolfin::Array<double> mu(viscosity->value_size());
        dolfin::Array<double> rho(density->value_size());
        dolfin::Array<double> dt_inv(k->value_size());
        dolfin::Array<double> Cinv(cinv->value_size());
        viscosity->eval(mu, x, c);
        density->eval(rho, x, c);
        k->eval(dt_inv, x, c);
        cinv->eval(Cinv, x, c);

        double u_inner = 0.;
        dolfin::Array<double> w(u->value_size());
        u->eval(w, x, c);
        for (uint i = 0; i < w.size(); ++i)
            u_inner += w[i]*w[i];

        values[0] = 1./sqrt(pow(dt_inv[0], 2) + \
                            pow(2/h, 2)*u_inner + \
                            pow(Cinv[0]*mu[0]/rho[0]/(h*h), 2));
    }
};

PYBIND11_MODULE(SIGNATURE, m)
{
    py::class_<tau, std::shared_ptr<tau>, dolfin::Expression>
    (m, "tau")
    .def(py::init<>())
    .def_readwrite("u", &tau::u)
    .def_readwrite("density", &tau::density)
    .def_readwrite("viscosity", &tau::viscosity)
    .def_readwrite("k", &tau::k)
    .def_readwrite("cinv", &tau::cinv);
}
'''
                elif lc == 'max':
                    raise NotImplementedError('h_max based tau not implemented'
                                              ' for Shakib with parameter_'
                                              'element_constant=True.'
                                              'Use False or h_average')

                tau = CompiledExpression(compile_cpp_code(tau_cpp_code).tau(),
                                         element=FiniteElement(
                                             'DG', self.mesh.ufl_cell(), 0),
                                         domain=self.mesh
                                         # mpi_comm=self.mesh.mpi_comm()
                                         )
                tau.viscosity = self.mu.cpp_object()
                tau.density = self.rho.cpp_object()
                tau.k = self.kt.cpp_object()
                # tau.alpha = alpha.cpp_object()
                # tau.alpha = float(alpha)
                tau.cinv = Cinv.cpp_object()
                tau.u = u.cpp_object()

                if opt['length_scale'] == 'metric':
                    tau.G = G.cpp_object()
            else:
                if lc == 'metric':
                    tau_cpp_code = '''
class tau : public Expression
{
public:
    std::shared_ptr<GenericFunction> viscosity, density, k, cinv;
    std::shared_ptr<GenericFunction> u;
    std::shared_ptr<GenericFunction> G;

    tau() : Expression() { }

    void eval(Array<double>& values, const Array<double>& x,
              const ufc::cell& c) const
    {
        const std::shared_ptr<const Mesh> mesh = u->function_space()->mesh();
        Array<double> mu(viscosity->value_size());
        Array<double> rho(density->value_size());
        Array<double> dt_inv(k->value_size());
        Array<double> Cinv(cinv->value_size());
        viscosity->eval(mu, x, c);
        density->eval(rho, x, c);
        k->eval(dt_inv, x, c);
        cinv->eval(Cinv, x, c);

        // Compute l2 norm of velocity
        double u_inner_metric = 0.;
        double v = 0.;
        Array<double> w(u->value_size());
        Array<double> g(G->value_size());
        u->eval(w, x, c);
        G->eval(g, x, c);
        for (uint i = 0; i < w.size(); ++i)
        {
            v = 0.;
            for (uint j = 0; j < w.size(); ++j)
                v += g[i*w.size()+j]*w[j];
                // u_inner += w[i]*g[i][j]*w[j];
            u_inner_metric += w[i]*v;
        }

        double gg = 0.;
        for (uint i = 0; i < g.size(); ++i)
            gg += g[i]*g[i];

        values[0] = 1./sqrt(pow(dt_inv[0], 2) + \
                            u_inner_metric + \
                            Cinv[0]*pow(mu[0]/rho[0], 2)*gg);
    }
};
'''
                elif lc == 'average':
                    tau_cpp_code = '''
    class tau : public Expression
    {
public:
    std::shared_ptr<GenericFunction> viscosity, density, k, cinv;
    std::shared_ptr<GenericFunction> u;

    tau() : Expression() { }

    void eval(Array<double>& values, const Array<double>& x,
              const ufc::cell& c) const
    {
        const std::shared_ptr<const Mesh> mesh = u->function_space()->mesh();
        const Cell cell(*mesh, c.index);
        double h = cell.h();
        Array<double> mu(viscosity->value_size());
        Array<double> rho(density->value_size());
        Array<double> dt_inv(k->value_size());
        Array<double> Cinv(cinv->value_size());
        viscosity->eval(mu, x, c);
        density->eval(rho, x, c);
        k->eval(dt_inv, x, c);
        cinv->eval(Cinv, x, c);

        double u_inner = 0.;
        Array<double> w(u->value_size());
        u->eval(w, x, c);
        for (uint i = 0; i < w.size(); ++i)
            u_inner += w[i]*w[i];

        values[0] = 1./sqrt(pow(dt_inv[0], 2) + \
                            pow(2/h, 2)*u_inner + \
                            pow(Cinv[0]*mu[0]/rho[0]/(h*h), 2));
    }
};
'''
                elif lc == 'max':
                    raise NotImplementedError('h_max based tau not implemented'
                                              ' for Shakib with parameter_'
                                              'element_constant=True.'
                                              'Use False or h_average')

                tau = Expression(tau_cpp_code,
                                 element=FiniteElement(
                                     'DG', self.mesh.ufl_cell(), 0),
                                 domain=self.mesh,
                                 mpi_comm=self.mesh.mpi_comm()
                                 )
                tau.viscosity = self.mu
                tau.density = self.rho
                tau.k = self.kt
                tau.cinv = Cinv
                tau.u = u

                if opt['length_scale'] == 'metric':
                    tau.G = G

        if self._tau_info_printed is False:
            self.logger.info('SUPG parameter: Shakib, lc = {}, Cinv = {}'
                             .format(lc, float(Cinv)))
            self._tau_info_printed = True

        return tau

    def tau_codina(self, u):
        ''' Codina SUPG parameter [Cod+01]_, [JS08]_.

        .. [Cod+01] Codina, R., Blasco, J., Buscaglia, G. C., & Huerta, A.
           (2001).  Implementation of a stabilized finite element formulation
           for the incompressible Navier–Stokes equations based on a pressure
           gradient projection. International Journal for Numerical Methods in
           Fluids, 37(4), 419–444.

        .. [JS08] John, V., & Schmeyer, E. (2008). Finite element methods for
           time-dependent convection–diffusion–reaction equations with small
           diffusion. Computer Methods in Applied Mechanics and Engineering,
           198(3–4), 475–494.  https://doi.org/10.1016/j.cma.2008.08.016
        '''
        opt = self.opt['fem']['stabilization']['streamline_diffusion']
        if opt['length_scale'] == 'metric':
            raise Exception('Metric length scale not supported for Codina SUPG'
                            ' parameter. Available options: max, average')

        if opt['length_scale'] == 'average':
            h = CellDiameter(self.mesh)
        elif opt['length_scale'] == 'max':
            h = MaxCellEdgeLength(self.mesh)
        else:
            raise Exception('SUPG length scale "{}" not recognized'.format(lc))

        if ('parameter_element_constant' in opt and
                opt['parameter_element_constant']):
            raise NotImplementedError('Codina with elementwise constant '
                                      'parameter '
                                      '(paramenter_element_constant=True) not '
                                      'implemented.')

        rho, mu, k = self.rho, self.mu, self.kt
        tau = 1./(4*mu/rho/h**2 + sqrt(inner(u, u))/h + 1.5*k)

        if self._tau_info_printed is False:
            self.logger.info('SUPG parameter: Codina, lc = {}'
                             .format(opt['length_scale']))
            self._tau_info_printed = True

        return tau

    def tau_klr(self, w):
        ''' SUPG parameter KLR from [1, 2].

        [1] Knopp, T., Lube, G., & Rapin, G. (2002). Stabilized finite element
        methods with shock capturing for advection–diffusion problems. Computer
        Methods in Applied Mechanics and Engineering, 191(27), 2997–3013.
        https://doi.org/10.1016/S0045-7825(02)00222-0

        [2] John, V., & Schmeyer, E. (2008). Finite element methods for
        time-dependent convection–diffusion–reaction equations with small
        diffusion. Computer Methods in Applied Mechanics and Engineering,
        198(3–4), 475–494.  https://doi.org/10.1016/j.cma.2008.08.016
        '''
        if not (self.opt['timemarching']['monolithic']['timescheme'] == 'gmp'
                and self.opt['timemarching']['monolithic']['theta'] == 1 and
                self.opt['timemarching']['monolithic']['nonlinear']['method']
                == 'constant_extrapolation'):
            raise Exception('SD parameter "KLR" only supported for '
                            'gmp(theta=1) scheme with constant extrapolation!')

        opt = self.opt['fem']['stabilization']['streamline_diffusion']

        if not opt['length_scale'] == 'average':
            raise Exception('KLR parameter requires setting '
                            'length_scale=average')

        if not ('parameter_element_constant' in opt
                and opt['parameter_element_constant']):
            self.logger.warning('Ignoring setting '
                                'paramenter_element_constant=False')

        if dolfin.__version__ >= '2018' or (hasattr(dolfin, 'has_pybind11')
                                            and has_pybind11()):

            tau_cpp_code = '''
#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <dolfin/function/GenericFunction.h>
#include <dolfin/function/Expression.h>
#include <dolfin/common/Array.h>

class tau : public dolfin::Expression
{
public:
    std::shared_ptr<dolfin::GenericFunction> viscosity, density, dt_inv, pk;
    std::shared_ptr<dolfin::GenericFunction> u;

    tau() : dolfin::Expression() { }

    void eval(dolfin::Array<double>& values, const dolfin::Array<double>& x,
              const ufc::cell& c) const
    {
        // Get dolfin cell and its diameter
        const std::shared_ptr<const Mesh> mesh = u->function_space()->mesh();
        const Cell cell(*mesh, c.index);
        double h = cell.h();
        // Evaluate viscosity at given coordinates
        dolfin::Array<double> mu(viscosity->value_size());
        dolfin::Array<double> rho(density->value_size());
        dolfin::Array<double> k(dt_inv->value_size());
        dolfin::Array<double> c0(pk->value_size());
        viscosity->eval(mu, x, c);
        pk->eval(c0, x, c);
        density->eval(rho, x, c);
        dt_inv->eval(k, x, c);
        // Compute l2 norm of velocity
        double u_norm = 0.0;
        dolfin::Array<double> w(u->value_size());
        u->eval(w, x, c);
        for (uint i = 0; i < w.size(); ++i)
            u_norm += w[i]*w[i];

        u_norm = sqrt(u_norm);

        double eps = 1.0e-10;

        values[0] = std::min(0.5*h/*u_norm + eps), \
            std::min(2./3./k[0], h*h/(c0[0]*mu[0]/rho[0])));
        // FIXME check factor 2/3
    }
};

PYBIND11_MODULE(SIGNATURE, m)
{
    py::class_<tau, std::shared_ptr<tau>, dolfin::Expression>
    (m, "tau")
    .def(py::init<>())
    .def_readwrite("u", &tau::u)
    .def_readwrite("pk", &tau::pk)
    .def_readwrite("dt_inv", &tau::dt_inv)
    .def_readwrite("density", &tau::density)
    .def_readwrite("viscosity", &tau::viscosity);
}
'''

            tau = CompiledExpression(compile_cpp_code(tau_cpp_code).tau(),
                                     element=FiniteElement(
                                         'DG', self.mesh.ufl_cell(), 0),
                                     domain=self.mesh
                                     )
            tau.viscosity = self.mu.cpp_object()
            tau.density = self.rho.cpp_object()
            tau.dt_inv = self.kt.cpp_object()
            tau.pk = Constant(self.Vi.ufl_element().degree()**2).cpp_object()
            tau.u = w.sub(0).cpp_object()

        else:
            tau_cpp_code = '''
class tau : public Expression
{
public:
  std::shared_ptr<GenericFunction> viscosity, density, pk;
  std::shared_ptr<GenericFunction> u;
  double dt_inv;

  tau() : Expression() { }

  void eval(Array<double>& values, const Array<double>& x,
            const ufc::cell& c) const
  {
    // Get dolfin cell and its diameter
    // FIXME: Avoid dynamical allocation
    const std::shared_ptr<const Mesh> mesh = u->function_space()->mesh();
    const Cell cell(*mesh, c.index);
    double h = cell.h();
    // Evaluate viscosity at given coordinates
    // FIXME: Avoid dynamical allocation
    Array<double> mu(viscosity->value_size());
    Array<double> rho(density->value_size());
    Array<double> c0(pk->value_size());
    viscosity->eval(mu, x, c);
    pk->eval(c0, x, c);
    density->eval(rho, x, c);
    // Compute l2 norm of velocity
    double u_norm = 0.0;
    // FIXME: Avoid dynamical allocation
    Array<double> w(u->value_size());
    u->eval(w, x, c);
    for (uint i = 0; i < w.size(); ++i)
      u_norm += w[i]*w[i];

    u_norm = sqrt(u_norm);

    double eps = 1.0e-10;

    values[0] = std::min(0.5*h/(u_norm + eps), \
                         std::min(1./dt_inv, \
                         h*h/(c0[0]*mu[0]/rho[0])));
  }
};
'''
            tau = Expression(tau_cpp_code,
                             element=FiniteElement('DG', self.mesh.ufl_cell(),
                                                   0),
                             domain=self.mesh,
                             mpi_comm=self.mesh.mpi_comm())
            tau.pk = Constant(w.function_space().ufl_element().degree()**2)
            tau.viscosity = self.mu
            tau.density = self.rho
            tau.dt_inv = float(self.kt)
            tau.u = w.sub(0)

        if self._tau_info_printed is False:
            self.logger.info('SUPG parameter: KLR')
            self._tau_info_printed = True

        return tau

    def tau_m(self, u, w=None):
        ''' Wrapper for different definitions of tau_m, the momentum
        stabilization parameter. '''
        param = (self.opt['fem']['stabilization']['streamline_diffusion']
                 ['parameter'])

        # generator = {
        #     'default': self.tau_sd,
        #     'standard': self.tau_sd,
        #     'shakib': self.tau_shakib,
        #     'codina': self.tau_codina,
        #     'klr': self.tau_klr,
        # }
        # return generator[param](u)
        if param in ('default', 'standard'):
            return self.tau_sd(w)
        elif param == 'shakib':
            return self.tau_shakib(u)
        elif param == 'codina':
            return self.tau_codina(u)
        elif param == 'klr':
            return self.tau_klr(w)

    def tau_c(self, u, w=None):
        ''' Stabilization parameter for Grad-Div stabilization within SUPG
        framework.
        Args:
            var  0: standard, else: simplified
        '''
        opt = self.opt['fem']['stabilization']['streamline_diffusion']
        if opt['length_scale'] == 'metric':
            G = self.G
            tau = 1./(tr(G)*self.tau_m(u, w))
        else:
            if opt['length_scale'] == 'max':
                h = MaxCellEdgeLength(self.mesh)
            else:
                h = CellDiameter(self.mesh)
            tau = h**2/self.tau_m(u, w)

        return tau

    def _metric(self):
        ''' Temp _metric_x wrapper '''
        return self._metric_cpp()

    def _metric_py(self):
        ''' Return metric G of elements '''
        DIM = self.mesh.topology().dim()

        class Metric(Expression):
            def __init__(self, mesh, **kwargs):
                self.mesh = mesh
                self.dim = mesh.topology().dim()

            def eval_cell(self, values, x, cell):
                x = Cell(self.mesh, cell.index).get_vertex_coordinates()
                if self.dim == 2:
                    x0, x1, x2 = x.reshape(3, 2)
                    # [x, = x0*(1-s-t) + x1*s + x2*t = x0 + [x1-x0 x2-x0][s,
                    #  y]                                                 t]
                    F = np.c_[x1 - x0, x2 - x0]   # d(x, y)/d(s, t)
                elif self.dim == 3:
                    x0, x1, x2, x3 = x.reshape(4, 3)
                    F = np.c_[x1 - x0, x2 - x0, x3 - x0]   # d(x, y)/d(s, t)
                Finv = np.linalg.inv(F)
                values[:] = (Finv.T.dot(Finv)).flatten()

            def value_shape(self):
                return (DIM, DIM)

        self.logger.info('Initializing metric tensor')
        t0 = Timer('Z metric tensor')
        t0.start()
        G = Metric(self.mesh, degree=0)
        M = TensorFunctionSpace(self.mesh, 'DG', 0)
        Gh = interpolate(G, M)
        t0.stop()
        return Gh

    def _metric_cpp(self):
        ''' Metric C++ Expression '''
        self.logger.info('SD: initializing metric')
        dim = self.mesh.topology().dim()

        t0 = Timer('Z metric cpp tensor')
        t0.start()

        if dolfin.__version__ >= '2018' or (hasattr(dolfin, 'has_pybind11')
                                            and has_pybind11()):
            cppcode = '''
    #include <pybind11/pybind11.h>
    namespace py = pybind11;

    #include <Eigen/Dense>
    #include <dolfin/function/Expression.h>
    #include <dolfin/mesh/Mesh.h>
    #include <dolfin/mesh/Cell.h>

    class Metric : public dolfin::Expression
    {{
        public:
            std::shared_ptr<const dolfin::Mesh> mesh;

            Metric() : dolfin::Expression({D}, {D}) {{ }}

        void eval(Eigen::Ref<Eigen::VectorXd> values,
                  Eigen::Ref<const Eigen::VectorXd> x, const ufc::cell& c)
                const override
        {{
            const dolfin::Cell cell(*mesh, c.index);
            std::vector<double> coord;
            cell.get_vertex_coordinates(coord);
            Eigen::Map<Eigen::Matrix<double, {D1}, {D}, Eigen::RowMajor> > \
                XC(coord.data());
            Eigen::Matrix<double, {D}, {D}, Eigen::RowMajor> F, Finv;
            for (uint i=0; i < {D}; ++i)
                F.col(i) = XC.row(i+1) - XC.row(0);
            Finv = F.inverse();
            F = Finv.transpose() * Finv;
            values = Eigen::Map<Eigen::VectorXd>(F.data(), {DD});
        }}
    }};

    PYBIND11_MODULE(SIGNATURE, m)
    {{
        py::class_<Metric, std::shared_ptr<Metric>, dolfin::Expression>
        (m, "Metric")
        .def(py::init<>())
        .def_readwrite("mesh", &Metric::mesh);
    }}
    '''
            G = CompiledExpression(
                compile_cpp_code(cppcode.format(D=dim, D1=dim+1,
                                                DD=dim**2)).Metric(),
                element=TensorElement('DG', self.mesh.ufl_cell(), 0)
            )

        else:
            cppcode = '''
    class Metric : public Expression
    {{
        public:
            std::shared_ptr<const Mesh> mesh;

            Metric() : Expression({D}, {D}) {{ }}

        void eval(Eigen::Ref<Eigen::VectorXd> values,
                  Eigen::Ref<const Eigen::VectorXd> x, const ufc::cell& c)
                const override
        {{
            const Cell cell(*mesh, c.index);
            std::vector<double> coord;
            cell.get_vertex_coordinates(coord);
            Eigen::Map<Eigen::Matrix<double, {D1}, {D}, Eigen::RowMajor> > \
                XC(coord.data());
            Eigen::Matrix<double, {D}, {D}, Eigen::RowMajor> F, Finv;
            for (uint i=0; i < {D}; ++i)
                F.col(i) = XC.row(i+1) - XC.row(0);
            Finv = F.inverse();
            F = Finv.transpose() * Finv;
            values = Eigen::Map<Eigen::VectorXd>(F.data(), {DD});
        }}
    }};
    '''
            G = Expression(cppcode.format(D=dim, D1=dim+1, DD=dim**2),
                           element=TensorElement('DG', self.mesh.ufl_cell(),
                                                 0),
                           mpi_comm=self.mesh.mpi_comm())

        G.mesh = self.mesh
        Th = TensorFunctionSpace(self.mesh, 'DG', 0)
        Gh = interpolate(G, Th)
        t0.stop()
        return Gh


class PressStab(Stabilization):
    ''' Standard old-school pressure stabilization with Laplacian
    perturbation of the continuity equation '''
    def __init__(self, nsproblem, residual=None):
        super().__init__(nsproblem)
        epsilon = (nsproblem.options['fem']['stabilization']['monolithic']
                   ['pressure_stab_constant'])
        assert (type(epsilon) in (float, int)) and epsilon >= 0
        self.eps = Constant(epsilon)
        self.mu = nsproblem.mu
        self.rho = nsproblem.rho
        self.h = CellDiameter(nsproblem.mesh)
        self.W = nsproblem.W

    def form(self, u_new, u, u_, u0, p):
        eps = self.eps
        mu = self.mu
        h = self.h
        _, q = TestFunctions(self.W)
        return eps/mu*h**2*dot(grad(p), grad(q))*dx


class StokesProblem(ProblemBase):
    ''' Steady state Stokes problem. '''
    def __init__(self, inputfile=None):
        super().__init__(inputfile)

    def variational_form(self):
        ''' Caller for variational form functions.
        Build forms of bilinear (i.e., linearized) and nonlinear problem.
        Store in self.nls_form (F, J) and self.ls_form (F, J). ls_form is
        extended by Newton linearization terms (for quasi-Newton methods) and
        stored in self.qnls_form.
        '''

        if not self.W:
            self.mixed_functionspace()
        assert self.W and self.w, ('Function space W and function w not'
                                   'initialized.')

        self.bilinear_form()

        return self

    def bilinear_form(self):
        mu = Constant(self.options['dynamic_viscosity'])

        zero = Constant((0.,)*self.ndim)

        (u, p) = TrialFunctions(self.W)
        (v, q) = TestFunctions(self.W)

        a = mu*inner(grad(u), grad(v))*dx - p*div(v)*dx + div(u)*q*dx
        L = dot(zero, v)*dx

        # a_bc, L_bc = self._form_weak_bcs(w)

        J = sum([a])
        F = sum([L])
        self.ls_form = [F, J]
        self.nls_form = [F, J]

        return self


class StokesUnsteadyProblem(ProblemBase):
    ''' Steady state Stokes problem. '''
    def __init__(self, inputfile=None):
        super().__init__(inputfile)
        self.time_expr = []

        # dummy for NSUnsteady
        self.w0 = None
        self.w1 = None
        self.fs_theta = None

    def variational_form(self):
        ''' Caller for variational form functions.
        Build forms of bilinear (i.e., linearized) and nonlinear problem.
        Store in self.nls_form (F, J) and self.ls_form (F, J). ls_form is
        extended by Newton linearization terms (for quasi-Newton methods) and
        stored in self.qnls_form.
        '''

        if not self.W:
            self.mixed_functionspace()
        assert self.W and self.w, ('Function space W and function w not'
                                   'initialized.')

        self.bilinear_form()

    def bilinear_form(self):
        self.set_constants()

        (u, p) = TrialFunctions(self.W)
        (v, q) = TestFunctions(self.W)
        (u0, _) = split(self.w)

        a = mu*inner(grad(u), grad(v))*dx - p*div(v)*dx + div(u)*q*dx
        a += rho/dt*dot(u, v)*dx
        L = rho/dt*dot(u0, v)*dx
        # a_bc, L_bc = self._form_weak_bcs(w)

        J = sum([a])
        F = sum([L])
        self.ls_form = [F, J]
        # self.nls_form = [F, J]


generate = {
    'nssteady': NSSteadyProblem,
    'nsunsteady': NSUnsteadyProblem,
    'stokessteady': StokesProblem,
    'stokesunsteady': StokesUnsteadyProblem
}


def problem(inputfile):
    opt = inout.read_parameters(inputfile)
    if 'stokes' in opt and opt['stokes']:
        problem = 'stokes'
    else:
        problem = 'ns'
    if opt['timemarching']['monolithic']['timescheme'] == 'steady':
        problem += 'steady'
    else:
        problem += 'unsteady'

    # logger.info('Invoking {}'.format(generate[problem].__name__))
    return generate[problem](inputfile)
