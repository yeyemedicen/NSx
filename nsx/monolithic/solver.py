''' Monolithic Navier-Stokes solver module

Author: David Nolte (dnolte@dim.uchile.cl)
Date:   2018-03-19
'''
from dolfin import *
import dolfin
from petsc4py import PETSc
from common import inout, utils
import numpy as np
import pickle
import shutil
import os
from pathlib import Path
import platform
import warnings
from abc import ABC, abstractmethod
from ..logger.logger import LoggerBase


if not dolfin.__version__ >= '2018':
    class MPI(MPI):
        comm_world = mpi_comm_world()

    class TimingClear:
        keep = TimingClear_keep
        clear = TimingClear_clear

    class TimingType:
        wall = TimingType_wall
        user = TimingType_user
        system = TimingType_system


def rank0(func):
    ''' Rank 0 decorator: decorated function "does nothing" if rank > 0 '''
    def inner(*args, **kwargs):
        if MPI.rank(MPI.comm_world) == 0:
            func(*args, **kwargs)
    return inner


class GeneralProblem(NonlinearProblem, LoggerBase):
    ''' This class represents a nonlinear variational problem:
        Find w in W s.t.
            F(w; z) = 0  f.a. z in W^

        GeneralProblem is a subclass of dolfin.NonlinearProblem.

    '''
    def __init__(self, F, w,
                 J=None, bcs=[], time_expr=[], bnds=[], fs_theta=[],
                 nullspace=None):
        ''' Create a nonlinear variational problem for the given form.

        If the Jacobian is not given, it is computed automatically.

        Args:
            F               variational form of residual
            w               solution function
            J (optional)    variational form of Jacobian or its approximation
            bcs (optional)  list of essential boundary conditions
            time_expr, bnds  list of timedependent BC expressions and
                boundaries -> enriched function spaces workaround
            fs_theta        list of fractional step-theta constants
        '''
        LoggerBase.__init__(self)
        NonlinearProblem.__init__(self)
        self.fform = F
        self.w = w
        self.bcs = bcs
        self.time_expr = time_expr
        self.bnds = bnds
        self.nullspace = nullspace

        if fs_theta and not (len(fs_theta) == 3):
            raise Exception('Expected 3 constants for FS-Theta method')
        self.fs_theta = fs_theta

        if J:
            self.jacobian = J
        else:
            self.jacobian = derivative(F, w)

        pass

    def update_bcs(self, t):
        ''' Update time dependent boundary conditions.

        Args:
            t       current time
        '''
        for expr in self.time_expr:
            if hasattr(expr, 't'):
                expr.t = float(t)
                # XXX DIRTIEST OF THE DIRTY HACKS
                if (self.w.function_space().sub(0).sub(0).
                        ufl_element().family() == 'EnrichedElement'):
                    t0 = time()
                    self.logger.warn('Enriched elements: Updating DBC id 0.'
                                     ' Make sure this is the inlet and only t'
                                     ' dependency!')
                    self.bcs[0] = DirichletBC(
                        self.w.function_space().sub(0),
                        project(expr,
                                self.w.function_space().sub(0).collapse(),
                                solver_type='mumps'),
                        self.bnds, 1)
                    self.logger.info('DBC time elapsed: {}'.format(time() -
                                                                   t0))
        pass

    def F(self, b, x=None):
        ''' Assemble residual vector and apply boundary conditions, if given.
        The boundary conditions for the increment are set as the difference
        between the problem's boundary conditions and the current iterate.

        Args:
            b       target PETScVector
            x       current solution PETScVector
        '''
        self.logger.debug('Assembling residual F')
        assemble(self.fform, tensor=b)
        if x:
            [bc.apply(b, x) for bc in self.bcs]
            # as_backend_type(x).update_ghost_values()
            # apparently not necessary
        else:
            [bc.apply(b) for bc in self.bcs]
        # as_backend_type(b).update_ghost_values()
        pass

    def J(self, A, x=None):
        ''' Assemble the Jacobian matrix and set boundary conditions, if given.

        Args:
            A       target PETScMatrix
            x       current solution PETScVector
        '''
        self.logger.debug('Assembling Jacobian J')
        assemble(self.jacobian, tensor=A)
        [bc.apply(A) for bc in self.bcs]
        if self.nullspace:
            # A.set_nullspace(self.nullspace)
            # ??
            A.set_near_nullspace(self.nullspace)
        pass


class MultiStepProblem(NonlinearProblem):
    def __init__(self, Flist, w, Jlist, bcs=[], time_expr=[], bnds=[]):
        ''' Create variational problem for the given forms of
        multi-step/multi-stage time-stepping problem.
        Can be used, e.g., for the Fractional step-Theta method or BDF2, where
        Flist, Jlist are lists of the residual and Jacobian forms of each
        stage/step.
        To set the current step/stage and select the correct forms, call the
        method 'step(i)' (i=0 for the first) before assembly.

        Args:
            F               variational form of residual
            w               solution function
            J               variational form of Jacobian or its approximation
            bcs (optional)  list of essential boundary conditions
        '''
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.addHandler(ch)

        NonlinearProblem.__init__(self)

        self.forms = Flist
        self.jacobians = Jlist

        self.step(0)

    def step(self, i):
        ''' Set step/stage of the multi-step/stage problem.

        Args:
            i (int)     step number (0,...,n-1)
        '''
        self.jacobian = self.jacobians[i]
        self.fform = self.forms[i]

    def F(self, b, x=None):
        ''' Assemble residual vector and apply boundary conditions, if given.
        The boundary conditions for the increment are set as the difference
        between the problem's boundary conditions and the current iterate.

        Args:
            b       target PETScVector
            x       current solution PETScVector
        '''
        self.logger.debug('Assembling residual F')
        assemble(self.fform, tensor=b)
        if x:
            [bc.apply(b, x) for bc in self.bcs]
            # as_backend_type(x).update_ghost_values()
            # apparently not necessary
        else:
            [bc.apply(b) for bc in self.bcs]
        # as_backend_type(b).update_ghost_values()
        pass

    def J(self, A, x=None):
        ''' Assemble the Jacobian matrix and set boundary conditions, if given.

        Args:
            A       target PETScMatrix
            x       current solution PETScVector
        '''
        self.logger.debug('Assembling Jacobian J')
        assemble(self.jacobian, tensor=A)
        [bc.apply(A) for bc in self.bcs]
        pass


class AitkenAccelerator(LoggerBase):
    ''' Class for Aitken accelerator, based on velocity increment. Stores
    one previous velocity increment and relaxation parameter and calculates
    new omega.

    References:

    .. [IT69] Irons, B. M., & Tuck, R. C. (1969). A version of the Aitken
       accelerator for computer iteration. International Journal for Numerical
       Methods in Engineering, 1(3), 275–277.

    .. [KW08] Küttler, U., & Wall, W. A. (2008). Fixed-point fluid–structure
       interaction solvers with dynamic relaxation. Computational Mechanics,
       43(1), 61–72.
    '''
    def __init__(self, options):
        ''' Initialize Aitken accelerator.

        Args:
            options     options dictionary

        Attributes:
            use (int):  switch, 1: enable Aitken for Picard iteration
                                2: enable Aitken at all times
                                0: don't use Aitken acceleration
            picard:     True if picard iteration is used, False otherwise
            omega:      relaxation parameter, default: 1
            du_old:     velocity increment of previous time step
        '''
        # TODO: check dw instead of du
        super().__init__()

        opt = options['timemarching']['monolithic']['nonlinear']
        self.use = opt['use_aitken']
        # legacy bool switch compatibility
        if self.use is True:
            self.use = 1
        if self.use is False:
            self.use = 0

        self.picard = bool(opt['method'] == 'picard')

        self.omega = 1.0
        ''' relaxation parameter '''
        self.du_old = None
        self.du = None
        ''' old residual '''
        pass

    def relax(self, dw, init=False):
        ''' Compute relaxation parameter, if use > 0 and not first
        iteration (self.dw_old already set).

        Note: keeping du as an instance variable and using assign() on each
        iteration halves the (fairly short) computation time.

        Args:
            dw              current solution increment (dolfin Function)
            init (bool)     True/False switch indicates initialization phase

        Return:
            a1 (float)
        '''
        self.logger.debug('Performing Aitken relaxation')
        if self.use == 2 or (self.use and self.picard) or (self.use and init):
            if self.du_old and self.du:
                assign(self.du, dw.sub(0))
                self.omega *= -(
                    np.dot(
                        self.du_old.vector().array(),
                        (self.du.vector() - self.du_old.vector()).array()
                    ) / norm(self.du.vector() - self.du_old.vector(), 'l2')**2
                )
            else:
                # initialize
                (self.du, _) = dw.split(deepcopy=True)
                self.du_old = Function(self.du.function_space())

            assign(self.du_old, self.du)

        return self.omega


class PETScSolver(LoggerBase):
    ''' Solver class that handles preconditioned Krylov solvers.
        Should work for Stokes problems, NS steady and unsteady, within any
        solver (newton, picard, snes)
    '''
    # TODO: snes interface (ksp = x, setPC = y)
    # TODO: not necessary to setup solver at every time step? (overhead
    #       measureable?)

    def __init__(self, generalproblem, options, logging_fh=None):
        super().__init__()
        self._logging_filehandler = logging_fh
        if self._logging_filehandler:
            self.logger.addHandler(self._logging_filehandler)

        self.options_global = options

        self.residuals = []
        self.iterations = None
        self.timing = {'ksp': 0, 'pc': 0}

        if self._use_petsc():
            self.logger.info('Initializing PETSc Krylov solver')
            inputfile = options['linear_solver']['inputfile']
            self.param = inout.read_parameters(inputfile)
            self._reuse_ksp = False
            # this is not necessary !
            self._reuse_Pmat = False    # test this
            self.generalproblem = generalproblem
            self._set_options()
            self.create_ksp()
            self.create_pc()
        else:
            if not options['linear_solver']['method']:
                raise Exception('Specify inbuilt solver (lu, mumps) or input '
                                'file for PETSc!')
            self.logger.info('using inbuilt linear solver')
            self.logger.warn('prefer MUMPS via PETSc interface!')

    def __del__(self):
        ''' Clean up PETScOptions '''
        self._unset_options()
        self.close_logs()

    def _set_options(self):
        self.pc_name = self.param['config_name']
        self.petsc_options = [s.split() for s in self.param['petsc_options']]
        self.logger.info('  PC: {0}'.format(self.pc_name))
        self.logger.info('  Setting PETScOptions:')
        for popt in self.petsc_options:
            self.logger.info('  -' + ' '.join(popt))
            PETScOptions.set(*popt)

    def _unset_options(self):
        ''' Unset all petsc options.
            ATTENTION: not all options are actually deleted. E.g.,
            pc_ml_Threshold survives!
        '''
        self.logger.info('  Clearing PETScOptions')
        if hasattr(self, 'petsc_options'):
            for popt in self.petsc_options:
                PETScOptions.clear(popt[0])
                self.logger.debug('  clear {}'.format(popt[0]))

    def create_ksp(self):
        ''' Create KSP instance and set options '''
        self.logger.info('  Creating KSP from options')
        self.ksp = PETSc.KSP().create()
        if not hasattr(self, 'petsc_options') or self.petsc_options is None:
            self._set_options()
        self.ksp.setFromOptions()

    def create_pc(self):
        self.fstype = self._get_fieldsplit_type()
        # assert self.fstype, 'fieldsplit type not recognized'
        t0 = Timer('PETScSolver Build PC')
        self.logger.info('  Creating PC of type: {}'.format(self.fstype))
        if self.fstype is None:
            self.logger.info('No FieldSplit. Creating solver from options.')
        elif self.fstype in ('SCHUR_USER_DIAG', 'SCHUR_USER_FULL',
                             'SCHUR_USER_TRI'):
            self._create_user_schurPC()
        elif self.fstype in ('SCHUR_LSC', 'SCHUR'):
            self._create_alg_schurPC()
        elif self.fstype in ('ADD', 'MULT'):
            self._create_Pmat()
        elif self.fstype == 'MG_FIELDSPLIT':
            pass
            # self.logger.info('Setting FieldSplit IS for coupled MG')
            # self._set_fieldsplit_IS()
        elif self.fstype == 'LU':
            self._create_direct_solver()
        else:
            raise Exception('Fieldsplit type not supported!  {}'
                            .format(self.fstype))
        self.timing['pc'] = t0.stop()
        self.ksp.getPC().setReusePreconditioner(self._reuse_Pmat)

    def solve(self, A, x, b):
        ''' Wrap PETSc and "inbuilt" FEniCS solvers.

        Args:
            A   PETScMatrix
            x   PETScVector
            b   PETScVector
        '''
        if self._use_petsc():
            self.solve_ksp(A, x, b)
        else:
            solve(A, x, b, self.options_global['linear_solver']['method'])

    def solve_ksp(self, A, x, b):
        ''' Solve the system Ax = b.
            1. Decide which operators have to be set (A or (A, P))
            2. setUp
            3. solve
            4. clear settings
        '''
        self.logger.debug('KSP: Setting up')
        if self.fstype in ('ADD', 'MULT'):
            # or just keep it constant?
            self._assemble_Pmat()
            self.ksp.setOperators(A.mat(), self.P.mat())
        else:
            self.ksp.setOperators(A.mat())

        self.ksp.setConvergenceHistory()
        self.ksp.setUp()

        # self._mg_levels_setup()

        self.logger.debug('KSP: Solving Ax=b')
        t0 = Timer('PETScSolver SOLVE ')
        self.ksp.solve(as_backend_type(b).vec(), as_backend_type(x).vec())
        as_backend_type(x).update_ghost_values()
        self.timing['ksp'] = t0.stop()
        self.iterations = self.ksp.getIterationNumber()
        self.residuals = self.ksp.getConvergenceHistory()
        self.conv_reason = self.ksp.getConvergedReason()

        if len(self.residuals) > 0:
            convstr = 'CONVERGED' if self.conv_reason > 0 else 'DIVERGED'
            self.logger.info('{c} ({cr}) after {it} iterations. Residual: '
                             '{res}'.format(c=convstr, cr=self.conv_reason,
                                            it=self.iterations,
                                            res=self.residuals[-1]))

    def _mg_levels_setup(self):
        ''' Settigns for MG levels '''
        raise NotImplementedError('UNFINISHED WIP, DONT USE _mg_levels_setup!')
        print(self.fstype)
        if not self.fstype == 'MG_FIELDSPLIT':
            return

        self.logger.info('Setting FieldSplit IS for coupled MG')
        pc = self.ksp.getPC()
        nlevels = pc.getMGLevels()
        print(nlevels)
        for i in range(nlevels):
            ksp_level_i = pc.getMGSmoother(i)
            pc_level_i = ksp_level_i.getPC()
            print(pc_level_i.getType())
            self._set_fieldsplit_IS(ksp_level_i)

    def _set_fieldsplit_IS(self, ksp=None):
        ''' Sets and returns index fields for field split. For unstabilized
        saddle point problems, PETSc can detect the blocks by itself (zero
        diagonal in C=0).
            [  A  B^T ]
            [ -B  C   ]
        For stabilized problems with C != 0, the index sets need to be given
        according to the dofs of the u,p subspaces of the mixed function space,
        W. In the case of a user-defined preconditioner matrix for the Schur
        complement, the indices need to be known, too (are returned by this
        function.

        Args:
            ksp (optional)  PETSc KSP object, uses self.ksp if not given
        Returns:
            (is0, is1)  Tuple of PETSc index fields
        '''
        if not ksp:
            ksp = self.ksp
        W = self.generalproblem.w.function_space()
        pc = ksp.getPC()
        is0 = PETSc.IS().createGeneral(W.sub(0).dofmap().dofs())
        is1 = PETSc.IS().createGeneral(W.sub(1).dofmap().dofs())
        fields = [('0', is0), ('1', is1)]
        pc.setFieldSplitIS(*fields)
        return is0, is1

    def _create_user_schurPC(self):
        assert 'SCHUR_USER' in self.fstype, (
            'create_SchurPC called unexpectedly')
        W = self.generalproblem.w.function_space()
        if 'reuse_SchurPC' in self.param and self.param['reuse_SchurPC']:
            if hasattr(self, 'Sp') and self.Sp:
                return
            else:
                raise Exception('Cannot use stored Sp -- does not exist!')

        if 'SchurPC' in self.param and not self.param['SchurPC'] == 'PMM':
            raise Exception('Specified SchurPC matrix not recognized!')
        else:
            # build pressure mass matrix
            self.logger.info('  Building Sp = pressure mass matrix')
            if 'fluid' in self.options_global:
                mu = self.options_global['fluid']['dynamic_viscosity']
            else:
                mu = self.options_global['dynamic_viscosity']
            mu_inv = Constant(1./mu)
            (_, p) = TrialFunctions(W)
            (_, q) = TestFunctions(W)
            aschur = mu_inv*p*q*dx
        if self.fstype == 'SCHUR_USER_DIAG':
            aschur *= -1

        (is0, is1) = self._set_fieldsplit_IS()
        Sp = PETScMatrix()
        assemble(aschur, tensor=Sp)
        pc = self.ksp.getPC()
        pc.setFieldSplitSchurPreType(PETSc.PC.SchurPreType.USER,
                                     Sp.mat().createSubMatrix(is1, is1))
        # DEBUG: save Sp matrix in CSR format
        # (ai, aj, av) = Sp.mat().createSubMatrix(is1, is1).getValuesCSR()
        # np.savez('Sp_mat', ai=ai, aj=aj, av=av)

    def _create_Pmat(self):
        ''' Create preconditioner matrix P for complete system matrix A '''
        assert self.fstype in ('ADD', 'MULT')
        self.logger.info('  Building Pmat = pressure mass matrix')
        W = self.generalproblem.w.function_space()
        if 'fluid' in self.options_global:
            mu = self.options_global['fluid']['dynamic_viscosity']
        else:
            mu = self.options_global['dynamic_viscosity']
        mu_inv = Constant(1./mu)
        (_, p) = TrialFunctions(W)
        (_, q) = TestFunctions(W)
        self.apform = self.generalproblem.jacobian + mu_inv*p*q*dx
        self.P = PETScMatrix()

    def _assemble_Pmat(self):
        ''' Assemble the preconditioner matrix P from form defined by
        _create_Pmat.
        '''
        # if 'stokes' in self.options_global and self.options_global['stokes']:
        #     if hasattr(self, '_P_assembled') and self._P_assembled is True:
        #         return
        if not (hasattr(self, '_reuse_Pmat') and self._reuse_Pmat):
            assemble(self.apform, tensor=self.P)
            [bc.apply(self.P) for bc in self.generalproblem.bcs]

    def _create_direct_solver(self):
        assert self.fstype == 'LU'
        self.logger.info('  Using direct solver')

    def _create_alg_schurPC(self):
        ''' Schur preconditioner is built automatically by PETSc. Set
        fieldsplit indes sets for the case of C != 0. '''
        self.logger.info('  {} PC built automatically from A'
                         .format(self.fstype))
        self._set_fieldsplit_IS()
        pass

    def _get_fieldsplit_type(self):
        ''' Detects fieldsplit types:
                Schur complement PCs with user defined preconditioner Sp:
                    SCHUR_USER_FULL
                    SCHUR_USER_DIAG
                    SCHUR_USER_TRI (=UPPER/LOWER)
                    SCHUR_USER_LSC *** FIXME
                Schur PCs with automatic (algebraic) preconditioner:
                    SCHUR_LSC
                    SCHUR
                Multiplicative:
                    MULT
                Additive:
                    ADD
                Direct solver:
                    LU

            returns: FieldSplit_Type (str)
        '''
        petsc_opt = self.param['petsc_options']
        if 'pc_type lu' in petsc_opt:
            return 'LU'
        elif 'pc_fieldsplit_type schur' in petsc_opt:
            if 'pc_fieldsplit_schur_precondition user' in petsc_opt:
                if 'fieldsplit_1_pc_type lsc' in petsc_opt:
                    return 'SCHUR_USER_LSC'
                elif 'pc_fieldsplit_schur_fact_type diag' in petsc_opt:
                    return 'SCHUR_USER_DIAG'
                elif 'pc_fieldsplit_schur_fact_type full' in petsc_opt:
                    return 'SCHUR_USER_FULL'
                elif ('pc_fieldsplit_schur_fact_type upper' in petsc_opt or
                      'pc_fieldsplit_schur_fact_type lower' in petsc_opt):
                    return 'SCHUR_USER_TRI'
            elif 'fieldsplit_1_pc_type lsc' in petsc_opt:
                return 'SCHUR_LSC'
            else:
                return 'SCHUR'
        elif 'pc_fieldsplit_type additive' in petsc_opt:
            return 'ADD'
        elif 'pc_fieldsplit_type multiplicative' in petsc_opt:
            return 'MULT'
        elif 'mg_levels_pc_type fieldsplit' in petsc_opt:
            return 'MG_FIELDSPLIT'
        else:
            return None

    def _use_petsc(self):
        return ('inputfile' in self.options_global['linear_solver'] and
                self.options_global['linear_solver']['inputfile'])


class NSSolverBase(LoggerBase, ABC):
    ''' Abstract Navier-Stokes solver class, that the Steady and Unsteady
    solvers are built upon, providing a common interface and overlapping
    functionality.
    '''
    def __init__(self, nsproblem):
        ''' Initialization for all solvers. Receives an instance of NSProblem
        and links the solver to it.

        Args:
            nsproblem       instance of NSProblem
        '''
        super().__init__()
        self._logging_filehandler = nsproblem._logging_filehandler
        if self._logging_filehandler:
            self.logger.addHandler(self._logging_filehandler)

        self.logger.info('Initializing')
        # initialize (or new?)
        self.options = nsproblem.options
        self.inputfile = nsproblem.inputfile
        self.Flin, self.Jlin = nsproblem.ls_form
        self.bcs = nsproblem.bcs
        self.w = nsproblem.w
        self.W = nsproblem.W

        # self._u_supg = nsproblem._u_supg

        self.t = np.longdouble(0)
        self.it = None
        self.residuals = []
        self.iterations_ksp = []
        self.residuals_ksp = []

    @abstractmethod
    def solve(self):
        ''' Abstract solver method '''
        pass

    def monitor(self, b, dw, a1, it=None):
        ''' Monitor convergence of iteration method. Check if convergence
        criterion is met and store residual in self.residuals.

        Args:
            b (PETScVector):    residual vector

        Return:
            converged (bool):   True if converged, False otherwise
        '''
        # TODO: specify this in input file !?
        lp = 'l2'  # l2 vs linf
        residual = norm(b, lp)
        if it is None:
            it = self.it

        energy = None
        opt_nonlin = self.options['timemarching']['monolithic']['nonlinear']
        if opt_nonlin['report'] == 2:
            # XXX this should be a temporary hack
            # resid_form = assemble(self.F)
            # energy = np.dot(resid_form.array(), self.w.vector().array())
            energy = assemble(self.energy_form)
            self.energy.append(energy)

        atol = opt_nonlin['atol']
        rtol = opt_nonlin['rtol']
        stol = opt_nonlin['stol']
        init_atol = (opt_nonlin['init_atol'] if 'init_atol' in opt_nonlin else
                     None)

        dw_norm = norm(dw.vector(), lp)
        init = opt_nonlin['init_steps']

        if hasattr(self, '_init') and self._init and residual <= init_atol:
            converged = True
            reason = 'INIT_ATOL'

        elif residual <= atol:
            converged = True
            reason = 'ATOL'

        # elif residual <= rtol*norm(self.w.vector(), 'l2'):
        #     # FIXME
        #     warnings.warn('In PETSc/SNES, RTOL crit is r0/r_k, not r_k/w_k!')
        #     warnings.warn('My RTOL definition physically meaningless??')
        elif (len(self.residuals) >= init + 2 and
              residual <= rtol*self.residuals[init + 1]):
            converged = True
            reason = 'RTOL'
            warnings.warn(('RTOL checking starts AFTER init_steps for'
                           'comparability with SNES.'))

        elif (dw_norm <= stol*norm(self.w.vector(), lp) and
              it > opt_nonlin['init_steps'] + 1):
            converged = True
            reason = 'STOL'

        else:
            converged = False
            reason = ''

        self.residuals.append(residual)

        if opt_nonlin['report']:
            self.logger.info(
                'it {it}\t{init}\trelax: {a:.3f}\t residual: {r}' '{s} {e}'
                .format(it=it, r=residual, a=a1,
                        init='i' if hasattr(self, '_init') and self._init
                        else '',
                        e=energy if energy else '',
                        s='\t R(u, u): ' if energy else ''))

        return converged, reason

    def build_nullspace(self):
        ''' Build constant pressure nullspace.
        Returns:    nullspace
        '''
        if not self.options['fix_pressure'] == 2:
            return None

        nullspace_basis = self.w.vector().copy()
        self.W.sub(1).dofmap().set(nullspace_basis, 1.0)
        nullspace_basis.apply('insert')
        nullspace = VectorSpaceBasis([nullspace_basis])
        nullspace.orthonormalize()
        return nullspace

    def newtonlike(self, problem, init_steps=None):
        ''' Newton type solver method for full Newton, quasi-Newton, Picard
        iteration, with optional Aitken delta^2 acceleration.

        If called with init_steps != None, used to compute an initial solution
        for a Newton method.

        Computes until maxit = (init_steps, maxit) or convergence criterion
        reached. Convergence is checked by the method monitor. The
        AitkenAccelerator is controlled by the class AitkenAccelerator.

        Args:
            init_steps (int, default None):   if set, number of initial steps

        Return:

        '''

        solver = PETScSolver(problem, self.options, self._logging_filehandler)
        A = PETScMatrix()
        b = PETScVector()
        dw = Function(self.W)

        aitken = AitkenAccelerator(self.options)
        a1 = 1.
        maxit = (self.options['timemarching']['monolithic']['nonlinear']
                 ['maxit'])
        if init_steps is not None:
            maxit = init_steps

        while self.it < maxit:

            # this way, only one call/loop for picard init and newton solver
            # and same AitkenAccelerator can be used!
            # but logic/simplicity not ideal and special treatment for snes
            # necessary
            # solver should be oblivious to problem it is used solving for
            # if init_steps:
            #     self.problem.Jlin(A, self.w.vector())
            #     self.problem.Flin(b, self.w.vector())
            # else:
            #     self.problem.J(A, self.w.vector())
            #     self.problem.F(b, self.w.vector())

            # assemble jacobian and residual
            problem.F(b, self.w.vector())
            problem.J(A, self.w.vector())

            converged, reason = self.monitor(b, dw, a1)
            if converged:
                self.logger.info('CONVERGED to {r} in {i} iterations'.
                                 format(i=self.it+1, r=reason))
                self.converged_reason = 1
                break

            solver.solve(A, dw.vector(), b)

            a1 = aitken.relax(dw, self._init)
            self.w.vector().axpy(-a1, dw.vector())

            self.it += 1

        else:
            if init_steps is None:
                self.logger.info('NOT CONVERGED, MAX_IT REACHED.')
                self.converged_reason = -5

        return self

    def snes_manual(self, problem):
        ''' Solve the nonlinear problem with SNES with the standard options
        set.
        '''
        # TODO: control over snes options!
        # TODO: check out newtonls (linesearch) variants !

        # FIXME: this does not work. PETScOptions are not accepted... (???)
        # linear_solver = PETScSolver(problem, self.options)
        self.logger.info('SNES manual version')

        A = PETScMatrix()
        b = PETScVector()
        x = as_backend_type(self.w.vector())

        snes = PETSc.SNES().create()

        opt = self.options['timemarching']['monolithic']['nonlinear']
        PETScOptions.set('snes_monitor')
        PETScOptions.set('snes_type', 'newtonls')
        PETScOptions.set('snes_linesearch', 'basic')
        PETScOptions.set('snes_converged_reason')
        PETScOptions.set('snes_atol', opt['atol'])
        PETScOptions.set('snes_rtol', opt['rtol'])
        PETScOptions.set('snes_stol', opt['stol'])
        PETScOptions.set('snes_max_it', opt['maxit'])
        PETScOptions.set('ksp_type', 'preonly')
        PETScOptions.set('pc_type', 'lu')
        PETScOptions.set('pc_factor_mat_solver_package', 'mumps')

        snes.setFromOptions()

        # FormFunction = problem.F
        # FormJacobian = problem.J

        def FormFunction(snes, x, b):
            b_ = PETScVector(b)
            x_ = PETScVector(x)
            # w_ = PETScVector(as_backend_type(self.w.vector()).vec())
            w_ = as_backend_type(self.w.vector())

            # took this from
            # https://bitbucket.org/pefarrell/varfrac-solvers/src/9fd3e78ad7b60774baa4dff2f6f7dfa18dda7f92/fracture/petscsnessolver.py?at=master&fileviewer=file-view-default
            # copy necessary for snes internal computations(?)
            # x and w need to be cast to PETsVector x_, w_ in order to update
            # ghost cells
            x_.update_ghost_values()
            x.copy(as_backend_type(self.w.vector()).vec())
            w_.update_ghost_values()

            # self.logger.debug('F(x) residual assembly: BEFORE')
            problem.F(b_, w_)
            # self.logger.debug('F(x) residual assembly: AFTER')
            # self.logger.debug('Residual: {}'.format(norm(b_)))
            # self.logger.debug('norm(dx): {}'.format(norm(x_)))
            # self.logger.debug('norm(w): {}'.format(norm(w_)))

        def FormJacobian(snes, x, J, P):
            J_ = PETScMatrix(J)
            w_ = PETScVector(as_backend_type(self.w.vector()).vec())
            self.logger.debug('J(x) residual assembly: BEFORE')
            problem.J(J_, w_)
            self.logger.debug('J(x) residual assembly: AFTER')

        snes.setFunction(FormFunction, b.vec())
        snes.setJacobian(FormJacobian, A.mat())

        snes.setConvergenceHistory()
        # snes.setKSP(linear_solver.ksp)
        # snes.setType('ncg')
        snes.solve(None, x.vec())

        print(snes.getType())
        # solver = PETScSNESSolver()
        # solver.init(problem, self.w.vector())
        # snes = solver.snes()
        # snes.setFromOptions()
        # snes.solve(None, as_backend_type(self.w.vector()).vec())
        self.residuals = np.hstack((self.residuals,
                                    snes.getConvergenceHistory()[0]))
        self.converged_reason = snes.getConvergedReason()
        print(PETSc.Options().getAll())

        return self

    def snes(self, problem):
        ''' Solve the nonlinear problem with SNES with the standard options
        set.
        '''
        # TODO: control over snes options!
        # TODO: check out newtonls (linesearch) variants !

        # FIXME: works with PETScSolver for mumps but not KSP...?
        self.logger.info('SNES')
        linear_solver = PETScSolver(problem, self.options,
                                    self._logging_filehandler)
        opt = self.options['timemarching']['monolithic']['nonlinear']

        PETScOptions.set('snes_monitor')
        PETScOptions.set('snes_type', 'newtonls')
        PETScOptions.set('snes_converged_reason')
        PETScOptions.set('snes_atol', opt['atol'])
        PETScOptions.set('snes_rtol', opt['rtol'])
        PETScOptions.set('snes_stol', opt['stol'])
        PETScOptions.set('snes_max_it', opt['maxit'])
        # PETScOptions.set('ksp_type', 'preonly')
        # PETScOptions.set('pc_type', 'lu')
        # PETScOptions.set('pc_factor_mat_solver_package', 'mumps')
        solver = PETScSNESSolver()
        solver.init(problem, self.w.vector())
        snes = solver.snes()
        snes.setFromOptions()
        snes.setKSP(linear_solver.ksp)
        snes.setConvergenceHistory()
        snes.solve(None, as_backend_type(self.w.vector()).vec())
        self.residuals = np.hstack((self.residuals,
                                    snes.getConvergenceHistory()[0]))
        self.converged_reason = snes.getConvergedReason()
        self.logger.debug(PETSc.Options().getAll())

        return self

    def solve_ls(self, A, x, b):
        ''' Solve a linear system
                        A*x = b
            directly with the FEniCS interface.

            Args:
                A           FEniCS/PETSc matrix
                x           DOLFIN Function
                b           rhs PETSc vector
        '''
        self.logger.warn('Using "built-in" direct solver. Prefer PETScSolver, '
                         'always faster!')
        solve(A, x, b, self.options['linear_solver']['method'])

        return self

    def reset(self):
        ''' Reset solution w to zero. '''
        self.w.vector().zero()
        self.residuals = []
        if hasattr(self, '_init'):
            self._init = None
        self.it = None

        return self

    def write_checkpoint(self, i):
        ''' Write HDF5 checkpoint of u, p to a ./checkpoints folder.

        Args:
            i   iteration count
        '''
        if not self.options['io']['write_checkpoints']:
            return

        path = (self.options['io']['write_path'] +
                '/checkpoint/{i}/'.format(i=i))
        comm = self.w.function_space().mesh().mpi_comm()

        u, p = self.w.split()
        inout.write_HDF5_data(comm, path + '/u.h5', u, '/u', t=self.t)
        inout.write_HDF5_data(comm, path + '/p.h5', p, '/p', t=self.t)

    def _close_xdmf(self):
        ''' close XDMF Files '''
        if hasattr(self, '_dont_close_xdmf') and self._dont_close_xdmf:
            # a little hacky I admit
            return

        if hasattr(self, '_xdmf_u'):
            del self._xdmf_u
        if hasattr(self, '_xdmf_p'):
            del self._xdmf_p

    def cleanup(self):
        self._close_xdmf()
        self.close_logs()

    def write_xdmf(self):
        ''' Write solution to XDMF files. If file objects have not been
        created, initialize. This works for steady and unsteady solvers with
        timestepping output.
        '''
        if not self.options['io']['write_xdmf']:
            return

        if (not (hasattr(self, '_xdmf_u') or hasattr(self, '_xdmf_p')) or not
                (self._xdmf_u or self._xdmf_p)):
            self._xdmf_u = XDMFFile(self.options['io']['write_path'] +
                                    '/u.xdmf')
            self._xdmf_p = XDMFFile(self.options['io']['write_path'] +
                                    '/p.xdmf')
            self._xdmf_u.parameters['rewrite_function_mesh'] = False
            self._xdmf_p.parameters['rewrite_function_mesh'] = False
        u, p = self.w.split()
        u.rename('u', 'velocity')
        p.rename('p', 'pressure')
        self._xdmf_u.write(u, float(self.t))
        self._xdmf_p.write(p, float(self.t))


class NSUnsteady(NSSolverBase):
    ''' Unsteady Navier-Stokes solver class for solving a given
    **time-dependent** variational problem (object of the NSProblem class).

    Usage:
        ...

    The implemented solvers are:
        - Backward Euler

    # TODO: Fractional step theta solvers
    '''
    def __init__(self, nsproblem, dump_parameters=True):
        ''' Initialize Navier-Stokes solver. Call abstract class initializer
        first, then add sub-solver specifics.
        Dumps yaml file and git sha to results directory.

        Args:
            nsproblem (object of NSProblem):      the problem to solve
        '''
        super().__init__(nsproblem)

        # set unsteady options
        self.t = np.longdouble(nsproblem.t)
        self.T = self.options['timemarching']['T']
        self.dt = nsproblem.dt
        self.dt0 = nsproblem.dt0

        self.w0 = nsproblem.w0
        self.w1 = nsproblem.w1

        # XXX HACKED
        self.time_expr = nsproblem.time_expr
        self.fs_theta = nsproblem.fs_theta

        # XXX HACKED
        self.bnds = nsproblem.bnds

        self._xdmf_u = None
        self._xdmf_p = None

        if dump_parameters:
            self.dump_parameters()
        pass

        self.read_checkpoint()
        self.backup_restart()

    @rank0
    def dump_parameters(self):
        ''' Write parameters and git rev hash to files to io:write_path

        TODO: EXTEND THIS TO STEADY SOLVER! (NO RESULTS DIR EXISTS APRIORI)
        '''
        self.logger.info('Copying inputfile to results directory')
        path = self.options['io']['write_path']
        self.logger.debug('inputfile: ' + self.inputfile)
        self.logger.debug('results dir: ' + path)
        if (os.path.abspath(self.inputfile) ==
                os.path.abspath(path + '/input.yaml')):
            self.logger.debug('Same input.yaml, not copying.')
        else:
            if not os.path.exists(path):
                os.makedirs(path)
            shutil.copy2(self.inputfile, path + '/input.yaml')

        githash = utils.get_git_rev_hash(__file__)

        with open(path + '/git_rev_hash', 'w') as fp:
            fp.write(githash)

    def backup_restart(self):
        ''' If restarting from checkpoint, and checkpointing is enabled, backup
        old checkpoint directories and solution files. '''
        io = self.options['io']
        if not ('restart' in io and io['restart']['path'] and
                io['write_checkpoints']):
            return

        # check if restart path is checkpoint path and backup
        restart_path = Path(io['restart']['path'])
        write_path = Path(io['write_path'])
        checkpoint_path = write_path.joinpath('checkpoint')

        try:
            checkpoint_path.resolve()
        except FileNotFoundError:
            return

        if restart_path.parent.resolve() == checkpoint_path.resolve():
            import glob
            dirs = glob.glob(str(write_path.joinpath('checkpoint_[0-9]*')))
            imax = max(map(lambda s: int(str(s).split('_')[-1]), dirs)) if \
                dirs else -1
            backup_path = Path(io['write_path']).joinpath('checkpoint_{}'.
                                                          format(imax + 1))
            self.logger.info('Moving old checkpoint folder to safe location: '
                             + str(backup_path))

            MPI.barrier(MPI.comm_world)
            if MPI.rank(MPI.comm_world) == 0:
                checkpoint_path.rename(backup_path)
                pth = Path(io['write_path'])
                for f in ('u.xdmf', 'u.h5', 'p.xdmf', 'p.h5'):
                    pth.joinpath(f).replace(backup_path.joinpath(f))
                for g in ('stats.*.dat', 'timings.*'):
                    [f.rename(backup_path.joinpath(f.name)) for f in
                     pth.glob(g)]

            MPI.barrier(MPI.comm_world)

    def read_checkpoint(self):
        ''' Read stored checkpoint, if io.restart.path is given and
            io.restart.time > 0.
        '''
        io = self.options['io']
        if not ('restart' in io and io['restart']['path'] and
                io['restart']['time']):
            self.logger.info('No checkpoint given')
            return

        self.logger.info('Reading checkpoint at time {t} from {p}'.
                         format(t=io['restart']['time'],
                                p=io['restart']['path']))

        # check if restart path is checkpoint path and backup
        restart_path = Path(io['restart']['path'])

        if not restart_path.name == 'u.h5':
            restart_path = restart_path.joinpath('u.h5')

        if not restart_path.is_file():
            raise FileNotFoundError('{}'.format(restart_path))

        comm = self.w.function_space().mesh().mpi_comm()
        u, _ = self.w.sub(0).split(True)
        t_u = inout.read_HDF5_data(comm, str(restart_path), u, '/u')

        self.w.sub(0).assign(u)

        assert np.allclose(t_u, io['restart']['time'])
        self.t = np.longdouble(io['restart']['time'])

    def solve(self):
        ''' Caller for time stepper, according to options.
        '''
        method = (self.options['timemarching']['monolithic']['timescheme']
                  .lower())

        timer = Timer('TimeStepping')
        timer.start()
        if method in ('generalized_midpoint', 'gmp'):
            problem = GeneralProblem(self.Flin, self.w, J=self.Jlin,
                                     bcs=self.bcs, time_expr=self.time_expr,
                                     bnds=self.bnds)
            self.gmp_solve(problem)

        elif method == 'bdf2':
            problem = MultiStepProblem(self.Flin, self.w, self.Jlin,
                                       bcs=self.bcs)
            self.bdf2(problem)

        elif method in ('fs', 'fractionalstep'):
            problem = GeneralProblem(self.Flin, self.w, J=self.Jlin,
                                     bcs=self.bcs, time_expr=self.time_expr,
                                     bnds=self.bnds, fs_theta=self.fs_theta)
            self.fractionalstep_theta(problem)

        else:
            raise Exception('Timestepper {} not found!'.format(method))

        self.t_elapsed = timer.stop()
        self.write_statistics()
        self.cleanup()

    # @profile
    def gmp_solve(self, problem):
        ''' Backward Euler time descretization scheme.

        Args:
            problem     instance of GeneralProblem class
        '''
        self.logger.info('Time scheme: generalized midpoint rule (GMP), theta '
                         '= {}'.format(self.options['timemarching']
                                       ['monolithic']['theta']))
        solver = PETScSolver(problem, self.options, self._logging_filehandler)
        A = PETScMatrix()
        b = PETScVector()

        opt_nonlin = self.options['timemarching']['monolithic']['nonlinear']
        if opt_nonlin['method'] in ('newton', 'picard'):
            aitken = AitkenAccelerator(self.options)
        dw = Function(self.W)

        dt = self.options['timemarching']['dt']
        self.dt.assign(dt)

        w0 = self.w.copy(deepcopy=True)

        # adaptive_timestep = self.options['timemarching']['adaptive_timestep']
        # dtlist = []
        # if adaptive_timestep:
        #     epsilon = 1e-4
        #     wp = Function(self.W)
        #     dwp = Function(self.W)
        # # self.monitor(0)
        # efile = XDMFFile(self.options['io']['write_path'] + '/energy.xdmf')
        # P1 = FunctionSpace(w0.function_space().mesh(), 'CG', 1)
        # q = TestFunction(P1)
        # bc = [DirichletBC(P1, Constant(0.), problem.bnds, 1),
        #       DirichletBC(P1, Constant(0.), problem.bnds, 3),
        #       DirichletBC(P1, Constant(0.), problem.bnds, 4)]
        # # e_fun = Function(w0.function_space().sub(1).collapse())
        # e_fun = Function(P1)
        # e_fun.rename('energy', 'e')

        it = 1
        assert isinstance(self.t, np.longdouble)
        while self.t < self.T - 1e-12:
            # accumulation of rounding errors in self.t
            dt = self.set_timestep(it)
            self.t += dt

            problem.update_bcs(self.t)

            # time step control:
            # Gresho 1999, p. 714
            # if it > 1 and adaptive_timestep:
            #     # something like this .. Forward Euler predictor
            #     # - use dw from previous iteration
            #     wp.assign(self.w)  # + dt_float*dw)
            #     wp.vector().axpy(dt_float, dw.vector())
            if opt_nonlin['method'] == 'linear_extrapolation':
                self.w1.assign(w0)
                w0.assign(self.w)

            if opt_nonlin['method'] in ('constant_extrapolation',
                                        'linear_extrapolation'):
                # problem.F(b)
                # problem.J(A)
                # solver.solve(A, self.w.vector(), b)
                # u0, p0 = self.w.split(deepcopy=True)
                self.solve_ts_linear(problem, A, b, dw, solver)

                # u, p = self.w.split(deepcopy=True)
                # # u1, p1 = self.w1.split(deepcopy=True)
                # # u0, p0 = w0.split(deepcopy=True)
                # um = 0.5*(u + u0)
                # u_ = u0
                # temam = self.options['fem']['convection_skew_symmetric']
                # energy_glob = assemble(
                #     - 0.035*inner(grad(um), grad(um))*dx
                #     - dot(grad(um)*u_, um)*dx
                #     - Constant(temam)*0.5*div(u0)*dot(um, um)*dx
                #     + p*div(um)*dx
                #     - p*div(u)*dx)
                # energy_diff = assemble(
                #     - 0.035*inner(grad(um), grad(um))*dx
                # )
                # energy_conv = assemble(
                #     - dot(grad(um)*u_, um)*dx
                #     - Constant(temam)*0.5*div(u0)*dot(um, um)*dx
                # )
                # energy_compr = assemble(
                #     + p*div(um)*dx
                #     - p*div(u)*dx
                # )

                # # dot(u, u)*dx = ...
                # # - 0.5/dt*(u**2 - u0**2)*q*dx
                # energy_loc = assemble(
                #     # dot(u0, u0)*q*dx +
                #     # 2*dt*
                #     (
                #         - 0.035*inner(grad(um), grad(um))*q
                #         - dot(grad(um)*u_, um)*q
                #         - Constant(temam)*0.5*div(u0)*dot(um, um)*q
                #         + p*div(um)*q
                #         - p*div(u)*q)*dx)

                # [bc_.apply(energy_loc) for bc_ in bc]
                # e_fun.vector()[:] = energy_loc
                # efile.write(e_fun, self.t)

                # print('div(u) = {}'.format(assemble(div(u)*dx)))
                # # print('div(u*) = {}'.format(assemble(div(2*u0 - u1)*dx)))
                # print('Temam  = {}'.format(
                #     assemble(0.5*div(u0)*dot(u, u)*dx)))
                # print('conv   = {}'.format(
                #     assemble(dot(grad(u)*u0, u)*dx)))
                # print('sum    = {}'.format(
                #     assemble(dot(grad(u)*u0, u)*dx +
                #              0.5*div(u0)*dot(u, u)*dx)))
                # print('')
                # print('dE/dt  = {}'.format(energy_glob))
                # print('dE_dif = {}'.format(energy_diff))
                # print('dE_cv  = {}'.format(energy_conv))
                # print('dE_div = {}'.format(energy_compr))
                # print('E_kin  = {}'.format(assemble(0.5*dot(u, u)*dx)))
                # print('')

                # u1, p1 = self.w1.split(deepcopy=True)
                # u0, p0 = w0.split(deepcopy=True)
                # print('|| u(n-1) || = {}'.format(norm(u1.vector())))
                # print('|| u(n) ||   = {}'.format(norm(u0.vector())))
                # print('|| u(n+1) || = {}'.format(norm(u.vector())))
                # print('|| u_ ||     = {}'.format(norm(2*u0.vector() -
                #                                       u1.vector())))

            if opt_nonlin['method'] in ('newton', 'picard'):
                converged_reason = self.solve_ts_nonlinear(problem, A, b, dw,
                                                           solver, aitken)
                if converged_reason < 0:
                    break

                # u, p = self.w.split(deepcopy=True)
                # u0, p0 = self.w.split(deepcopy=True)
                # # u1, p1 = self.w1.split(deepcopy=True)
                # # u0, p0 = w0.split(deepcopy=True)
                # um = 0.5*(u + u0)
                # u_ = u0
                # temam = self.options['fem']['convection_skew_symmetric']
                # energy_glob = assemble(
                #     - 0.035*inner(grad(um), grad(um))*dx
                #     - dot(grad(um)*um, um)*dx
                #     - Constant(temam)*0.5*div(u0)*dot(um, um)*dx
                #     + p*div(um)*dx
                #     - p*div(u)*dx)

                # energy_loc = assemble(
                #     # - 0.5/dt*(u**2 - u0**2)*q*dx
                #     - 0.035*inner(grad(um), grad(um))*q*dx
                #     - dot(grad(um)*u, um)*q*dx
                #     - Constant(temam)*0.5*div(u0)*dot(um, um)*q*dx
                #     + p*div(um)*q*dx
                #     - p*div(u)*q*dx)

                # e_fun.vector()[:] = energy_loc
                # efile.write(e_fun, self.t)

                # print('div(u) = {}'.format(assemble(div(u)*dx)))
                # # print('div(u*) = {}'.format(assemble(div(2*u0 - u1)*dx)))
                # print('Temam  = {}'.format(
                #     assemble(0.5*div(u0)*dot(u, u)*dx)))
                # print('conv   = {}'.format(
                #     assemble(dot(grad(u)*u0, u)*dx)))
                # print('sum    = {}'.format(
                #     assemble(dot(grad(u)*u0, u)*dx +
                #              0.5*div(u0)*dot(u, u)*dx)))
                # print('dE/dt  = {}'.format(energy_glob))

            elif opt_nonlin['method'] == 'snes':
                self.snes(problem, solver, it)

            # if it > 1 and adaptive_timestep:
            #     # u, _ = self.w.split(deepcopy=True)
            #     # up, _ = wp.split(deepcopy=True)
            #     dwp.assign(self.w - wp)
            #     dup, _ = dwp.split(deepcopy=True)
            #     dt_float = dt_float*np.sqrt(
            #         epsilon/np.sqrt(assemble(dot(dup, dup)*dx)))
            #     self.dt.assign(dt_float)
            #     print('dt = {}'.format(dt_float))

            self.monitor_time(it)

            self.iterations_ksp.append(solver.iterations)
            self.residuals_ksp.append(solver.residuals)

            it += 1
        #     dtlist.append(dt_float)

        # np.savez(self.options['io']['write_path'] + '/dt.npz',
        #          dt=np.array(dtlist))

        return self

    def fractionalstep_theta(self, problem):
        ''' Fractional step-Theta time stepping algorithm

        Args:
            problem     instance of GeneralProblem class
        '''
        self.logger.info('Time scheme: Fractional-Step Theta (FS)')
        solver = PETScSolver(problem, self.options, self._logging_filehandler)
        A = PETScMatrix()
        b = PETScVector()

        if opt_nonlin['method'] in ('newton', 'picard'):
            aitken = AitkenAccelerator(self.options)
        dw = Function(self.W)

        # only for linear extrapolation
        w1_ = self.w.copy(deepcopy=True)

        # self.monitor(0)

        dt_float = self.options['timemarching']['dt']  # dt as float
        self.dt.assign(Constant(dt_float))

        it = 1
        while self.t < self.T - 1e-12:
            # assemble jacobian and RHS
            tn = self.t + dt_float
            for stage in range(3):
                # update previous time step size
                self.dt0.assign(self.dt)

                dt_frac = self.fracstep_update_stage(stage, dt_float, tn)

                problem.update_bcs(self.t)

                if opt_nonlin['method'] == 'linear_extrapolation':
                    self.w1.assign(w1_)
                    w1_.assign(self.w)

                if opt_nonlin['method'] in ('constant_extrapolation',
                                            'linear_extrapolation'):
                    self.solve_ts_linear(problem, A, b, dw, solver)

                elif opt_nonlin['method'] in ('newton', 'picard'):
                    converged_reason = self.solve_ts_nonlinear(problem, A, b,
                                                               dw, solver,
                                                               aitken)
                    if converged_reason < 0:
                        break

                elif opt_nonlin['method'] == 'snes':
                    self.snes(problem, solver, it)

                # call this outside of the stage loop?? FIXME
                self.monitor_time(it, dt=dt_frac, stage=stage)

                self.iterations_ksp.append(solver.iterations)
                self.residuals_ksp.append(solver.residuals)

                it += 1

        return self

    def fracstep_update_stage(self, stage, dt_float, tn):
        ''' Updates coefficients and time to current stage '''
        theta = 1 - np.sqrt(2)/2
        alpha = (1 - 2*theta)/(1 - theta)
        coef = [[alpha*theta, theta, (1 - alpha)*theta],
                [(1 - alpha)*(1 - 2*theta), 1 - 2*theta,
                 alpha*(1 - 2*theta)],
                [alpha*theta, theta, (1 - alpha)*theta]]
        if stage == 0:
            self.t += theta*dt_float
            dt_frac = theta*dt_float
        elif stage == 1:
            self.t = tn - theta*dt_float
            dt_frac = (1 - 2*theta)*dt_float
        else:
            self.t = tn
            dt_frac = theta*dt_float

        for th, upd in zip(self.fs_theta, coef[stage]):
            th.assign(upd)

        return dt_frac

    def solve_ts_linear(self, problem, A, b, dw, solver):
        ''' Advance timestep by solving linearized system '''
        # u_, _ = self.w.split(True)
        # self._u_supg.assign(u_)
        problem.F(b, self.w.vector())
        problem.J(A, self.w.vector())
        solver.solve(A, dw.vector(), b)
        self.w.vector().axpy(-1., dw.vector())

    def solve_ts_nonlinear(self, problem, A, b, dw, solver, aitken):
        ''' Advance one timestep by nonlinear iterations of A(x*)x = b '''
        maxit = (self.options['timemarching']['monolithic']['nonlinear']
                 ['maxit'])
        a1 = 1.
        nlit = 0
        self.w0.assign(self.w)
        # if it > 0 and adaptive_timestep:
        #     # nonlinear: use prediction as first guess for Newton!
        #     self.w.assign(wp)
        while nlit < maxit:
            problem.F(b, self.w.vector())
            problem.J(A, self.w.vector())

            converged, reason = self.monitor(b, dw, a1, it=nlit)
            if converged:
                self.logger.info('CONVERGED to {r} in {i} iterations'.
                                 format(i=nlit+1, r=reason))
                self.converged_reason = 1
                break

            solver.solve(A, dw.vector(), b)

            a1 = aitken.relax(dw)
            self.w.vector().axpy(-a1, dw.vector())

            nlit += 1

        else:
            self.logger.info('NOT CONVERGED, MAX_IT REACHED.')
            self.converged_reason = -5

        return self.converged_reason

    def snes(self, problem, linear_solver, time_iter=0):
        if time_iter == 0:
            self.logger.info('SNES')
            opt = self.options['timemarching']['monolithic']['nonlinear']

            PETScOptions.set('snes_monitor')
            PETScOptions.set('snes_type', 'newtonls')
            PETScOptions.set('snes_converged_reason')
            PETScOptions.set('snes_atol', opt['atol'])
            PETScOptions.set('snes_rtol', opt['rtol'])
            PETScOptions.set('snes_stol', opt['stol'])
            PETScOptions.set('snes_max_it', opt['maxit'])
            # PETScOptions.set('ksp_type', 'preonly')
            # PETScOptions.set('pc_type', 'lu')
            # PETScOptions.set('pc_factor_mat_solver_package', 'mumps')
            solver = PETScSNESSolver()
            solver.init(problem, self.w.vector())
            snes = solver.snes()
            snes.setFromOptions()
            snes.setKSP(linear_solver.ksp)
            snes.setConvergenceHistory()
            self._snes_solver = solver
        else:
            snes = self._snes_solver.snes()

        self.w0.assign(self.w)
        snes.solve(None, as_backend_type(self.w.vector()).vec())
        # self.residuals = np.hstack((self.residuals,
        #                             snes.getConvergenceHistory()[0]))
        # self.converged_reason = snes.getConvergedReason()
        # self.logger.debug(PETSc.Options().getAll())

        return self

    def set_timestep(self, it):
        ''' Set time step size. Current only need (no time adaptivity): reduce
        dt by a factor of 10 for the first time step, IFF linear extrapolation
        is selected for the convective velocity. I.e., the first full time step
        is split into two steps with 10% and 90%.

        TODO: modify this function when adpativity is implemented!!

        Args:
            it      time iteration count
        Return:
            dt (float)  current timestep
        '''
        if (self.options['timemarching']['monolithic']['nonlinear']['method']
                == 'linear_extrapolation'):
            # dt = self.options['timemarching']['dt']
            # factor = 0.1
            # if it == 0:
            #     self.dt.assign(dt*factor)
            #     self.dt0.assign(self.dt)
            # elif it == 1:
            #     # dt0 is already/still set
            #     self.dt.assign(dt*(1 - factor))
            # elif it == 2:
            #     self.dt0.assign(self.dt)
            #     self.dt.assign(dt)
            # elif it == 3:
            #     self.dt0.assign(self.dt)
            self.dt0.assign(self.dt)

        return self.dt.values()[0]

    def monitor_time(self, it, dt=None, stage=None):
        ''' Solution monitor, output interface.

        Args:
            it  iteration count
        '''
        tol = 1e-8
        writeout = 0
        if self.options['io']['write_xdmf']:
            write_dt = self.options['timemarching']['write_dt']
            # if every > 0 and np.mod(it, every) == 0 or self.T - self.t <= 0:
            if write_dt > 0:
                if not hasattr(self, '_t_write'):
                    dt = self.dt.values()[0] if not dt else dt
                    self._t_write = self.t - dt
                if (self._t_write <= self.t + tol and
                        (stage is None or stage == 2)):
                    self.write_timestep()
                    self._t_write += write_dt
                    writeout = 1
            elif self.T - self.t <= 0:
                # if write_dt is <= 0 (no timestep output) but
                #   write_xdmf is True, write last time step
                self.write_xdmf()
                writeout = 1

        if self.options['io']['write_checkpoints']:
            checkpt_dt = self.options['timemarching']['checkpoint_dt']
            if checkpt_dt > 0:
                if not hasattr(self, '_t_checkpt'):
                    if self.t > checkpt_dt:
                        self._t_checkpt = (self.t + checkpt_dt -
                                           self.t % checkpt_dt)
                    else:
                        self._t_checkpt = checkpt_dt
                if it == 1:
                    # always write checkpoint for first time step
                    self.write_checkpoint(it)
                    writeout += 2
                elif (self._t_checkpt <= self.t + tol and
                        (stage is None or stage == 2)):
                    self.write_checkpoint(it)
                    self._t_checkpt += checkpt_dt
                    writeout += 2
            if self.T - self.t <= 0 and writeout < 2:
                self.write_checkpoint(it)
                writeout += 2

        if self.options['timemarching']['report']:
            wstr = ''
            if np.mod(writeout, 2):
                wstr = 'W*'
            if writeout >= 2:
                wstr += ' CP*'

            if self.options['timemarching']['report'] == 1:
                self.logger.info('t = {t:.{width}f} \t({T}) \t{w}'
                                 .format(t=self.t, T=self.T, w=wstr, width=6))
            else:
                self.logger.info('t = {t:.{width}f} \t({T}) \t{w} \t|w|: {n}'
                                 .format(t=self.t, T=self.T, w=wstr, width=6,
                                         n=norm(self.w)))

    def write_hdf5_timeseries(self):
        ''' Write timestep to dolfin HDF5 TimeSeries.
        '''
        # TODO: unite with write_xdmf?
        if not self.options['io']['write_hdf5_timeseries']:
            return

        if (not hasattr(self, '_hdf5_ts_u') or not self._hdf5_ts_u or
                not hasattr(self, '_hdf5_ts_p') or not self._hdf5_ts_p):
            self.logger.info('Creating TimeSeries')
            self._hdf5_ts_u = TimeSeries(
                self.W.mesh().mpi_comm(),
                self.options['io']['write_path'] + '/u_timeseries'
            )
            self._hdf5_ts_p = TimeSeries(
                self.W.mesh().mpi_comm(),
                self.options['io']['write_path'] + '/p_timeseries'
            )

        self.logger.debug('Writing solution at time t = {t} to TimeSeries'
                          .format(t=self.t))
        (u, p) = self.w.split(deepcopy=True)
        self._hdf5_ts_u.store(u.vector(), float(self.t))
        self._hdf5_ts_p.store(p.vector(), float(self.t))
        return self

    def write_timestep(self):
        ''' Combined Checkpoint and XDMF output so that split(w) is needed only
            once.
        '''
        if (self.options['io']['write_hdf5_timeseries'] or
                self.options['io']['write_xdmf']):
            (u, p) = self.w.split()
            u.rename('u', 'velocity')
            p.rename('p', 'pressure')

        if self.options['io']['write_hdf5_timeseries']:
            if (not hasattr(self, '_hdf5_ts_u') or not self._hdf5_ts_u or
                    not hasattr(self, '_hdf5_ts_p') or not self._hdf5_ts_p):
                self.logger.info('Creating TimeSeries')
                self._hdf5_ts_u = TimeSeries(
                    self.W.mesh().mpi_comm(),
                    self.options['io']['write_path'] + '/u_timeseries'
                )
                self._hdf5_ts_p = TimeSeries(
                    self.W.mesh().mpi_comm(),
                    self.options['io']['write_path'] + '/p_timeseries'
                )

            self.logger.debug('Writing solution at time t = {t} to TimeSeries'
                              .format(t=self.t))
            self._hdf5_ts_u.store(u.vector(), float(self.t))
            self._hdf5_ts_p.store(p.vector(), float(self.t))

        if self.options['io']['write_xdmf']:
            if (not (hasattr(self, '_xdmf_u') or hasattr(self, '_xdmf_p')) or
                    not (self._xdmf_u or self._xdmf_p)):
                self._xdmf_u = XDMFFile(self.options['io']['write_path'] +
                                        '/u.xdmf')
                self._xdmf_p = XDMFFile(self.options['io']['write_path'] +
                                        '/p.xdmf')
                self._xdmf_u.parameters['rewrite_function_mesh'] = False
                self._xdmf_p.parameters['rewrite_function_mesh'] = False
            self.logger.debug('Writing XDMF: t = {t}'.format(t=self.t))
            self._xdmf_u.write(u, float(self.t))
            self._xdmf_p.write(p, float(self.t))

        return self

    # @rank0
    def write_statistics(self):
        ''' write number of linear iterations (KSP) and timings to files '''
        rank = MPI.rank(MPI.comm_world)
        opt = self.options['io']

        outfile = Path(opt['write_path']).joinpath('stats.{}.dat'.format(rank))
        with outfile.open('wb') as fout:
            data = {
                'iterations_ksp': self.iterations_ksp,
                'residuals_ksp': self.residuals_ksp,
                'time': self.t_elapsed,
                'np': MPI.comm_world.size,
                'node': platform.uname()[1],
            }
            pickle.dump(data, fout, protocol=pickle.HIGHEST_PROTOCOL)

        outfile = Path(opt['write_path']).joinpath('timings.{}'.format(rank))
        with outfile.open('wt') as fout:
            # or: mode 'wb' and string.encode('utf-8')
            fout.write(timings(TimingClear.keep,
                               [TimingType.wall, ]).str(True))


class NSSteady(NSSolverBase):
    ''' Steady state Navier-Stokes solver class for solving a given variational
    problem (object of the NSProblem class).

    Usage:
        ...

    The implemented solvers are:
        - PETSc SNES Newton method
        - 'manual' exact Newton method
        - Quasi-Newton method
        - Picard iteration with Aitken acceleration
    '''
    def __init__(self, nsproblem):
        ''' Initialize Navier-Stokes solver. Call abstract class initializer
        first, then add sub-solver specifics.

        Args:
            nsproblem (object of NSProblem):      the problem to solve
        '''
        super().__init__(nsproblem)
        # self.logger = logging.getLogger(self.__class__.__name__)
        # self.logger.addHandler(ch)

        self.F, self.J = (nsproblem.nls_form if nsproblem.nls_form
                          else (None, None))
        self.Fq, self.Jq = (nsproblem.qnls_form if nsproblem.qnls_form
                            else (None, None))
        self.energy_form = (nsproblem.energy_form
                            if hasattr(nsproblem, 'energy_form') else None)

        self.converged_reason = None
        self._init = None

        self.energy = []
        pass

    def solve(self):
        ''' Caller for manual solvers (newtonlike()) and snes().
        Calls of the Newton solvers are preceded by initialization with Picard
        iterations.

        Return:
            self.w (dolfin Function)         final solution
            self.residuals (np.ndarray)       residual
        '''
        opt_nonlin = self.options['timemarching']['monolithic']['nonlinear']
        method = opt_nonlin['method']
        self.it = 0

        nullspace = self.build_nullspace()

        if method == 'picard':
            problem = GeneralProblem(self.Flin, self.w, J=self.Jlin,
                                     bcs=self.bcs, nullspace=nullspace)
            self.newtonlike(problem)
        else:
            init_steps = opt_nonlin['init_steps']
            # initialize with n=init_steps Picard iterations
            self.initial_solution(init_steps=init_steps)

            if method in ('snes', 'snes_manual', 'newton'):
                problem = GeneralProblem(self.F, self.w, J=self.J,
                                         bcs=self.bcs, nullspace=nullspace)
            elif method == 'qnewton':
                problem = GeneralProblem(self.Fq, self.w, J=self.Jq,
                                         bcs=self.bcs, nullspace=nullspace)

            if method == 'snes':
                self.snes(problem)
            elif method == 'snes_manual':
                self.snes_manual(problem)
            elif method in ['newton', 'qnewton']:
                self.newtonlike(problem)

        self.write_xdmf()
        self.write_checkpoint(0)
        self.cleanup()

    def initial_solution(self, init_steps=0):
        ''' Compute initial solution.

        Args:
            init_steps          if 0, compute Stokes solution, otherwise
                                perform n=init_steps Picard iterations
                (default: 0)
        '''
        if init_steps > 0:
            self._init = True
            nullspace = self.build_nullspace()
            lin_problem = GeneralProblem(self.Flin, self.w, J=self.Jlin,
                                         bcs=self.bcs, nullspace=nullspace)
            self.newtonlike(lin_problem, init_steps=init_steps)
            self._init = False
        return self


class StokesUnsteady(NSUnsteady):
    def __init__(self, nsproblem):
        ''' Initialize unsteady Stokes solver.
        Inherited from NSUnsteady instead of NSSolverBase!
        Only bdf1 is modified very slightly. This is done in order to keep
        NSUnsteady.bdf1 as clean as possible and to avoid cluttering with
        "if is_stokes()" checks.

        Args:
            nsproblem (object of NSProblem):      the problem to solve
        '''
        super().__init__(nsproblem)

    # @profile
    def bdf1(self, problem):
        ''' Backward Euler time descretization scheme.
        In the case of Stokes, where the system matrix does not change in time,
        reassembly is necessary ONLY for the RHS due to instationary boundary
        conditions.

        Args:
            problem     instance of GeneralProblem class
        '''

        solver = PETScSolver(problem, self.options, self._logging_filehandler)
        A = PETScMatrix()
        b = PETScVector()
        problem.J(A)

        # self.monitor(0)

        Niter = int((self.T - self.t)/dt_float)

        for it in range(1, Niter + 1):
            # assemble jacobian and RHS
            self.t += dt_float
            problem.update_bcs(self.t)
            problem.F(b)

            solver.solve(A, self.w.vector(), b)

            self.monitor_time(it)

        return self


class ReynoldsContinuation:
    ''' Class for applying the Reynolds continuation technique to an instance
    of NSProblem. Starting at a low value, the Reynolds number is gradually
    increased. '''
    def __init__(self, problem, length, u_bulk, Re_sequence=None, Re_start=10,
                 Re_end=None, Re_num=10, init_steps_sequence=None,
                 init_start=2, init_end=10):
        ''' Initialize & set instance variables.

        Args:
            problem     instance of NSProblem
            length      characteristic length for Reynolds number, e.g.
                        hydraulic diameter
            u_bulk      bulk velocity, e.g. for parabolic inlet 2/3*umax

        Optional:
            Re_start (default 10)   if no sequence given, start Reynolds number
            Re_end (default None)   target Re, if None calculate value from
                                    input file
            Re_num (default 10)     number of steps in Reynolds continuation
            Re_sequence             predefined list of Reynolds numbers to
                                    solve for
            init_start (default 2)  number of Picard iterations to start Newton
                                    method for lowest Reynolds number
            init_end (default 10)   number of initial Picard iterations for
                                    target Reynolds number
            init_steps_sequence     predefined list of Picard iterations to
                                    start a Newton method, for each Reynolds
                                    number to be calculated

        '''

        raise Exception('ReynoldsContinuation not supported anymore, needs '
                        'major rewrite!')

        self.problem = problem
        self.length = length
        self.ub = u_bulk
        self.Re_start = Re_start
        self.Re_end = Re_end
        self.Re_num = Re_num
        self.Re_sequence = Re_sequence
        self.init_steps_sequence = init_steps_sequence
        self.init_start = init_start
        self.init_end = init_end
        pass

    def comp_Re_number_from_options(self):
        ''' Calculates Reynolds number from u_bulk, mu, rho, characteristic
        length.
        '''
        L = self.length
        rho = self.problem.options['density']
        mu = self.problem.options['dynamic_viscosity']
        ub = self.ub
        self.Re_end = rho*L*ub/mu

        return self

    def comp_mu_from_Re(self, Re):
        ''' Compute viscosity mu from a given Reynolds number with L, U,
        specified and rho from parameters.

        Args:
            Re  Reynolds number

        Return:
            mu  viscosity
        '''
        L = self.length
        rho = self.problem.options['density']
        ub = self.ub

        mu = rho*L*ub/Re

        return mu

    def create_Re_sequence(self):
        ''' Calculate Reynolds number sequence. Predefined spacing is x^1.5,
        which works well for the backward-facing step case and Re = 800.
        '''
        if not self.Re_end:
            self.comp_Re_number_from_options()

        x = np.linspace(self.Re_start**1.5, self.Re_end**1.5, self.Re_num)
        self.Re_sequence = x**(1./1.5)

        return self

    def create_init_steps_sequence(self):
        ''' Calculate init step number sequence, log spaced. '''
        x = np.logspace(np.log10(self.init_start),
                        np.log10(self.init_end), self.Re_num)
        self.init_steps_sequence = x.round().astype(int)

        return self

    def solve(self):
        ''' Loop over Reynolds numbers and solve problem f.e. Re. Stop in case
        of non-convergence of a solve.
        '''
        if self.Re_sequence is None:
            self.create_Re_sequence()
        else:
            self.Re_num = len(self.Re_sequence)

        if self.init_steps_sequence is None:
            self.create_init_steps_sequence()

        for i, (isteps, Re) in enumerate(zip(self.init_steps_sequence,
                                             self.Re_sequence)):
            print('{i}/{N}\t isteps: {s}\t Re: {Re}'
                  .format(i=i+1, N=self.Re_num, s=isteps, Re=Re))
            self.problem.options['dynamic_viscosity'] = \
                self.comp_mu_from_Re(Re)
            self.problem.options['timemarching']['monolithic'][
                'nonlinear']['init_steps'] = isteps
            if i == 0:
                self.problem.init()
            else:
                self.problem.variational_form()

            self.sol = NSSteady(self.problem)
            self.sol.solve()
            if self.sol.converged_reason < 0:
                raise Exception('Diverged in step {0} for Re = {1}'
                                .format(i, Re))
                break

        self.w = self.sol.w

        return self


generate = {
    'steady': NSSteady,
    'unsteady': NSUnsteady,
    'stokesunsteady': StokesUnsteady
}


def solver(problem):
    opt = problem.options
    if ('timemarching' in opt and not
            (opt['timemarching']['monolithic']['timescheme'] == 'steady')):
        solver = 'unsteady'
        if 'stokes' in opt and opt['stokes']:
            solver = 'stokesunsteady'
    else:
        solver = 'steady'

    # logger.info('Invoking {}'.format(generate[solver].__name__))
    return generate[solver](problem)
