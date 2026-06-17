''' Fractional-Step Navier-Stokes solver module

Author: Jeremias Garay
Date:   2026-04-25

'''

from mpi4py import MPI
import dolfinx
import dolfinx.fem as fem
from dolfinx.fem import (
    Function, functionspace, form as fem_form, assemble_scalar,
)
from dolfinx.fem.petsc import (
    assemble_matrix, assemble_vector,
    apply_lifting, set_bc,
    create_matrix, create_vector,
)
from dolfinx.common import Timer
from dolfinx.io import XDMFFile
import ufl
from ufl import (
    TrialFunction, TestFunction, inner, dot, grad, dx,
    FacetNormal, Identity, Measure,
)
from petsc4py import PETSc
import numpy as np
import pickle
import shutil
import os
import platform
from common import inout, utils
from ..logger.logger import LoggerBase
from pathlib import Path

from packaging.version import Version as _V

if _V(dolfinx.__version__) < _V('0.7'):
    raise Exception('DOLFINx version 0.7 or higher required!')

def rank0(func):
    ''' Rank 0 decorator: decorated function "does nothing" if rank > 0 '''
    def inner(*args, **kwargs):
        if MPI.COMM_WORLD.rank == 0:
            func(*args, **kwargs)
    return inner

def _assemble_mat(form, bcs=None, mat=None):
    ''' Assemble bilinear UFL form into PETSc.Mat.
    If mat is provided, re-assembles into it (zeroing first).
    '''
    import dolfinx.cpp.fem.petsc as _cpp_petsc
    from dolfinx.fem import pack_constants, pack_coefficients
    compiled = fem_form(form)
    if mat is None:
        result = assemble_matrix(compiled, bcs=bcs or [])
        result.assemble()
        return result
    mat.zeroEntries()
    _cpp_petsc.assemble_matrix(mat, compiled._cpp_object,
                               pack_constants(compiled),
                               pack_coefficients(compiled),
                               [bc._cpp_object for bc in (bcs or [])],
                               False)
    mat.assemble()
    return mat

def _assemble_vec(form):
    ''' Assemble linear UFL form into PETSc.Vec. '''
    result = assemble_vector(fem_form(form))
    result.ghostUpdate(addv=PETSc.InsertMode.ADD,
                       mode=PETSc.ScatterMode.REVERSE)
    return result

def _mat_vec(A, x):
    ''' Compute y = A * x, returning a new PETSc.Vec. '''
    y = A.createVecLeft()
    A.mult(x, y)
    return y

def _apply_dbc_to_mat(mat, bcs, diag=1.0):
    ''' Zero rows AND cols for DirichletBC DOFs and set diagonal to diag.
    Use only for pressure matrices (symmetric treatment). '''
    for bc in bcs:
        dofs = bc.dof_indices()[0]
        mat.zeroRowsColumnsLocal(dofs, diag)

def _apply_dbc_rows_to_mat(mat, bcs, diag=1.0):
    ''' Zero rows only for DirichletBC DOFs and set diagonal to diag.
    Matches legacy FEniCS bc.apply(A) behaviour for velocity matrices:
    columns are left intact so no apply_lifting is needed on the RHS. '''
    for bc in bcs:
        dofs = bc.dof_indices()[0]
        mat.zeroRowsLocal(dofs, diag)

def _zero_mat_rows(mat, bcs):
    ''' Zero rows of PETSc.Mat corresponding to DirichletBC DOFs. '''
    for bc in bcs:
        dofs = bc.dof_indices()[0]
        mat.zeroRowsLocal(dofs)


class Solver(LoggerBase):
    # =========================================================================
    # Initialization
    # =========================================================================

    def __init__(self, problem, dump_parameters=True):
        ''' Solver Initializer.

        Args:
            problem (navierstokes.fractionalstep.problem): Problem object
            dump_parameters (bool): yes/no dump parameters to file input.yaml
        '''
        super().__init__()
        self._logging_filehandler = problem._logging_filehandler
        if self._logging_filehandler:
            self.logger.addHandler(self._logging_filehandler)

        self.logger.info('Initializing')
        self.options = problem.options
        self.inputfile = problem.inputfile
        self.t = 0.
        self.it = 0
        self._t_write = 0.
        self._t_checkpt = 0.
        self._writeout = 0
        self.ndim = problem.ndim

        # Solution fields
        self.bc_dict = problem.bc_dict
        self.u_lst = problem.u_lst
        self.u0_lst = problem.u0_lst
        self.u0_mapdd_lst = problem.u0_mapdd_lst
        self.u_conv_assigned = problem.u_conv_assigned
        self._u_tmp_lst = [Function(problem.Vi) for i in range(self.ndim)]
        self.p = problem.p
        self.u = problem.u
        self.bnds = problem.bnds

        # ALE deformation-gradient references
        self.F, self.J = problem.F, problem.J
        self.F0, self.J0 = problem.F0, problem.J0

        # Feature flags
        self.ale = problem.ale
        self._using_ale = problem._using_ale
        self.wk = problem.wk
        self._using_wk = problem._using_wk
        self._using_mapdd = problem._using_mapdd

        # Variational forms (assembled in init_assembly)
        self.forms = problem.forms

        # IPCS uses a separate correction pressure phi; CT reuses p
        if self.options['timemarching']['fractionalstep']['scheme'] == 'IPCS':
            self.phi = Function(problem.Q, name='phi')
        else:
            self.phi = self.p

        # DOF block size for split/merge between V and Vi
        self._bs = problem.V.dofmap.index_map_bs

        # Element-constant SUPG tau updater (None if SUPG disabled or pointwise)
        self._sd_param = getattr(problem, '_sd_param', None)

        # Correction velocity increments
        self.du = [Function(problem.Vi, name='du_i') for i in range(self.ndim)]
        self.du_vec = Function(problem.V, name='du')

        self._init_state_flags()
        self._init_ale_fields(problem)
        self._init_mat_vec_storage()

        if dump_parameters:
            self.dump_parameters()

        problem.close_logs()

    def _split_vec_to_lst(self, u, u_lst):
        ''' Copy vector Function u into list of scalar Functions u_lst. '''
        bs = self._bs
        for i, ui in enumerate(u_lst):
            ui.x.array[:] = u.x.array[i::bs]

    def _merge_lst_to_vec(self, u_lst, u):
        ''' Copy list of scalar Functions u_lst into vector Function u. '''
        bs = self._bs
        for i, ui in enumerate(u_lst):
            u.x.array[i::bs] = ui.x.array

    def _init_state_flags(self):
        ''' Initialize boolean state and option flags. '''
        self._diverged = False
        self._optimizing = False       # set to True by an external optimizer
        self._optimize_robin = False
        self._optimize_windkessel = False
        self._initialized = False

        self._applying_dc_on_update = bool(
            self.options['fem'].get('DC_in_update', False))
        self._applying_pen_on_update = bool(
            self.options['fem'].get('PEN_in_update', False))

        self._using_fnv_semi_implicit = (
            'fnv' in self.forms['u'] and
            self.forms['u'].get('fnv_type') == 'semi-implicit')

        # Which velocity field represents the "state" for ROUKF
        self.state_velocity = self.options['fluid'].get('state_velocity',
                                                        'update')

    def _init_ale_fields(self, problem):
        ''' Link ALE-specific fields from the problem object. '''
        if not self._using_ale:
            return

        self.k = problem.k
        self.d = problem.d
        self.d0 = problem.d0
        self.upd = problem.upd
        self.upd_lst = problem.upd_lst

        self.d_bc = Function(problem.D)
        self.v_bc = Function(problem.V)
        self.d_lst_bc = [Function(problem.Di) for _ in range(self.ndim)]
        self.v_lst_bc = [Function(problem.Vi) for _ in range(self.ndim)]

        if self.ale['type'] == 'external':
            self.S = problem.S
            self.d_s = problem.d_s
            self.v_s = problem.v_s

        # FSI semi-implicit projection coupling data (see form_pressure)
        self.v_si = getattr(problem, 'v_si', None)

    def _init_mat_vec_storage(self):
        ''' Initialize dicts for assembled matrices and vectors. '''
        self.mat = {}
        self.vec = {}

        if self._using_ale:
            self.mat['d'] = {'diff': None, 'div': None}
            self.vec['d'] = {'rhs_const': None}

        self.mat['u'] = {
            'mass': None, 'conv': None, 'rhs': None,
            'pdiv': None, 'gradp': None, 'fnv': None,
            'lhs_navslip': {i: [] for i in range(self.ndim)},
            'rhs_navslip': {i: [] for i in range(self.ndim)},
            'lhs_trans': {i: [] for i in range(self.ndim)},
            'rhs_trans': {i: [] for i in range(self.ndim)},
            'p_trans': {i: None for i in range(self.ndim)},
        }
        # mass_robin: boundary mass matrices scaled by rho/dt
        # mass_bound: unscaled boundary mass matrices (transpiration)
        self.mat['p'] = {
            'rhs_u': None, 'laplacian': None,
            'mass_robin': None, 'mass_bound': [], 'u_norm_bound': [],
        }
        self.vec['u'] = {'rhs_const': None}
        self.vec['p'] = {'rhs_const': None}

        if self._using_wk:
            self.pi_functions = {k: [] for k in
                                 self.bc_dict['p']['windkessel']['params']}
            if self.wk['implicit']:
                self.vec['p'].update({
                    'windkessel_rhs': [],
                    'windkessel_lhs_lrc_diag': None,
                })
                self.mat['p'].update({'windkessel_lhs_lrc': None})

    @rank0
    def dump_parameters(self):
        ''' Write parameters and git rev hash to files to
        timemarching>write_path if timemarching>write is set.
        '''
        self.logger.info('Copying inputfile to results directory')
        path = self.options['io']['write_path']
        self.logger.info('inputfile: ' + self.inputfile)
        self.logger.info('results dir: ' + path)
        if (os.path.abspath(self.inputfile) ==
                os.path.abspath(path + '/input.yaml')):
            self.logger.info('Same input.yaml, not copying.')
        else:
            if not os.path.exists(path):
                os.makedirs(path)
            shutil.copy2(self.inputfile, path + '/input.yaml')

        #githash = utils.get_git_rev_hash(__file__)

        #with open(path + '/git_rev_hash', 'w') as fp:
        #    fp.write(githash)

    def init(self):
        ''' Initialize matrices and solvers '''
        self.init_assembly()
        self.init_solvers()
        self._initialized = True
        self.read_checkpoint()
        self.backup_restart()

        self.write_initial_condition()

    def write_initial_condition(self):
        ''' Write initial condition XDMF and HDF5 checkpoints '''
        if (self.options['io']['write_xdmf'] or
                self.options['io']['write_checkpoints']):
            self.logger.info('Writing initial condition')

        self.write_xdmf(t=0.)
        self.write_checkpoint(0)

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

            MPI.COMM_WORLD.Barrier()
            if MPI.COMM_WORLD.rank == 0:
                checkpoint_path.rename(backup_path)
                pth = Path(io['write_path'])
                files = ('u.xdmf', 'u.h5', 'p.xdmf', 'p.h5')
                if self._using_ale:
                    # FIXME add upd separately?
                    files.update(('d.xdmf', 'd.h5'))
                for f in files:
                    pth.joinpath(f).replace(backup_path.joinpath(f))

                for g in ('stats.*.dat', 'timings.*'):
                    [f.rename(backup_path.joinpath(f.name)) for f in
                     pth.glob(g)]

            MPI.COMM_WORLD.Barrier()

    def read_checkpoint(self):
        ''' Read stored checkpoint, if io.restart.path is given and
            io.restart.time > 0.
        '''
        io = self.options['io']
        if not ('restart' in io and io['restart']['path'] and
                io['restart']['time']):
            self.logger.info('No checkpoint given')
            return

        if self.options['timemarching']['fractionalstep']['scheme'] == 'IPCS':
            raise Exception('checkpointing not yet tested for IPCS')

        self.logger.info('Reading checkpoint at time {t} from {p}'.
                         format(t=io['restart']['time'],
                                p=io['restart']['path']))

        # check if restart path is checkpoint path and backup
        restart_path = Path(io['restart']['path'])
        w_path = restart_path / 'w.h5'
        if not w_path.exists():
            raise FileNotFoundError(
                'Checkpoint file not found: {}'.format(w_path))
        if self._using_ale and not (restart_path / 'd.h5').exists():
            raise FileNotFoundError(
                'ALE checkpoint not found: {}'.format(restart_path / 'd.h5'))

        import h5py
        comm = self.u.function_space.mesh.comm
        w_file = restart_path / 'w.h5'

        def _scatter_full(fn, full_arr):
            idx = fn.function_space.dofmap.index_map
            start, end = idx.local_range
            fn.x.array[:idx.size_local] = full_arr[start:end]
            fn.x.scatter_forward()

        if comm.rank == 0:
            with h5py.File(str(w_file), 'r') as f:
                u_full = np.array(f['u'])
                p_full = np.array(f['p'])
                t_u = float(f.attrs.get('t', 0.0))
        else:
            u_full = None
            p_full = None
            t_u = 0.0
        u_full = comm.bcast(u_full, root=0)
        p_full = comm.bcast(p_full, root=0)
        t_u = comm.bcast(t_u, root=0)

        _scatter_full(self.u, u_full)
        _scatter_full(self.p, p_full)

        if self._using_ale:
            if comm.rank == 0:
                with h5py.File(str(restart_path / 'd.h5'), 'r') as f:
                    d_full = np.array(f['d'])
            else:
                d_full = None
            d_full = comm.bcast(d_full, root=0)
            _scatter_full(self.d, d_full)

        assert np.allclose(t_u, io['restart']['time'])

        # self.t is time of first computed time step
        t0 = io['restart']['time']
        self.t = t0
        self._t_checkpt = t0
        self._t_write = t0

        if self.options['timemarching']['fractionalstep']['scheme'] == 'IPCS':
            t0 = inout.read_HDF5_data(comm, restart_path.joinpath('u0.h5'),
                                      self.u, '/u')
            assert np.allclose(t0 + self.options['timemarching']['dt'], t_u)
            self.logger.info('Loading u0 at time {}'.format(t0))

    # =========================================================================
    # Time loop
    # =========================================================================

    def solve(self):
        ''' Solve fractional step scheme '''
        from nsx.reporter import ProgressReporter
        timer = Timer('Z TimeStepping')

        self.init()

        dt = self.options['timemarching']['dt']
        T = self.options['timemarching']['T']
        times = np.arange(self.t + dt, T + dt, dt)

        rank = MPI.COMM_WORLD.rank
        reporter = ProgressReporter(len(times), dt, rank=rank)
        reporter.print_banner()
        reporter.start()
        self._reporter = reporter

        try:
            for i, t in enumerate(times):
                it = i + 1
                self.it = it
                self.timestep(it, t)

                #self.monitor(it)

                reporter.update(
                    it, t,
                    norm_u_tent=getattr(self, '_norm_u_tent', float('nan')),
                    norm_p=getattr(self, '_norm_p', float('nan')),
                    norm_u_upd=getattr(self, '_norm_u_upd', float('nan')),
                )

                if self._diverged:
                    break
        finally:
            reporter.close()

        self.t_elapsed = timer.stop()
        self.write_statistics()
        self.cleanup()

    def cleanup(self):
        ''' Cleanup '''
        self.close_xdmf()
        self.close_logs()
        PETSc.Options().clear()

    def timestep(self, i=0, t=None, state=None, parameters=None,
                restart=False, observations=None):
        ''' Timestep interface for ROUKF algorithm.

        Args (optional):
            i (int):        iteration count
            t (float)       current (new) time
            state (list):   state variables (list of fenics functions)
            parameters (list):   parameters (list of numbers)
            restart (bool): precomputes internal variables from state
            observations (list):   list of Functions to save observations
        '''
        
        if restart and i == 0:
            self.init_state(state)

        if not self._initialized:
            #if restart:
            #    self.update_state(state)
            self.init()

        if t:
            self.t = t

        if restart:
            self.restart_timestep(state, parameters)

        if self._using_ale:
            if (self.ale['io']['read_checkpoints'] and
                    self.ale['type'] == 'external'):

                self.read_timestep(i)

            self.update_displacement()
            self.solve_displacement()
            # self.logger.debug('|d| = {:.4e}'.format(norm(self.d)))

        self.solve_tentative_velocity()
        self._norm_u_tent = np.sqrt(
            fem.assemble_scalar(fem.form(ufl.inner(self.u, self.u) * ufl.dx))
        )

        if self._using_wk:
            self.solve_windkessel()

        if (state and (self.options['timemarching']['fractionalstep']
                       ['scheme'] == 'CT')):
            if (self.state_velocity == 'tentative' and (
                        self.wk['explicit'])):
                self.update_state(state)
            elif (self.state_velocity == 'tentative' and not (
                        self._using_wk)):
                self.update_state(state)

        if (observations and (self.options['estimation']['roukf']
                              ['observation_operator'] == 'state')):
            self.observation(observations)

        # post processing
        self.solve_pressure()
        self._norm_p = np.sqrt(
            fem.assemble_scalar(fem.form(self.p * self.p * ufl.dx))
        )

        if self.write_velocity == 'tentative':
            self.write_timestep(i)

        if self._using_wk and self.wk['implicit']:
            self.solve_windkessel(flow=False)

        if (state and (self.options['timemarching']['fractionalstep']
                       ['scheme'] == 'CT')):
            if (self.state_velocity == 'tentative' and (
                    self.wk['implicit'])):
                self.update_state(state)

        self.solve_velocity_update()
        self.pressure_increment()
        self._norm_u_upd = np.sqrt(
            fem.assemble_scalar(fem.form(ufl.inner(self.u, self.u) * ufl.dx))
        )
        if (state and (self.options['timemarching']['fractionalstep']
                       ['scheme'] == 'CT')):
            if self.state_velocity == 'update':
                self.update_state(state)

        if (state and (self.options['timemarching']['fractionalstep']
                       ['scheme'] == 'IPCS')):
            self.update_state(state)

        if (observations and (self.options['estimation']['roukf']
                              ['observation_operator'] == 'postprocessing')):
            self.observation(observations)

        if self.write_velocity == 'update':
            self.write_timestep(i, update=True)

    def restart_timestep(self, state, parameters):
        ''' Restart time step.

        Assign previous state and parameters.
        Compute internal variables from the given state, as required for
        computing the next state.

        Args:
            state (list of Function): list of state functions
            parameters (numpy.ndarray):    array of parameters
        '''
        t0 = self.t
        self.t -= self.options['timemarching']['dt']
        scheme = self.options['timemarching']['fractionalstep']['scheme']

        if not state:
            raise Exception('State required for restarting algorithm')

        self.assign_state(state)
        self.assign_parameters(parameters)

        if self.state_velocity == 'tentative':
            if scheme == 'CT':
                if self._using_wk:
                    self.solve_windkessel(restart=True)
                self.solve_pressure()
                self.solve_velocity_update()

            elif scheme == 'IPCS':
                # IPCS algorithm is restarting
                # FIXME need to assign u0_lst and u_lst
                pass


        if self.wk['implicit']:
            self.update_windkessel_LRC()

        self.t = t0

    # =========================================================================
    # ROUKF estimation interface
    # =========================================================================

    def init_observations(self):
        ''' Initialize observations for ROUKF.

        Reads mesh or meshes, creates function space(s) and initializes and
        returns function(s).

        Returns:
            fun_lst     list of measurement/observation functions, for each
                        given mesh
        '''
        measurement_lst = self.options['estimation']['measurements']
        if not isinstance(measurement_lst, list):
            measurement_lst = [measurement_lst]

        mesh_lst = [meas['mesh'] for meas in measurement_lst]

        if 'fe_degree' in self.options['estimation']['measurements'][0]:
            degree = self.options['estimation']['measurements'][0]['fe_degree']
        else:
            degree = 1

        for measurement in measurement_lst[1:]:
            if not degree == measurement['fe_degree']:
                raise Exception('fe_degree must the the same for all '
                                'measurements!')

        # All measurements need to be given as 3D (no velocity_direction) or
        # 1D, mixing is not supported
        # Check:
        veldir_given = []
        numpy_given = []
        for meas in measurement_lst:
            if 'velocity_direction' in meas:
                direction = meas['velocity_direction']
                if isinstance(direction, list):
                    if not all(isinstance(d, (int, float)) for d in direction):
                        raise Exception('velocity_direction should be list of '
                                        'numbers or False/None omitted. Is: {}'
                                        .format(direction))

                    veldir_given.append(True)
                else:
                    assert not direction
                    veldir_given.append(False)
            else:
                veldir_given.append(False)
            if 'numpy' in meas:
                numpy_given.append(meas['numpy'])
            else:
                numpy_given.append(False)

        all_scalar = all([flag is True for flag in veldir_given])
        all_vector = all([flag is False for flag in veldir_given])
        all_numpy = all([flag is True for flag in numpy_given])
        all_fenics = all([flag is False for flag in numpy_given])

        if not (all_scalar or all_vector):
            raise Exception('All measurements need to be given as 3D (no '
                            'velocity_direction) or 1D, mixing is not '
                            'supported')

        if not (all_numpy or all_fenics):
            raise Exception('All measurements need to be given either '
                            'as Numpy arrays or as Fenics functions, '
                            'mixing is not supported')

        if degree == 1:
            element_family = 'P'
        elif degree == 0:
            element_family = 'DG'
        else:
            raise Exception('Unsupported measurement FE degree: {}'
                            .format(degree))

        fun_lst = []
        fun_aux_lst = []
        V_aux = None
        for meshfile in mesh_lst:
            mesh, _, _ = inout.read_mesh(meshfile)
            if all_scalar:
                V = functionspace(mesh, ("Lagrange" if element_family == 'P' else "DG", degree))
                V_aux = functionspace(mesh, ("Lagrange" if element_family == 'P' else "DG", degree, (self.ndim,)))
                fun_aux_lst.append(Function(V_aux))
            else:
                V = functionspace(mesh, ("Lagrange" if element_family == 'P' else "DG", degree, (self.ndim,)))
            fun_lst.append(Function(V))

        self._observation_fun_aux_lst = fun_aux_lst
        
        res_lst = []
        self._observation_res_lst = res_lst
        if all_numpy:
            np_lst = []
            for meas in measurement_lst:
                res = meas['resolution']
                if isinstance(res, list):
                    if not all(isinstance(d, (int, float)) for d in res):
                        raise Exception('resolution should be list of '
                                        'numbers. Is: {}'
                                        .format(direction))
                res_lst.append(meas['resolution'])
            for meshfile, res in zip(mesh_lst, res_lst):
                Mesh_coords = mesh.geometry.x

                xmin = np.min(Mesh_coords[:,0])
                xmax = np.max(Mesh_coords[:,0])
                ymin = np.min(Mesh_coords[:,1])
                ymax = np.max(Mesh_coords[:,1])
                zmin = np.min(Mesh_coords[:,2])
                zmax = np.max(Mesh_coords[:,2])

                #calculate spatial resolution
                padding = 0 if not 'padding' in meas else meas['padding']
                [Nx, Ny, Nz] = [int(np.ceil((xmax-xmin)/res[0]))+padding, int(np.ceil((ymax-ymin)/res[1]))+padding,
                         int(np.ceil((zmax - zmin)/res[2]))+padding]
                if all_vector:
                    np_array = np.zeros([Nx, Ny, Nz, 3])
                else:
                    np_array = np.zeros([Nx, Ny, Nz])
                np_lst.append(np_array)
                
            self._observation_np_aux_fun_lst = fun_lst
            self._observation_res_lst = res_lst
            return np_lst

        return fun_lst

    def init_parameters(self):
        ''' ROUKF interface: Initialize parameters.

        Returns:
            tuple: tuple containing

                * theta_arr (numpy.ndarray):  numpy array of initial conditions
                  of parameters, in correct order
                * theta_sd_arr (numpy.ndarray):  numpy array with corresp.
                  standard deviations
        '''
        bc_param_lst = self.options['estimation']['boundary_conditions']

        self.theta_internal = []
        theta_sd_lst = []
        theta_arr = []

        bid_transp = self.bc_dict['u']['transpiration']['id']
        bid_slip = self.bc_dict['u']['navierslip']['id']

        for bc in bc_param_lst:

            if bc['type'] == 'navierslip':

                if bc['id'] not in bid_slip:
                    raise Exception('Estimation NavierSlip ID does not match '
                                    'ID of boundary condition.')

                self.theta_internal.append(self.bc_dict['u']['navierslip']
                                           ['coef'][bid_slip.index(bc['id'])])
                theta_arr.append(float(self.theta_internal[-1]))
                theta_sd_lst.append(bc['initial_stddev'])

            elif bc['type'] == 'transpiration':

                if bc['id'] not in bid_transp:
                    raise Exception('Estimation Transpiration ID does not '
                                    'match ID of boundary condition.')

                self.theta_internal.append(self.bc_dict['u']['transpiration']
                                           ['coef'][bid_transp.index(bc['id'])]
                                           )
                theta_arr.append(float(self.theta_internal[-1]))
                theta_sd_lst.append(bc['initial_stddev'])

                # have to 'reassemble' the pressure system matrix in
                # self.assemble_pressure()
                self._optimize_robin = True

            elif bc['type'] == 'dirichlet':
                # find DBCs on the corresponding boundary with expression
                i_lst = []
                th_index = []
                for expr_dict in self.bc_dict['u']['dbc_expressions'].values():

                    if expr_dict['id'] == bc['id']:

                        # NOTE XXX
                        # make one copy/entry of the expression for EACH
                        # PARAMETER TO BE OPTIMIZED in that expression !!!
                        # so that the length of the numpy array equals that of
                        # self.theta_internal

                        if not bc['id'] in i_lst:
                            # first time visiting this boundary:
                            for prm in bc['parameters']:

                                assert prm in \
                                    expr_dict['expression'].user_parameters

                                self.theta_internal.append({
                                    'expression_lst': [
                                        expr_dict['expression']],
                                    'parameter': prm
                                })
                                theta_arr.append(
                                    expr_dict['expression'].user_parameters[
                                        prm]
                                )
                                i_lst.append(bc['id'])
                                th_index.append(len(self.theta_internal) - 1)

                        else:
                            # visited this boundary before
                            # instead of appending new data set to
                            # theta_internal, add expression to expression list
                            # of existing parameter

                            i = i_lst.index(bc['id'])

                            if (not self.theta_internal[th_index[i]]
                                    ['parameter'] in bc['parameters']):
                                raise Exception('Parameters dont match!')

                            self.theta_internal[th_index[i]] \
                                ['expression_lst'].append(
                                    expr_dict['expression'])


                if isinstance(bc['initial_stddev'], list):
                    assert (isinstance(bc['parameters'], list) and
                            len(bc['parameters']) == len(bc['initial_stddev']))
                    theta_sd_lst.extend(bc['initial_stddev'])
                else:
                    assert isinstance(bc['parameters'], str), (
                        'Conflichting options: Dimension of expression '
                        'parameters and initial_stddev must match!')
                    theta_sd_lst.append(bc['initial_stddev'])

            elif bc['type'] == 'windkessel':
                if isinstance(bc['parameters'], str):
                    opt_lst = [bc['parameters']]
                elif isinstance(bc['parameters'], list):
                    opt_lst = bc['parameters']
                else:
                    raise Exception('{} not supported'.format(
                                    type(bc['parameters'])))
                
                bid = bc['id']
                wk_dict = self.bc_dict['p']['windkessel']['params']
                assert bid in wk_dict.keys(), 'BC id not in windkessel'

                for prm in opt_lst:
                    assert prm in ('R_d', 'R_p', 'C'), f'prm {prm}'
                    self.theta_internal.append(wk_dict[bid][prm])
                    theta_arr.append(float(self.theta_internal[-1]))

                if not isinstance(bc['initial_stddev'], list):
                    opt_std = [bc['initial_stddev']]
                else:
                    opt_std = bc['initial_stddev']

                if len(opt_std) == len(opt_lst):
                    theta_sd_lst.extend(opt_std)
                else:
                    raise Exception('Required more stddevs')

                self._optimize_windkessel = True
                

            elif bc['type'] == 'mapdd':

                bid = bc['id']
                mdict = self.bc_dict['p']['mapdd']['params'][bid]
                self.theta_internal.append(mdict['l'])
                theta_arr.append(float(self.theta_internal[-1]))
                opt_std = [bc['initial_stddev']]
                theta_sd_lst.extend(opt_std)

            else:
                raise NotImplementedError('BC type "{}" not yet supported for '
                                          'optimization'.format(bc['type']))
        
        theta_arr = np.array(theta_arr)
        theta_sd_arr = np.array(theta_sd_lst)

        return theta_arr, theta_sd_arr

    def observation(self, Xobs_lst):
        ''' Compute observation by applying the observation operator to the
        state, H(X).

        Args:
            Xobs_lst    list of receiving measurement functions
        '''
        if not self._observation_fun_aux_lst:
            Xobs_aux_lst = [None]*len(Xobs_lst)
        else:
            Xobs_aux_lst = self._observation_fun_aux_lst

        if not self._observation_res_lst:
            for i, (Xobs, Xobs_aux) in enumerate(zip(Xobs_lst, Xobs_aux_lst)):
                if Xobs_aux:
                    # Xobs is scalar, Xobs_aux vector

                    direction = (self.options['estimation']['measurements'][i]
                                 ['velocity_direction'])

                    # handle cartesian component selection manually for performance
                    if direction.count(0) == 2 and direction.count(1) == 1:
                        idx = direction.index(1)
                        Xobs.interpolate(self.u_lst[idx])

                    else:
                        assert not Xobs.value_shape(), 'Xobs is not a scalar'
                        # normalize projection direction
                        direction = np.array(direction, dtype=float)
                        direction /= np.sqrt(np.dot(direction, direction))

                        Xobs_aux.interpolate(self.u)

                        bs = Xobs_aux.function_space.dofmap.index_map_bs
                        Xobs_aux_i = [Function(Xobs.function_space) for _ in
                                      range(self.ndim)]
                        Xobs_aux_i[0] = Xobs
                        for k in range(self.ndim):
                            Xobs_aux_i[k].x.array[:] = Xobs_aux.x.array[k::bs]
                        Xobs.x.array[:] *= direction[0]
                        for Xi, d in zip(Xobs_aux_i[1:], direction[1:]):
                            if d:
                                Xobs.x.array[:] += d * Xi.x.array
                else:
                    Xobs.interpolate(self.u)
        else:
            assert type(Xobs_lst[0]) == np.ndarray
            for i, (Xobs, X_fun, Xobs_aux) in enumerate(zip(Xobs_lst, self._observation_np_aux_fun_lst, Xobs_aux_lst)):
                direction = None
                if Xobs_aux:
                    # Xobs is scalar, Xobs_aux vector

                    direction = (self.options['estimation']['measurements'][i]
                                 ['velocity_direction'])    
                X_fun = mritools.mritools.SpatialInterpolation([self.u], self.u.function_space, X_fun.function_space, Xobs_aux, direction)

                padding = 0 if not 'padding' in self.options['estimation']['measurements'][i] else self.options['estimation']['measurements'][i]['padding']
                Xobs_l = mritools.mritools.to_numpy_array(X_fun[0], None, self._observation_res_lst[i], box=False, padding=padding)
                Xobs *= 0
                Xobs += Xobs_l

    def assign_parameters(self, parameters):
        ''' ROUKF interface: Update PDE parameters from ROUKF.

        Args:
            parameters   list of parameters or None
        '''
        if parameters is None:
            return

        if not hasattr(self, 'theta_internal'):
            raise Exception('Need to call init_parameters() before '
                            'assign_parameters()!')

        assert len(self.theta_internal) == len(parameters)
        for th_old, th_new in zip(self.theta_internal, parameters):

            if isinstance(th_old, fem.Constant):
                th_old.value = float(th_new)

            elif isinstance(th_old, dict):
                # parameters are expression parameters
                expr_lst = th_old['expression_lst']
                prms = th_old['parameter']

                for expr in expr_lst:
                    # TODO: expr.user_parameters is legacy; update if needed
                    expr.user_parameters[prms] = th_new
                    # TODO ... dirty/hacky
                    for dbc, dict_ in (self.bc_dict['u']['dbc_expressions']
                                       .items()):
                        if dict_['expression'] == expr:
                            if dbc != 'inflow':
                                self.project_enriched_dbc(dbc, expr)
                                break

            else:
                raise Exception('Parameter type not recognized')
        
    def assign_state(self, state):
        ''' ROUKF interface: Update instance solution functions from state
        variable (inverse to update_state).

        Args:
            state       list of state variables
        '''
        if not state:
            return

        self.u.x.array[:] = state[0].x.array
        self._split_vec_to_lst(self.u, self.u_lst)
        enum_wk = enumerate(self.bc_dict['p']['windkessel']['params'].items())
        for k, (bid, prm) in enum_wk:
            # adding wk pressure if C != 0
            if abs(float(prm['C'])) > 1e-14:
                value = state[k+1].x.array
                mpi_comm = self.u.function_space.mesh.comm

                if mpi_comm.Get_size() > 1:
                    # parallel -- maybe not the best solution, but works
                    # we don't know which proc owns the single dof of the
                    # windkessel state 'real' function space, so gather from
                    # all procs on root=0 and filter
                    value_gathered = mpi_comm.gather(value, root=0)
                    # info(f'DBG: gathered: {value_gathered}')
                    if mpi_comm.Get_rank() == 0:
                        value = list(
                            filter(lambda x: x.size > 0, value_gathered)
                        )
                        assert len(value) == 1
                        value = value[0]
                        # info(f'DBG: filtered: {value}')
                    else:
                        value = np.empty(1)

                    mpi_comm.Bcast(value, root=0)
                    # info(f'DBG: bcast: {value}')

                prm['pi'].value = float(value)
                prm['pi0'].value = float(value)
                
                # info(f"value = {value}, const = {prm['pi'].values()}")

        if (self.options['timemarching']['fractionalstep']['scheme']
                == 'IPCS'):
            assert True, 'experimental placeholder'
            # should work but not tested
            self._split_vec_to_lst(state[1], self.u0_lst)
            self.p.x.array[:] = state[2].x.array

    def init_state(self,state):
        ''' initialize the state variables according inputfile values'''

        state[0].x.array[:] = 0.0  # initial velocity starts from 0
        if self._using_wk:
            enum_wk = enumerate(self.bc_dict['p']['windkessel']['params'].items())
            # update wk part of the state
            for k, (bid, prm) in enum_wk:
                if abs(float(prm['C'])) > 1e-14:
                    value = float(prm['pi'])
                    state[k+1].x.array[:] = value

    def update_state(self, state):
        ''' ROUKF interface: update state variables from solution functions
        (inverse to assign_state).

        Args:
            state       list of state variables
        '''
        if not state:
            return

        state[0].x.array[:] = self.u.x.array
        enum_wk = enumerate(self.bc_dict['p']['windkessel']['params'].items())
        # update wk part of the state
        for k, (bid, prm) in enum_wk:
            if abs(float(prm['C'])) > 1e-14:
                value = float(prm['pi'])
                state[k+1].x.array[:] = value
        # TODO: raise Exception for IPCS and windkessel
        if (self.options['timemarching']['fractionalstep']['scheme']
                == 'IPCS'):
            assert True, 'experimental placeholder'
            assert len(state) == 3, 'expected state space dimension = 3'
            # should work but not tested
            self._merge_lst_to_vec(self.u0_lst, state[1])
            state[2].x.array[:] = self.p.x.array

    # =========================================================================
    # Assembly and solver setup
    # =========================================================================

    def init_assembly(self):
        ''' Initialize, assemble static matrices. '''
        timer = Timer('Z init assembly')
        if self._using_ale:
            # matrices of displacement component space Di
            self.mat['d']['diff'] = _assemble_mat(self.forms['d']['diff'])
            self.mat['d']['div'] = _assemble_mat(self.forms['d']['div'])
            self.vec['d']['rhs_const'] = _assemble_vec(self.forms['d']['rhs_const'])

        # matrices of velocity component space Vi
        self.mat['u']['mass'] = _assemble_mat(self.forms['u']['mass'])
        self.mat['u']['diff'] = _assemble_mat(self.forms['u']['diff'])
        self.mat['u']['rhs'] = self.mat['u']['mass'].copy()

        if self.forms['u']['pres']:
            self.mat['u']['pdiv'] = [_assemble_mat(a) for a in
                                     self.forms['u']['pres']]
        if self.forms['u']['gradp']:
            self.mat['u']['gradp'] = [_assemble_mat(a) for a in
                                      self.forms['u']['gradp']]

        # init convection matrix with its own sparsity pattern (NOT mass.copy()).
        # The conv form (with SUPG) requires ghost DOFs that the mass form does
        # not; copying mass would give an incomplete ghost set and trigger PETSc
        # "Argument out of range" on the first step with non-zero velocity.
        self.mat['u']['conv'] = create_matrix(fem_form(self.forms['u']['conv']))

        # assembling inflow matrices
        if self.forms['u']['inflow_lhs']:
            self.mat['u']['inflow'] = self.mat['u']['mass'].copy()
            _assemble_mat(self.forms['u']['inflow_lhs'],
                          mat=self.mat['u']['inflow'])
        else:
            self.mat['u']['inflow'] = None

        # assembling fnv matrices
        if 'fnv' in self.forms['u'].keys():
            self.mat['u']['fnv'] = [self.mat['u']['mass'].copy()
                                    for _ in range(self.ndim)]
            for i in range(self.ndim):
                _assemble_mat(self.forms['u']['fnv'][i][0],
                              mat=self.mat['u']['fnv'][i])
        else:
            self.mat['u']['fnv'] = None

        self.vec['u']['rhs_const'] = [_assemble_vec(form) if form else None for
                                      key, form in
                                      self.forms['u']['neumann'].items()]

        if 'fnv' in self.forms['u'].keys():
            self.vec['u']['fnv'] = [_assemble_vec(form[1]) if form else None for
                                    key, form in
                                    self.forms['u']['fnv'].items()]
        else:
            self.vec['u']['fnv'] = None

        self.vec['u']['rhs_inflow'] = [_assemble_vec(form) if form else None for
                                       key, form in
                                       self.forms['u']['inflow'].items()]

        self.vec['u']['rhs_mapdd'] = [_assemble_vec(form) if form else None for
                                      key, form in
                                      self.forms['u']['mapdd'].items()]

        # matrices of pressure space Q
        # right hand side matrix to be multiplied by u.x.petsc_vec
        # div(u) and in case of transpiration BCs: dot(u, n) term
        self.mat['p']['rhs_u'] = _assemble_mat(self.forms['p']['rhs_u'])
        self.mat['p']['laplacian'] = _assemble_mat(self.forms['p']['laplacian'])

        # apply bdry conditions to pressure Laplacian
        _apply_dbc_to_mat(self.mat['p']['laplacian'],
                          self.bc_dict['p']['dirichlet'])
        # if windkessel, apply bdry conditions for explicit
        # or assemble LRC matrix for implicit method
        if self._using_wk:
            if not self.wk['implicit']:
                _apply_dbc_to_mat(self.mat['p']['laplacian'],
                                  self.bc_dict['p']['windkessel']['dirichlet'])
            else:
                self.assembly_windkessel(self.mat['p']['laplacian'])

        if self.forms['p']['neumann']:
            self.vec['p']['rhs_const'] = _assemble_vec(self.forms['p']['neumann'])
        else:
            self.vec['p']['rhs_const'] = 0

        # Transpiration and Navier-Slip matrices
        self.init_assembly_robin()

        del timer
        self._init_assembly_done = True

    def assembly_windkessel(self, A):
        ''' Initialize and assemble vectors and matrices required for implicit
        Windkessel formulation, using low rank update (Woodbury).

                A + UDU'            (1)

        where A is the n x n matrix, U is a tall, thin matrix n x m
        (p*ds(i)), with 1 column per Windkessel boundary and D diagonal
        m x m matrix with the variable coefficients.

        Args:
            A (PETScMatrix):    Assembled form

        NOTE:
        A NEW SYSTEM MATRIX (1) DEFINED WITH PETSc().Mat().createLRC
        REPLACES THE USUAL LAPLACIAN IN THE POISSON PROBLEM.
        THE NEW MATRIX CONTAINS ONLY REFERENCES TO A, U AND D, THUS
        NOR EXPLICITLY CREATED!
        '''

        self.logger.info('Creating LRC matrix from windkessel BCs')
        u_lst = []
        fac_l = []
        for bid, prm in self.bc_dict['p']['windkessel']['params'].items():
            u_lst.append(_assemble_vec(self.forms['p']['windkessel_lhs'][bid]))
            fac_l.append(float(prm['delta_l']))

        U_arr = np.hstack([v.getArray() for v in u_lst])
        diag = np.array(fac_l)

        imap = self.p.function_space.dofmap.index_map
        sizes_local = imap.size_local
        sizes_global = imap.size_global
        comm = self.p.function_space.mesh.comm
        ncol = len(u_lst)
        U = PETSc.Mat().createDense(
            size=((sizes_local, sizes_global), (ncol, ncol)),
            array=U_arr.reshape((sizes_local, ncol)), comm=comm
        )
        U.setUp()
        U.assemble()

        vec = PETSc.Vec().createSeq(size=len(diag))
        vec.setArray(diag)
        vec.assemble()
        self.vec['p']['windkessel_lhs_lrc_diag'] = vec
        self.mat['p']['windkessel_lhs_lrc'] = PETSc.Mat().createLRC(
            A, U, vec, U
        )

    def init_assembly_robin(self):
        ''' Initialize and assemble matrices related to Robin BCs
        (Navierslip/Transpiration) '''

        sparse_pat = self.mat['u']['mass']

        for i in range(self.ndim):
            tmp_form = 0
            tmp_dict = {}
            for i_bnd in range(len(self.forms['u']['navierslip']['coef'])):
                navslip_forms = self.forms['u']['navierslip']
                if not self._optimizing:
                    if navslip_forms[i][i_bnd]['semi-implicit']:
                        tmp_form += (navslip_forms['coef'][i_bnd] *
                                     navslip_forms[i][i_bnd]['semi-implicit'])
                    for j, a_ex in navslip_forms[i][i_bnd]['explicit'].items():
                        if j not in tmp_dict:
                            tmp_dict[j] = 0
                        tmp_dict[j] += navslip_forms['coef'][i_bnd]*a_ex
                else:
                    if navslip_forms[i][i_bnd]['semi-implicit']:
                        self.mat['u']['lhs_navslip'][i].append(
                            _assemble_mat(navslip_forms[i][i_bnd]['semi-implicit'],
                                          mat=sparse_pat.copy()))
                    self.mat['u']['rhs_navslip'][i].append(
                        {j: _assemble_mat(a) for j, a in
                         navslip_forms[i][i_bnd]['explicit'].items()})

            if not self._optimizing:
                self.mat['u']['rhs_navslip'][i].append(
                    {j: _assemble_mat(a) for j, a in
                     tmp_dict.items()})

                if tmp_form:
                    self.mat['u']['lhs_navslip'][i].append(
                        _assemble_mat(tmp_form, mat=sparse_pat.copy()))

                    # # DBG
                    # self.logger.warn('tmp_form: {}'.format(tmp_form))
                    # self.logger.warn('i = {}: ||Ai|| = {}'.format(
                    #     i, np.linalg.norm(
                    #         self.mat['u']['lhs_navslip'][i][-1].array())))
                    # #

        for i in range(self.ndim):
            tmp_form_u = 0
            tmp_form_p = 0
            tmp_dict = {}
            for i_bnd in range(len(self.forms['u']['transpiration']['coef'])):
                trans_forms = self.forms['u']['transpiration']
                if not self._optimizing:
                    if trans_forms[i][i_bnd]['semi-implicit']:
                        tmp_form_u += (trans_forms['coef'][i_bnd] *
                                       trans_forms[i][i_bnd]['semi-implicit'])
                    for j, a_ex in (
                            trans_forms[i][i_bnd]['explicit'].items()):
                        if j not in tmp_dict:
                            tmp_dict[j] = 0
                        tmp_dict[j] += trans_forms['coef'][i_bnd]*a_ex
                else:
                    if trans_forms[i][i_bnd]['semi-implicit']:
                        self.mat['u']['lhs_trans'][i].append(
                            _assemble_mat(trans_forms[i][i_bnd]['semi-implicit'],
                                          mat=sparse_pat.copy()))
                    self.mat['u']['rhs_trans'][i].append(
                        {j: _assemble_mat(a) for j, a in
                         trans_forms[i][i_bnd]['explicit'].items()})

                # in the case of CT, add (pn, vn)*ds(i) to boundary form
                if (self.options['timemarching']['fractionalstep']['scheme'] ==
                        'CT'):
                    tmp_form_p += trans_forms[i][i_bnd]['pressure']

            if not self._optimizing:
                self.mat['u']['rhs_trans'][i].append(
                    {j: _assemble_mat(a) for j, a in
                     tmp_dict.items()})
                if tmp_form_u:
                    self.mat['u']['lhs_trans'][i].append(
                        _assemble_mat(tmp_form_u, mat=sparse_pat.copy()))
            if tmp_form_p:
                self.mat['u']['p_trans'][i] = _assemble_mat(tmp_form_p)

        # PRESSURE BC
        self.mat['p']['mass_robin'] = [
            _assemble_mat(a, mat=self.mat['p']['laplacian'].copy()) for a in
            self.forms['p']['robin']]

        self.mat['p']['u_norm_bound'] = [
            _assemble_mat(a) for a in
            self.forms['p']['transpiration_dirichlet_u']]
        for a in self.forms['p']['transpiration_dirichlet_p']:
            Atmp = _assemble_mat(a)
            # ensure diagonal DOFs are non-zero (replaces ident_zeros)
            Atmp.setOption(PETSc.Mat.Option.KEEP_NONZERO_PATTERN, True)
            self.mat['p']['mass_bound'].append(Atmp)

        # set corresponding rows to zero in Robin mass matrices (if any)
        for mat in self.mat['p']['mass_robin']:
            _zero_mat_rows(mat, self.bc_dict['p']['dirichlet'])

        if not self._optimizing:
            for mat, coef in zip(self.mat['p']['mass_robin'],
                                 self.bc_dict['p']['transpiration']['coef']):
                self.mat['p']['laplacian'].axpy(1./float(coef), mat)
        else:
            # raise Exception('Robin BC optimization not implemented for PPE')
            pass

    def init_solvers(self):
        ''' Initialize linear system solvers '''
        if not (hasattr(self, '_init_assembly_done') and
                self._init_assembly_done):
            raise Exception('init_assembly() must be called before '
                            'init_solvers()')

        self.iterations_ksp = {}
        self.residuals_ksp = {}

        if self._using_ale:
            self.solver_d = PETScSolver(self.options, 'd_',
                                        self._logging_filehandler,
                                        verbose=True)

            self.iterations_ksp.update({'d': []})
            self.residuals_ksp.update({'d': []})

        self.solver_u_ten = PETScSolver(self.options, 'u_ten_',
                                        self._logging_filehandler,
                                        verbose=True)

        self.solver_p = PETScSolver(self.options, 'p_',
                                    self._logging_filehandler,
                                    verbose=True)

        if self._using_wk:
            if not self.wk['implicit']:
                self.solver_p.set_operator(
                        self.mat['p']['laplacian'])
            else:
                self.update_windkessel_LRC()
                self.solver_p.set_operator(
                        self.mat['p']['windkessel_lhs_lrc'],
                        self.mat['p']['laplacian'])
        else:
            self.solver_p.set_operator(self.mat['p']['laplacian'])

        self.solver_p_mass = []
        for M in self.mat['p']['mass_bound']:
            self.solver_p_mass.append(PETScSolver(self.options, 'p_mass_',
                                                  self._logging_filehandler,
                                                  verbose=True))
            self.solver_p_mass[-1].set_operator(M)

        self.solver_u_upd = PETScSolver(self.options, 'u_upd_',
                                        self._logging_filehandler,
                                        verbose=True)

        Aupd = self.mat['u']['mass'].copy()
        if self._applying_dc_on_update:
            if self.bc_dict['u']['same_dbc_boundaries']:
                _apply_dbc_rows_to_mat(Aupd, self.bc_dict['u']['dirichlet'][0])
        self.solver_u_upd.set_operator(Aupd)

        self.iterations_ksp.update({
            'u_ten': {i: [] for i in range(self.ndim)},
            'u_upd': {i: [] for i in range(self.ndim)},
            'p': []
        })
        self.residuals_ksp.update({
            'u_ten': {i: [] for i in range(self.ndim)},
            'u_upd': {i: [] for i in range(self.ndim)},
            'p': []
        })
        # TODO remove nullspace if no dirichlet BC!

    # =========================================================================
    # Displacement step  (ALE only)
    # =========================================================================

    def assemble_displacement(self):
        ''' Assemble changing matrices for displacement. '''
        if not getattr(self, '_assembled_d', False):
            A = self.mat['d']['diff'] # copy ?
            A.axpy(1., self.mat['d']['div'])
            for i in range(self.ndim):
                # Rows-only elimination: zeroing columns too (the old
                # _apply_dbc_to_mat) decouples the interior block from the
                # boundary data, and build_rhs_displacement has no
                # apply_lifting to compensate — the lifting solve then
                # returns d=0 everywhere off the interface and the ALE mesh
                # never follows the flag (cells collapse at ~one-layer
                # displacement).
                _apply_dbc_rows_to_mat(A, self.bc_dict['d']['dirichlet'][i])

            self.solver_d.set_operator(A)
            self._assembled_d = True

    def build_rhs_displacement(self):
        ''' Build RHS vector for displacement solve. '''
        bd = self.vec['d']['rhs_const']
        for i in range(self.ndim):
            set_bc(bd, self.bc_dict['d']['dirichlet'][i])

        return bd

    def solve_displacement(self):
        ''' Solve displacement problem '''
        timer = Timer('Z solve d')
        self.logger.info('Solve displacement')
        self.update_displacement_bcs()

        self.assemble_displacement()
        bd = self.build_rhs_displacement()

        self.solver_d.solve(self.d.x.petsc_vec, bd)

        if self.solver_d.conv_reason < 0:
            self.logger.error('Solver d DIVERGED ({})'.
                                format(self.solver_d.conv_reason))
            if len(self.iterations_ksp['d']) > 0:
                self._diverged = True

        self.iterations_ksp['d'].append(self.solver_d.iterations)
        self.residuals_ksp['d'].append(self.solver_d.residuals)

        # ALE mesh-quality guard: min deformed/reference cell area (det of
        # the ALE deformation gradient, DG0). A "successful" lifting solve
        # can still be wrong (the 2026-06 frozen-interior bug stayed silent
        # for weeks because nothing watched the cells) — print, don't trust.
        if not hasattr(self, '_alej_func'):
            from dolfinx.fem import functionspace as _fs, Function as _F, \
                Expression as _E
            V0 = _fs(self.d.function_space.mesh, ('DG', 0))
            Id = ufl.Identity(self.ndim)
            J = ufl.det(Id + ufl.grad(self.d))
            self._alej_func = _F(V0)
            self._alej_expr = _E(J, V0.element.interpolation_points)
        self._alej_func.interpolate(self._alej_expr)
        jmin = self.d.function_space.mesh.comm.allreduce(
            self._alej_func.x.array.min(), op=MPI.MIN)
        if jmin < 0.05:
            print('    [ale-guard] min cell area ratio = {:.3f} — mesh '
                  'near-degenerate, fields at the interface are garbage'
                  .format(jmin), flush=True)
        elif jmin < 0.2:
            print('    [ale-guard] min cell area ratio = {:.3f}'
                  .format(jmin), flush=True)

        timer.stop()

    # =========================================================================
    # Tentative velocity step
    # =========================================================================

    def assemble_tentative_velocity(self, i=None):
        ''' Assemble changing matrices for tentative velocity solve. '''
        # Note: Matrices are stored into mat['u']['conv'] and mat['u']['rhs']
        # A: system matrix. Assemble convection and add mass/diffusion
        if self.options['timemarching']['fractionalstep']['scheme'] == 'IPCS':
            sd_opt = (self.options['fem']['stabilization']
                      ['streamline_diffusion'])
            if (sd_opt['parameter'] in ('standard', 'default', 'klr') or
                    (sd_opt['parameter'] == 'shakib' and
                     'parameter_element_constant' in sd_opt and
                     sd_opt['parameter_element_constant'])):
                with Timer('Z assign conv'):
                    for uci, ui, u0i in zip(self._u_tmp_lst, self.u_lst,
                                            self.u0_lst):
                        uci.x.array[:] = 2*ui.x.array - u0i.x.array
                    self._merge_lst_to_vec(self._u_tmp_lst, self.u_conv_assigned)

        # Refresh element-constant SUPG tau from the current u_conv_assigned
        if self._sd_param is not None:
            self._sd_param.update_tau()

        # In ALE mode: recreate conv matrix each step (pre-allocated ghost set
        # becomes invalid after mesh deformation → PETSc "Argument out of range").
        if self._using_ale:
            self.mat['u']['conv'] = _assemble_mat(self.forms['u']['conv'])
        else:
            _assemble_mat(self.forms['u']['conv'], mat=self.mat['u']['conv'])
        A = self.mat['u']['conv']

        if (self.options['timemarching']['fractionalstep']['scheme'] == 'IPCS'
                and self.t > self.options['timemarching']['dt'] + 1e-14):
            cf = 1.5
        else:
            cf = 1

        if self._using_ale:
            _assemble_mat(self.forms['u']['mass'], mat=self.mat['u']['mass'])
            _assemble_mat(self.forms['u']['diff'], mat=self.mat['u']['diff'])
            # mat['u']['rhs'] changes over time when using ALE
            self.mat['u']['rhs'] = self.mat['u']['mass'].copy()

        A.axpy(cf, self.mat['u']['mass'], structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
        A.axpy(1., self.mat['u']['diff'], structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)

        if self.mat['u']['inflow']:
            # adding inflow lhs term if defined
            A.axpy(1., self.mat['u']['inflow'])

        if 'fnv_type' in self.forms['u'].keys() and self.forms['u']['fnv_type'] == 'explicit':
            # adding one arbitrary LHS of the fnv matrices
            A.axpy(1., self.mat['u']['fnv'][0])

        # assemble time/RHS SUPG matrices if present, add mass matrix
        # mat['u']['rhs'] was initialized to 'mass' and won't be changed
        # whenever ALE is not used
        if 'supg_time' in self.forms['u'] and self.forms['u']['supg_time']:
            assert True, 'supg_time should not be useable!!'
            _assemble_mat(self.forms['u']['supg_time'],
                          mat=self.mat['u']['rhs'])
            A.axpy(1., self.mat['u']['rhs'])
            self.mat['u']['rhs'].axpy(1., self.mat['u']['mass'])

        if self.bc_dict['u']['same_dbc_boundaries'] and not self._using_fnv_semi_implicit:
            _apply_dbc_rows_to_mat(A, self.bc_dict['u']['dirichlet'][0])

        return A

    def build_rhs_tentative_velocity(self, i):
        ''' Build RHS vector for tentative velocity solve, for the i'th
        component. '''
        if self.options['timemarching']['fractionalstep']['scheme'] == 'CTp':
            raise DeprecationWarning('CTp solver deprecated')
            x_i = self.u0_lst[i].x.petsc_vec
        elif (self.options['timemarching']['fractionalstep']['scheme'] ==
              'IPCS'):
            # convection matrix was assembled already, so we can safely
            # overwrite u0 (time step k-1) for the RHS
            tmp = self.u_lst[i].x.petsc_vec.copy()
            tmp.scale(2.0)
            tmp.axpy(-0.5, self.u0_lst[i].x.petsc_vec)
            x_i = tmp
        else:
            # CT
            if self._using_ale:
                x_i = self.upd_lst[i].x.petsc_vec
            elif self._using_mapdd:
                x_i = self.u0_lst[i].x.petsc_vec
            else:
                x_i = self.u_lst[i].x.petsc_vec

        bu_i = _mat_vec(self.mat['u']['rhs'], x_i)

        if self.mat['u']['pdiv']:
            if self._using_ale:
                [_assemble_mat(a, mat=A) for a, A in
                    zip(self.forms['u']['pres'], self.mat['u']['pdiv'])]
            bu_i.axpy(-1.0, _mat_vec(self.mat['u']['pdiv'][i], self.p.x.petsc_vec))

        if self.vec['u']['rhs_const'][i]:
            bu_i.axpy(1.0, self.vec['u']['rhs_const'][i])

        if self.vec['u']['rhs_inflow'][i]:
            self.vec['u']['rhs_inflow'][i] = _assemble_vec(self.forms['u']['inflow'][i])
            bu_i.axpy(1.0, self.vec['u']['rhs_inflow'][i])

        if self.vec['u']['rhs_mapdd'][i]:
            self.vec['u']['rhs_mapdd'][i] = _assemble_vec(self.forms['u']['mapdd'][i])
            bu_i.axpy(1.0, self.vec['u']['rhs_mapdd'][i])

        if self.vec['u']['fnv']:
            self.vec['u']['fnv'][i] = _assemble_vec(self.forms['u']['fnv'][i][1])
            bu_i.axpy(1.0, self.vec['u']['fnv'][i])

        if self.forms['u'].get('fsi_robin_rhs'):
            # FSI Robin-Neumann interface data (alpha*v_rb + t_rb)*w*js*ds:
            # v_rb/t_rb are updated by the coupler every sub-iteration and
            # js depends on the ALE deformation — reassemble each solve.
            bu_i.axpy(1.0, _assemble_vec(self.forms['u']['fsi_robin_rhs'][i]))

        return bu_i

    def assemble_robin_tentative_velocity(self, A, bu_i, i):
        ''' Add robin terms to A and bu_i '''
        if not (self.forms['u']['navierslip']['coef'] or
                self.forms['u']['transpiration']['coef']):
            return None

        if self.mat['u']['lhs_navslip'][0] or self.mat['u']['lhs_trans'][0]:
            A_robin = A.copy()
            implicit = True
        else:
            A_robin = None
            implicit = False

        if i == 0:
            self.x_i_presolve = [ui.x.petsc_vec.copy() for ui in self.u_lst]

        if not self._optimizing:
            if len(self.mat['u']['lhs_navslip'][i]):
                A_robin.axpy(1., self.mat['u']['lhs_navslip'][i][0])
            if len(self.mat['u']['lhs_trans'][i]):
                A_robin.axpy(1., self.mat['u']['lhs_trans'][i][0])

            for rhs_ns in self.mat['u']['rhs_navslip'][i]:
                for j, Nj in rhs_ns.items():
                    bu_i.axpy(-1., _mat_vec(Nj, self.x_i_presolve[j]))
            for rhs_t in self.mat['u']['rhs_trans'][i]:
                for j, Tj in rhs_t.items():
                    bu_i.axpy(-1., _mat_vec(Tj, self.x_i_presolve[j]))

        else:
            for i_bnd in range(len(self.forms['u']['navierslip']['coef'])):
                if len(self.mat['u']['rhs_navslip'][i]):
                    rhs_ns = self.mat['u']['rhs_navslip'][i][i_bnd]
                    coef_ns = float(self.forms['u']['navierslip']['coef']
                                    [i_bnd])

                    if implicit:
                        lhs_ns = self.mat['u']['lhs_navslip'][i][i_bnd]
                        A_robin.axpy(coef_ns, lhs_ns)

                    for j, Nj in rhs_ns.items():
                        bu_i.axpy(-coef_ns, _mat_vec(Nj, self.x_i_presolve[j]))

                if len(self.mat['u']['rhs_trans'][i]):
                    rhs_t = self.mat['u']['rhs_trans'][i][i_bnd]
                    coef_t = float(self.forms['u']['transpiration']['coef']
                                   [i_bnd])

                    if implicit:
                        lhs_t = self.mat['u']['lhs_trans'][i][i_bnd]
                        A_robin.axpy(coef_t, lhs_t)

                    for j, Tj in rhs_t.items():
                        bu_i.axpy(-coef_t, _mat_vec(Tj, self.x_i_presolve[j]))

        if self.mat['u']['p_trans'][i]:
            assert (self.options['timemarching']['fractionalstep']['scheme'] ==
                    'CT')
            bu_i.axpy(-1., _mat_vec(self.mat['u']['p_trans'][i], self.p.x.petsc_vec))

        # if A_robin:
        #     # just in case XXX check if necessary
        #     # this is now done in solve_tentative_velocity!
        #     [bc.apply(A_robin) for bc in self.bc_dict['u']['dirichlet'][i]]
        return A_robin

    def assemble_forcing_normal_velocity(self, A, i):
        ''' Add fnv terms to A and bu_i '''
        
        if not self._using_fnv_semi_implicit:
            return None
        else:
            A_fnv = A.copy()
            _assemble_mat(self.forms['u']['fnv'][i][0],
                          mat=self.mat['u']['fnv'][i])
            A_fnv.axpy(1., self.mat['u']['fnv'][i])
            return A_fnv

    def solve_tentative_velocity(self):
        ''' Solve tentative velocity PDE '''
        timer = Timer('Z solve u_ten')
        self.update_velocity_bcs()

        A = self.assemble_tentative_velocity()

        if self.bc_dict['u']['same_dbc_boundaries'] and not self._using_fnv_semi_implicit:
            self.solver_u_ten.set_operator(A)

        for i, u_i in enumerate(self.u_lst):

            bu_i = self.build_rhs_tentative_velocity(i)

            A_robin = self.assemble_robin_tentative_velocity(A, bu_i, i)
            A_fnv = self.assemble_forcing_normal_velocity(A, i)

            if (self.options['timemarching']['fractionalstep']['scheme'] in
                    ['IPCS', 'CT']) and not self._using_mapdd:
                # assembly of A, b done; u0_lst is "free"; assign
                # u_i(corrected, u^k) for next iteration
                # assign also when CT for FSI 
                self.u0_lst[i].x.array[:] = u_i.x.array

            if self._using_mapdd:
                # start with different previous velocity
                self.u0_mapdd_lst[i].x.array[:] = self.u0_lst[i].x.array

            if A_robin:
                _apply_dbc_to_mat(A_robin, self.bc_dict['u']['dirichlet'][i])
                apply_lifting(bu_i, [fem_form(self.forms['u']['mass'])],
                              [self.bc_dict['u']['dirichlet'][i]])
                set_bc(bu_i, self.bc_dict['u']['dirichlet'][i])
                self.solver_u_ten.set_operator(A_robin)
            elif A_fnv:
                _apply_dbc_to_mat(A_fnv, self.bc_dict['u']['dirichlet'][i])
                apply_lifting(bu_i, [fem_form(self.forms['u']['mass'])],
                              [self.bc_dict['u']['dirichlet'][i]])
                set_bc(bu_i, self.bc_dict['u']['dirichlet'][i])
                self.solver_u_ten.set_operator(A_fnv)
            elif self.bc_dict['u']['same_dbc_boundaries']:
                # A already has DBC applied (zeroRowsColumns) from
                # assemble_tentative_velocity; lifting is a no-op.
                set_bc(bu_i, self.bc_dict['u']['dirichlet'][i])
            else:
                A_cpy = A.copy()
                _apply_dbc_to_mat(A_cpy, self.bc_dict['u']['dirichlet'][i])
                apply_lifting(bu_i, [fem_form(self.forms['u']['mass'])],
                              [self.bc_dict['u']['dirichlet'][i]])
                set_bc(bu_i, self.bc_dict['u']['dirichlet'][i])
                self.solver_u_ten.set_operator(A_cpy)

            self.logger.info('\t component {}'.format(i))

            self.solver_u_ten.solve(u_i.x.petsc_vec, bu_i)

            if self.solver_u_ten.conv_reason < 0:
                self.logger.error('Solver u_ten {} DIVERGED ({})'.
                                  format(i, self.solver_u_ten.conv_reason))
                if len(self.iterations_ksp['u_ten'][i]) > 0:
                    self._diverged = True

            self.iterations_ksp['u_ten'][i].append(
                self.solver_u_ten.iterations)
            self.residuals_ksp['u_ten'][i].append(self.solver_u_ten.residuals)


        timer.stop()


        with Timer('Z assign'):
            self._merge_lst_to_vec(self.u_lst, self.u)

    # =========================================================================
    # Pressure projection step
    # =========================================================================

    def build_rhs_pressure(self):
        ''' Build RHS vector for pressure projection solve. '''
        if (self.options['timemarching']['fractionalstep']['scheme'] == 'IPCS'
                and self.t > self.options['timemarching']['dt'] + 1e-14):
            cf = 1.5
        else:
            cf = 1.
        bp = _mat_vec(self.mat['p']['rhs_u'], self.u.x.petsc_vec)
        bp.scale(cf)

        if self.forms['p'].get('fsi_si_rhs') is not None:
            # FSI semi-implicit projection coupling: Neumann data
            # rho*k*(u_tent - v_si)·n on the interface (see form_pressure).
            # v_si changes every FSI sub-iteration — assemble fresh.
            bp.axpy(1.0, _assemble_vec(self.forms['p']['fsi_si_rhs']))

        if self.vec['p']['rhs_const']:
            bp.axpy(1.0, self.vec['p']['rhs_const'])
        # check neumann form again for compatibility with FSI
        elif self.forms['p']['neumann'] and self.ale['type'] == 'external':
            bp.axpy(1.0, _assemble_vec(self.forms['p']['neumann']))

        if self._using_wk and self.wk['implicit']:
            self.vec['p']['windkessel_rhs'] = _assemble_vec(self.forms['p']['windkessel_rhs'])
            bp.axpy(1.0, self.vec['p']['windkessel_rhs'])


        apply_lifting(bp, [fem_form(self.forms['p']['laplacian'])],
                      [self.bc_dict['p']['dirichlet']])
        set_bc(bp, self.bc_dict['p']['dirichlet'])

        if self._using_wk and not self.wk['implicit']:
            set_bc(bp, self.bc_dict['p']['windkessel']['dirichlet'])
        

        # BCs are applied to Laplacian in init_assembly() !
        return bp

    def assemble_pressure(self):
        ''' Assemble matrix of pressure projection, in the case of variable
        coefficient Robin boundary conditions. '''
        
        if self._using_ale or self._using_mapdd:
            A = self.mat['p']['laplacian'].copy()
            _assemble_mat(self.forms['p']['laplacian'], mat=A)
            _apply_dbc_to_mat(A, self.bc_dict['p']['dirichlet'])

            # The divergence RHS operator div(J*F^{-1}*u) depends on the ALE
            # deformation d through J and F — it must track the same geometry
            # as the Laplacian, otherwise the projection enforces div(u)=0 on
            # a stale mesh and the mismatch acts as a spurious mass source
            # scaled by k=1/dt.
            _assemble_mat(self.forms['p']['rhs_u'], mat=self.mat['p']['rhs_u'])

            if self._using_wk:
                if not self.wk['implicit']:
                    _apply_dbc_to_mat(A, self.bc_dict['p']['windkessel']['dirichlet'])
                    self.solver_p.set_operator(A)
                else:
                    self.assembly_windkessel(A)
                    self.update_windkessel_LRC()
                    self.solver_p.set_operator(
                        self.mat['p']['windkessel_lhs_lrc'], A)
            else:
                self.solver_p.set_operator(A)


        if self._optimize_robin and not self._using_ale:
            if self._using_wk and self.wk['implicit']:
                # laplacian matrix wrongly updated!
                raise Exception('Robin BCs not suported with '
                                'implicit windkessel')
            self.assemble_pressure_robin_optim()

        # FIXME more elegant implementation for this when optimizing ....
        # if self._optimize_robin and not self._using_ale:
        #     A = self.mat['p']['laplacian'].copy()
        #     for mat, coef in zip(self.mat['p']['mass_robin'],
        #                          self.bc_dict['p']['transpiration']['coef']):
        #         A.axpy(\1)
        #     self.solver_p.set_operator(A)

    def assemble_pressure_windkessel_impl(self):
        ''' Assemble system matrix of pressure projection when implicit Windkessel
        formulation is used, set operator. '''
        if not self.wk['implicit']:
            return

        with Timer('Z assemble wk (impl) LHS'):
            self.logger.info('Assemble implicit WK LHS -- LRC')
            A = self.mat['p']['laplacian'].copy()
            self.update_windkessel_LRC()
            self.solver_p.set_operator(
                self.mat['p']['windkessel_lhs_lrc'], A)

    def assemble_pressure_robin_optim(self):
        ''' Assemble system matrix of pressure projection, in the case of
        variable coefficient Robin boundary conditions and set operator. '''
        # if not self._optimize_robin:
        #     return
        A = self.mat['p']['laplacian'].copy()
        for mat, coef in zip(self.mat['p']['mass_robin'],
                             self.bc_dict['p']['transpiration']['coef']):
            A.axpy(1./float(coef), mat)
        self.solver_p.set_operator(A)

    def solve_pressure(self):
        ''' Projection step. Solve pressure poisson equation. '''
        timer = Timer('Z solve p')

        self.update_pressure_bcs()
        # FIXME: necessary to adapt this if Robin coef. changes!
        self.assemble_pressure()
        bp = self.build_rhs_pressure()
        # self.logger.info('|bp|2: {}'.format(np.linalg.norm(bp.get_local())))

        self.solver_p.solve(self.phi.x.petsc_vec, bp)
        # self.logger.info('|p|2: {}'.format(np.linalg.norm(self.phi.x.array)))
        # self.logger.debug(f"|p| = {norm(self.phi)}")

        if self.solver_p.conv_reason < 0:
            self.logger.error('Solver p DIVERGED ({})'.
                              format(self.solver_p.conv_reason))
            self._diverged = True
        self.iterations_ksp['p'].append(self.solver_p.iterations)
        self.residuals_ksp['p'].append(self.solver_p.residuals)

        timer.stop()

    # =========================================================================
    # Velocity update step
    # =========================================================================

    def build_rhs_velocity_update(self, i):
        ''' Build RHS vector for tentative velocity solve, for the i'th
            component. '''
        if (self.options['timemarching']['fractionalstep']['scheme'] == 'IPCS'
                and self.t > self.options['timemarching']['dt'] + 1e-14):
            # inverse of 1.5, as multiplied on RHS
            cf = 2./3.
        else:
            cf = 1.

        A  = self.mat['u']['gradp'][i]
        if self._using_ale:
            _assemble_mat(self.forms['u']['gradp'][i], mat=A)

        bu_i = _mat_vec(A, self.phi.x.petsc_vec)
        bu_i.scale(cf)

        if self._applying_pen_on_update:
            if self.vec['u']['fnv']:
                self.vec['u']['fnv'][i] = _assemble_vec(self.forms['u']['fnv'][i][1])
                bu_i.axpy(1.0, self.vec['u']['fnv'][i])

        return bu_i

    def solve_velocity_update(self):
        ''' Solve velocity update PDE '''

        timer = Timer('Z solve u_upd')


        for i, u_i in enumerate(self.u_lst):
            self.logger.info('\t component {}'.format(i))
            if (self.options['timemarching']['fractionalstep']['scheme']
                    == 'CTp') or self._using_mapdd:
                # CTp: use old tentative velocity in time disc. term on RHS
                self.u0_lst[i].x.array[:] = u_i.x.array
            #self.solver_u_upd.set_operator(self.mat['u']['mass'])
            bu_i = self.build_rhs_velocity_update(i)

            if self._applying_dc_on_update:
                # modifying the update mass matrix if need it and rhs
                if not self.bc_dict['u']['same_dbc_boundaries']:
                    Aupd = self.mat['u']['mass'].copy()
                    if self._applying_pen_on_update:
                        if 'fnv_type' in self.forms['u'].keys() and self.forms['u']['fnv_type'] == 'explicit':
                            # adding one arbitrary LHS of the fnv matrices
                            Aupd.axpy(1., self.mat['u']['fnv'][0])

                    _apply_dbc_to_mat(Aupd, self.bc_dict['u']['dirichlet'][i])
                    self.solver_u_upd.set_operator(Aupd)

                set_bc(bu_i, self.bc_dict['u']['dirichlet'][i])
            
            self.solver_u_upd.solve(self.du[i].x.petsc_vec, bu_i)

            if not self._using_ale:
                u_i.x.petsc_vec.axpy(1.0, self.du[i].x.petsc_vec)
            else:
                self.upd_lst[i].x.array[:] = u_i.x.array
                self.upd_lst[i].x.petsc_vec.axpy(1.0, self.du[i].x.petsc_vec)
                # ALE: advance u_lst to the corrected (divergence-free)
                # velocity so the next-step BDF2 extrapolation
                # u_conv = 2*u_n - u_{n-1} uses physical velocities, not
                # stale tentative u*.  Without this, when save/restore snapshots
                # u_lst and line 1656 writes u0_lst ← u_lst, the BDF2 history
                # is corrupted with stale tentative velocities.
                u_i.x.array[:] = self.upd_lst[i].x.array
                u_i.x.scatter_forward()

                # Re-impose velocity Dirichlet BCs on the corrected velocity:
                # the update solve has no Dirichlet rows, so
                # du = -(1/(rho*k))*M^{-1}*G*phi is nonzero on Dirichlet
                # boundaries wherever grad(phi) != 0 there. Any pressure
                # boundary layer/spike (e.g. at a moving-wall corner) then
                # overwrites the wall velocity BC, and the polluted u feeds
                # the convection history and FSI traction.
                set_bc(u_i.x.petsc_vec, self.bc_dict['u']['dirichlet'][i])
                set_bc(self.upd_lst[i].x.petsc_vec,
                       self.bc_dict['u']['dirichlet'][i])
                u_i.x.scatter_forward()
                self.upd_lst[i].x.scatter_forward()

            if self.solver_u_upd.conv_reason < 0:
                self.logger.error('Solver u_upd {} DIVERGED ({})'.
                                  format(i, self.solver_u_upd.conv_reason))
                self._diverged = True

            self.iterations_ksp['u_upd'][i].append(
                self.solver_u_ten.iterations)
            self.residuals_ksp['u_upd'][i].append(self.solver_u_ten.residuals)
        timer.stop()
        with Timer('Z assign'):
            if not self._using_ale:
                self._merge_lst_to_vec(self.u_lst, self.u)
            else:
                self._merge_lst_to_vec(self.upd_lst, self.upd)
                # ALE: self.u must reflect the corrected (divergence-free)
                # velocity so ProblemCoupled's u_conv = 2*self.u - self.u0 and
                # JellyFSI traction computation use the physical u_{n+1}.
                # u_lst was already advanced to the corrected velocity above.
                self._merge_lst_to_vec(self.u_lst, self.u)

    def pressure_increment(self):
        ''' IPCS: increment pressure. '''
        if self.options['timemarching']['fractionalstep']['scheme'] == 'IPCS':
            self.p.x.petsc_vec.axpy(1., self.phi.x.petsc_vec)

    def update_displacement(self):
        ''' Performs displacement update. '''
        self.d0.x.array[:] = self.d.x.array

    def update_displacement_bcs(self):
        ''' Update time dependent boundary conditions. '''
        for bc, dict_ in self.bc_dict['d']['dbc_expressions'].items():
            expr = dict_['expression']
            if 't' in expr.user_parameters:
                expr.t = float(self.t)
                # self.project_enriched_dbc(bc, expr)

        if self.ale['type'] == 'external':
            self.logger.debug('Using external displacement for dbc')
            for bc, dict_ in self.bc_dict['d']['dbc_functions'].items():
                func = dict_['function']
                self.transfer_displacement(self.d_s, func) # in -> out

    def update_velocity_bcs(self):
        ''' Update time dependent boundary conditions. '''
        for bc, dict_ in self.bc_dict['u']['dbc_expressions'].items():
            if 'expression' in dict_:
                expr = dict_['expression']
            if 'inflow_func' in dict_:
                inflow_func = dict_['inflow_func']
                inflow_upd = inflow_func(self.t)
                expr.value = inflow_upd
            elif 'pinns_data' in dict_:
                dict_['uprofile'][0].x.array[:] = dict_['pinns_data']['ux'].item()[self.it-1][:,0]
                dict_['uprofile'][1].x.array[:] = dict_['pinns_data']['uy'].item()[self.it-1][:,0]
                dict_['uprofile'][2].x.array[:] = dict_['pinns_data']['uz'].item()[self.it-1][:,0]
            elif 'parable_funcs' in dict_:
                scale = dict_['parable_scale_func'](self.t)
                c, t1, t2 = dict_['centroid'], dict_['t1'], dict_['t2']
                n_hat = dict_['n']
                R1, R2, U = dict_['R1'], dict_['R2'], dict_['U']
                _eps = 1e-12
                for i, func_i in enumerate(dict_['parable_funcs']):
                    def _interp(x, _i=i, _c=c, _t1=t1, _t2=t2, _n=n_hat,
                                _R1=R1, _R2=R2, _U=U, _s=scale, _e=_eps):
                        pts = x.T - _c
                        profile = 1.0 - (pts @ _t1 / _R1) ** 2
                        if _R2 > _e:
                            profile -= (pts @ _t2 / _R2) ** 2
                        return _U * _s * np.clip(profile, 0.0, None) * _n[_i]
                    func_i.interpolate(_interp)
            else:
                if 't' in expr.user_parameters:
                    expr.t = float(self.t)
                    if bc != 'inflow':
                        self.project_enriched_dbc(bc, expr)
            
        if self.ale['type'] == 'external':
            self.logger.debug('Using external velocity for dbc')
            # XXX DBF1 ?
            with Timer('Z assign'):
                self.transfer_velocity(self.v_s, self.v_bc) # in -> out
                self._split_vec_to_lst(self.v_bc, self.v_lst_bc)
                for bc, dict_ in self.bc_dict['u']['dbc_functions'].items():
                    func, i = dict_['function'], dict_['i']
                    func.x.array[:] = self.v_lst_bc[i].x.array
                # FSI Robin-Neumann: the interface velocity enters as Robin
                # RHS data (v_rb) instead of a Dirichlet function
                if self.bc_dict['u'].get('fsi_robin'):
                    rb = self.bc_dict['u']['fsi_robin']
                    rb['v_rb'].x.array[:] = self.v_bc.x.array
                    rb['v_rb'].x.scatter_forward()

    # =========================================================================
    # Windkessel
    # =========================================================================

    def solve_windkessel(self, restart=False, flow = True):
        ''' Solve windkessel
        '''

        F, J = self.F, self.J
        ds = Measure('ds', domain=self.p.function_space.mesh,
                        subdomain_data=self.bnds)
        n = FacetNormal(self.p.function_space.mesh)

        for bid, prm in self.bc_dict['p']['windkessel']['params'].items():
            dt = float(self.options['timemarching']['dt'])
            R_d = float(prm['R_d'])
            R_p = float(prm['R_p'])
            C = float(prm['C'])

            alpha = R_d*C/(R_d*C + dt)
            beta = R_d*(1 - alpha)
            gamma = R_p + beta

            if flow:
                Q = self.p.function_space.mesh.comm.allreduce(
                    assemble_scalar(fem_form(dot(self.u, J*ufl.inv(F).T*n)*ds(bid))),
                    op=MPI.SUM)
                prm['Q'].value = Q

            pi = float(prm['pi'])
            pi_upd = pi

            if self.wk['explicit']:
                if not restart:
                    pi_upd = alpha*pi + beta*Q

                Pl_upd = R_p*Q + float(pi_upd)

                prm['pi'].value = float(pi_upd)
                prm['pi0'].value = pi
                prm['Pl'].value = Pl_upd
                # updating the Dirichlet boundary condition
                prms = self.bc_dict['p']['windkessel']['params']

                for _, opts in self.bc_dict['p']['windkessel']['dbc_params'].items():
                    expr, bid2 = opts['expr'], opts['bid']
                    if bid2 == bid:
                        expr.value = prms[bid]['Pl'].value
                        break

            elif self.wk['implicit']:
                if not flow and not restart:
                    elem = J*ufl.sqrt(inner(ufl.inv(F).T*n, ufl.inv(F).T*n))
                    comm = self.p.function_space.mesh.comm
                    area_new = comm.allreduce(
                        assemble_scalar(fem_form(elem*ds(bid))), op=MPI.SUM)
                    Pl = comm.allreduce(
                        assemble_scalar(fem_form(self.p*ds(bid))), op=MPI.SUM) / area_new
                    pi_upd = float((1 - beta/gamma)*alpha*pi + beta/gamma*Pl)

                    self.pi_functions[bid].append(pi)
                    prm['area'].value = area_new
                    prm['Pl'].value = Pl
                    prm['pi'].value = pi_upd
        

        if self.wk['implicit']:
            if restart:
                self.update_windkessel_LRC(restart=True)

    def update_windkessel_LRC(self, restart=False):
        ''' Update Windkessel low rank correction (LRC) diagonal scaling D
        in A + UDU'.

        Args:
            restart (bool): restart with internal parameters

        '''
        fac_l = []
        for bid, prm in self.bc_dict['p']['windkessel']['params'].items():
            dt = float(self.options['timemarching']['dt'])
            R_d = float(prm['R_d'])
            R_p = float(prm['R_p'])
            C = float(prm['C'])

            alpha = R_d*C/(R_d*C + dt)
            beta = R_d*(1 - alpha)
            gamma = R_p + beta

            if restart:
                if abs(float(prm['C'])) > 1e-14:
                    delta_l = 1/gamma + beta/gamma/(gamma - beta)
                    delta_r = 1/(gamma - beta)
                else:
                    delta_l = 1/gamma
                    delta_r = 0.0

            else:
                delta_l = 1/gamma
                delta_r = alpha/gamma

            # info(f'R_d: {R_d}, at id {bid}')
            # info(f'alpha: {alpha}')

            prm['delta_l'].value = delta_l
            prm['delta_r'].value = delta_r

            fac_l.append(delta_l)

        coef_array = np.array(fac_l)
        self.vec['p']['windkessel_lhs_lrc_diag'].setArray(coef_array)
        self.vec['p']['windkessel_lhs_lrc_diag'].assemble()

        # Change operator in pressure step
        self.solver_p.set_operator(
            self.mat['p']['windkessel_lhs_lrc'],
            self.mat['p']['laplacian'])

    def project_enriched_dbc(self, bc, expr):
        if utils.is_enriched(bc.function_space):
            V = bc.function_space
            projected = _project(expr, V)
            bc.value.x.array[:] = projected.x.array

        # for bc in self.bc_dict['u']['time_bcs']:
        #     bc['expression'].t = float(self.t)

        #     if utils.is_enriched(self.u_lst[0].function_space):
        #         Vi = self.u_lst[0].function_space
        #         bcs = self.bc_dict['u']['dirichlet'][bc['i']]
        #         bcs[bcs.index(bc['id'])] = DirichletBC(
        #             Vi, project(bc['expression'], Vi), self.bnds,
        #             bc['id']
        #         )

    def update_pressure_bcs(self):
        ''' Update pressure Dirichlet BCs before solving the PPE '''
        # ### Transpiration BCs
        bc_trans = self.bc_dict['p']['transpiration']
        # update Transpiration-Dirichlet BCs if given
        assert (len(bc_trans['dirichlet_functions']) ==
                len(self.mat['p']['u_norm_bound']))
        for p_fun, beta, solver_p_mass, mat_u in zip(
                bc_trans['dirichlet_functions'], bc_trans['coef'],
                self.solver_p_mass, self.mat['p']['u_norm_bound']):
            # FIXME need to solve here
            timer = Timer('Z pressure BC proj')
            rhs = _mat_vec(mat_u, self.u.x.petsc_vec)
            rhs.scale(float(beta))
            solver_p_mass.solve(p_fun.x.petsc_vec, rhs)
            timer.stop()
        # change Robin coefficient?

        if self._using_ale and self.ale['type'] == 'external':
            # required for FSI coupling
            self.transfer_velocity(self.v_s, self.v_bc)

    # =========================================================================
    # Monitoring and I/O
    # =========================================================================

    def monitor(self, it, t=0):
        ''' Solution monitor, output interface.

        Args:
            it (int):   iteration count
            t (float): current time (required for compatibility with ROUKF
        '''
        # self._writeout:
        # 0: not written, 1: wrote xdmf, +2: wrote checkpoint
        # => *: xdmf, **: checkpoint, ***: both
        wstr = ''
        if np.mod(self._writeout, 2):
            wstr = 'W*'
        if self._writeout >= 2:
            wstr += ' CP*'
        # TODO: extend flux_normalize to ALE
        if self.options['timemarching']['report'] == 2:
            n = FacetNormal(self.u.function_space.mesh)
            ds = Measure('ds', domain=self.p.function_space.mesh,
                    subdomain_data=self.bnds)
            comm = self.u.function_space.mesh.comm
            flux = comm.allreduce(
                assemble_scalar(fem_form(dot(self.u, n)*ds)), op=MPI.SUM)
            fs_opt = self.options['timemarching']['fractionalstep']
            if fs_opt.get('flux_report_normalize_boundary', False):
                denom = comm.allreduce(
                    assemble_scalar(fem_form(dot(self.u, n)*ds(
                        fs_opt['flux_report_normalize_boundary']))),
                    op=MPI.SUM)
                flux /= denom

            flux_str = 'sum of fluxes: {}'.format(flux)
        else:
            flux_str = ''

        if (self.options['timemarching']['report'] == 3
            and self._using_ale):
            comm = self.u.function_space.mesh.comm
            new = comm.allreduce(assemble_scalar(fem_form(self.J*dx)), op=MPI.SUM)
            old = comm.allreduce(assemble_scalar(fem_form(self.J0*dx)), op=MPI.SUM)
            self.logger.info('Volume change: {rate:.{width}f}'
                            .format(rate=new/old, width=6))

        if self.options['timemarching']['report'] and (wstr or flux_str):
            reporter = getattr(self, '_reporter', None)
            msg = 't = {t:.{width}f} \t{w}\t{f}'.format(
                t=self.t, w=wstr, f=flux_str, width=6)
            if reporter is not None and reporter._bar is not None:
                from tqdm import tqdm as _tqdm
                _tqdm.write(msg)
            else:
                self.logger.info(msg)

    def read_timestep(self, i):
        ''' HDF5 checkpoint read out

        Args:
            i   iteration count for checkpoint index
        '''
        tol = 1e-8
        readout = 0

        T = self.ale['timemarching']['T']

        if self.ale['io']['read_checkpoints']:
            dt_checkpt = self.ale['timemarching']['checkpoint_dt']
            if ((self.t > self._t_checkpt + dt_checkpt - tol)
                    or (self.t >= T - tol)):

                self.t_read = self.t
                readout += 2
                self.read_HDF5_checkpoint(i)

        self._readout = readout

    def read_HDF5_checkpoint(self, i):
        ''' Read HDF5 checkpoint of d from <read_path>/checkpoints folder.

        Args:
            i   iteration count
        '''
        if not self.ale['io']['read_checkpoints']:
            return

        path = (self.ale['io']['read_path']
                + '/checkpoint/{i}/'.format(i=i))

        comm = self.u.function_space.mesh.comm
        self.logger.info('Reading HDF5 data at iteration {}'.format(i))
        inout.read_HDF5_data(comm, path + '/u.h5', self.d_s, '/u')
        inout.read_HDF5_data(comm, path + '/v.h5', self.v_s, '/u')

    @staticmethod
    def _make_interp_data(V_to, V_from):
        ''' Build DOLFINx 0.10 non-matching mesh interpolation data. '''
        import numpy as np
        from dolfinx.fem import create_interpolation_data
        cells = np.arange(
            V_to.mesh.topology.index_map(V_to.mesh.topology.dim).size_local,
            dtype=np.int32)
        return cells, create_interpolation_data(V_to, V_from, cells, padding=1e-14)

    def transfer_displacement(self, d_in, d_out):
        ''' Transfer displacement between FE spaces via DOLFINx interpolation. '''
        self.logger.debug('Transfer d between FE spaces.')
        if not hasattr(self, '_interp_data_d'):
            self._interp_cells_d, self._interp_data_d = self._make_interp_data(
                d_out.function_space, d_in.function_space)
        d_out.interpolate_nonmatching(d_in, self._interp_cells_d, self._interp_data_d)
        d_out.x.scatter_forward()

    def transfer_velocity(self, u_in, u_out):
        ''' Transfer velocity between FE spaces via DOLFINx interpolation. '''
        self.logger.debug('Transfer u between FE spaces.')
        if not hasattr(self, '_interp_data_u'):
            self._interp_cells_u, self._interp_data_u = self._make_interp_data(
                u_out.function_space, u_in.function_space)
        u_out.interpolate_nonmatching(u_in, self._interp_cells_u, self._interp_data_u)
        u_out.x.scatter_forward()

    @property
    def write_velocity(self):
        ''' Which velocity field to save to disk: ``'tentative'`` (u*) or
        ``'update'`` (u).  Controlled by ``io.write_velocity`` in the input
        YAML; defaults to ``'update'``. '''
        return self.options['io'].get('write_velocity', 'update')

    def write_timestep(self, i, update=False):
        ''' Combined checkpoint and XDMF write out.

        Args:
            i       (int)  iteration count for checkpoint index
            update  (bool) specify which velocity to write (ALE)

        '''
        tol = 1e-8
        writeout = 0

        T = self.options['timemarching']['T']

        # TODO: ADAPT ROUKF IN THE SAME WAY

        if (self.options['io']['write_hdf5_timeseries']
                or self.options['io']['write_xdmf']):

            write_dt = self.options['timemarching']['write_dt']

            if (self.t > self._t_write + tol) and (
                    (self.t > self._t_write + write_dt - tol)
                    or (self.t >= T - tol)):
                # if (time for write) or (first) or (last) — guarded by
                # `self.t > self._t_write + tol` so that the "or last"
                # branch fires at most ONCE per physical time, even though
                # write_timestep() is called once per FSI sub-iteration
                # (would otherwise append a duplicate <Grid Time=T> entry
                # to the XDMF temporal collection for every sub-iteration
                # of the final step)
                self._t_write = self.t
                writeout = 1
                self.write_xdmf()

        if self.options['io']['write_checkpoints']:
            checkpt_dt = self.options['timemarching']['checkpoint_dt']

            if (self.t > self._t_checkpt + tol) and (
                    (self.t > self._t_checkpt + checkpt_dt - tol)
                    or (self.t >= T - tol)):
                # if (time for checkpoint) or (last) — same per-sub-iteration
                # guard as above

                self._t_checkpt = self.t
                writeout += 2
                self.write_checkpoint(i, update=update)

        self._writeout = writeout

    def write_xdmf(self, t=None):
        ''' Write solution to XDMF files. If file objects have not been
        created, initialize. This works for steady and unsteady solvers with
        timestepping output.

        Args:
            t       (optional) time of solution
        '''
        if not self.options['io']['write_xdmf']:
            return

        if not t:
            t = self.t

        comm = self.u.function_space.mesh.comm
        mesh = self.u.function_space.mesh
        write_path = self.options['io']['write_path']
        if (not hasattr(self, '_xdmf_u') or self._xdmf_u is None):
            import os
            os.makedirs(write_path, exist_ok=True)
            self._xdmf_u = XDMFFile(comm, write_path + '/u.xdmf', 'w')
            self._xdmf_u.write_mesh(mesh)
            self._xdmf_p = XDMFFile(comm, write_path + '/p.xdmf', 'w')
            self._xdmf_p.write_mesh(self.p.function_space.mesh)
            # ALE mesh displacement in its own file: ParaView's Warp By
            # Vector misbehaves when two fields share one XDMF block.
            if self._using_ale:
                self._xdmf_d = XDMFFile(comm, write_path + '/d_ale.xdmf', 'w')
                self._xdmf_d.write_mesh(mesh)

            # For velocity elements of degree > 1, ParaView requires P1
            # interpolation — XDMF stores data at geometry (P1) nodes only.
            vel_space = self.options['fem']['velocity_space'].lower().strip()
            if vel_space == 'p1':
                self._u_vis = None
            else:
                self._u_vis = Function(
                    functionspace(mesh, ('Lagrange', 1, (self.ndim,))),
                    name='u')

            # Same for pressure spaces that are not nodal P1.
            pres_space = self.options['fem']['pressure_space'].lower().strip()
            if pres_space == 'p1':
                self._p_vis = None
            else:
                self._p_vis = Function(
                    functionspace(mesh, ('Lagrange', 1)), name='p')

        if self._using_ale:
            self._xdmf_d.write_function(self.d, float(t))
            self._xdmf_u.write_function(self.upd, float(t))
        else:
            u_write = self.u
            if self._u_vis is not None:
                self._u_vis.interpolate(self.u)
                u_write = self._u_vis
            self._xdmf_u.write_function(u_write, float(t))

        p_write = self.p
        if self._p_vis is not None:
            self._p_vis.interpolate(self.p)
            p_write = self._p_vis
        self._xdmf_p.write_function(p_write, float(t))

    def write_timeseries(self):
        ''' Write solution to HDF5 TimeSeries, initialize on first call.  '''
        raise Exception('TimeSeries should not be used anymore!')

        if not self.options['io']['write_hdf5_timeseries']:
            return

        if (not hasattr(self, '_hdf5_ts_u') or not self._hdf5_ts_u or
                not hasattr(self, '_hdf5_ts_p') or not self._hdf5_ts_p):
            self.logger.info('Creating TimeSeries')
            self._hdf5_ts_u = TimeSeries(
                MPI.comm_world,
                self.options['io']['write_path'] + '/u_timeseries'
            )
            self._hdf5_ts_p = TimeSeries(
                MPI.comm_world,
                self.options['io']['write_path'] + '/p_timeseries'
            )

        self.logger.debug('Writing solution at time t = {t} to TimeSeries'
                          .format(t=self.t))
        self._hdf5_ts_u.store(self.u.x.petsc_vec, float(self.t))
        self._hdf5_ts_p.store(self.p.x.petsc_vec, float(self.t))

    def write_checkpoint(self, i, update=False):
        ''' Write checkpoint of u, p to <write_path>/checkpoint/<i>/w.h5.

        Both fields are stored together in a single HDF5 file as raw DOF
        arrays (/u, /p) plus a timestamp attribute, avoiding any split-order
        ambiguity that arises when reconstructing from a mixed-space function.
        ALE displacement is written separately to d.h5.

        Args:
            i       (int)  iteration count
            update  (bool) specifies which velocity to store (in ALE)
        '''
        if not self.options['io']['write_checkpoints']:
            return

        import h5py, os
        comm = self.u.function_space.mesh.comm
        path = self.options['io']['write_path'] + '/checkpoint/{i}/'.format(i=i)

        if comm.rank == 0:
            os.makedirs(path, exist_ok=True)
        comm.Barrier()

        u_out = (self.upd if (self._using_ale and update) else self.u)

        def _gather_full(fn):
            idx = fn.function_space.dofmap.index_map
            local_arr = fn.x.array[:idx.size_local].copy()
            gathered = comm.gather(local_arr, root=0)
            return np.concatenate(gathered) if comm.rank == 0 else None

        u_full = _gather_full(u_out)
        p_full = _gather_full(self.p)

        if comm.rank == 0:
            with h5py.File(path + 'w.h5', 'w') as f:
                f.create_dataset('u', data=u_full)
                f.create_dataset('p', data=p_full)
                f.attrs['t'] = float(self.t)

        if self._using_ale:
            d_full = _gather_full(self.d)
            if comm.rank == 0:
                with h5py.File(path + 'd.h5', 'w') as f:
                    f.create_dataset('d', data=d_full)
                    f.attrs['t'] = float(self.t)

        if self.options['timemarching']['fractionalstep']['scheme'] == 'IPCS':
            self.logger.warn('Checkpointing for IPCS experimental!')

    def write_statistics(self):
        ''' Write number of linear iterations (KSP) and timings to files '''
        #rank = MPI.COMM_WORLD.rank
        #fname = self.options['io']['write_path'] + '/stats.{}.dat'.format(rank)
        
        #with open(fname, 'wb') as fout:
        #    data = {
        #        'iterations_ksp': self.iterations_ksp,
        #        'residuals_ksp': self.residuals_ksp,
        #        'time': self.t_elapsed,
        #        'np': MPI.comm_world.size,
        #        'node': platform.uname()[1],
        #   }
        #    pickle.dump(data, fout, protocol=pickle.HIGHEST_PROTOCOL)
        #
        #fname = self.options['io']['write_path'] + '/timings.{}'.format(rank)
        
        #with open(fname, 'wt') as fout:
        #    # or: mode 'wb' and string.encode('utf-8')
        #    fout.write(timings(TimingClear.keep,
        #                       [TimingType.wall, ]).str(True))

        if self._using_wk:
            with open(self.options['io']['write_path'] + '/pi_functions.pickle', 'wb') as handle:
                pickle.dump(self.pi_functions, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def write_xdmf_obs(self, i, Xobs_i):
        path = self.options['io']['write_path'] + 'obs/Xobs_{}'.format(i) + '.xdmf'
        xdmf = XDMFFile(path)
        xdmf.write(Xobs_i)
        xdmf.close()

    def compute_innovation(self, Z_lst, Xobs_i, Zmod_lst=None):
        ''' Compute innovation for current iterates of observations Z
        (and current magnetizations moduli Zmod_lst) and sigma point solutions.

        The observation functions contained in Xobs_i are replaced by the
        corresponding innovation (subtracted from measurements).

        Args:
            Z_lst (list):   list of measurements
            Xobs_i (list):  list of observations for each sigma point X
                            measurement
            Zmod_lst (list):  list of magnetization moduli associated to
                            Xobs_i
        '''
        if not Zmod_lst:
            # iterate through sigma points
            for Xobs_lst in Xobs_i:
                # iterate through measurements
                for Z, X_i in zip(Z_lst, Xobs_lst):
                    if isinstance(X_i, fem.Function):
                        X_i.x.petsc_vec.axpy(-1., Z.x.petsc_vec)
                        X_i.x.array[:] *= -1
                        X_i.x.scatter_forward()
                    else:
                        # assumed to be numpy array
                        X_i -= Z
                        X_i *= -1
                # self.write_xdmf_innov_dbg(self.t, Xobs_i)
        else:
            # innovation for the magnetization case
            estimation_dict = self.options['estimation']
            measurement_lst = self.options['estimation']['measurements']
            if 'VENC' not in estimation_dict['roukf']['MAG_functional']:
                raise KeyError('VENC value not found')

            VENC_lst = [meas_lst['VENC'] for meas_lst in measurement_lst]

            # iterate through sigma points
            for Xobs_lst in Xobs_i:
                # iterate through measurements
                for Z, Zm, X_i, venc in zip(
                        Z_lst, Zmod_lst, Xobs_lst , VENC_lst):
                    if isinstance(Z, fem.Function):
                        xi_vector = X_i.x.array
                        zi_vector = Z.x.array
                        mod_vector = Zm.x.array
                        X_i.x.array[:] = (mod_vector/np.sqrt(2))*np.sin(np.pi/venc*(xi_vector - zi_vector))
                        X_i.x.scatter_forward()
                    else:
                        X_i = Zm/np.sqrt(2)*np.sin(np.pi/venc*(X_i - Z))

    def innovation_to_numpy(self, Z_lst, Xobs_i, sigma_Z , Zmod_lst=None):
        ''' Call compute_innovation and computes associated sensitivity
        into numpy arrays.

        Args:
            Z_lst (list):   list of measurements
            Xobs_i (list):  list of observations of particle states
            sigma_Z (list): list of measurement standard deviations
            Zmod_lst (list):  list of magnetizations

        Returns:
            numpy.ndarray:   numpy representation of innovation functions
        '''

        self.compute_innovation(Z_lst, Xobs_i , Zmod_lst)

        # Xobs_i is replaced by innovation
        # for explicitness:i
        # Innov_particles = Xobs_i
        if isinstance(Xobs_i[0][0], fem.Function):
            innov_numpy = np.array(
                [np.concatenate(
                    [1./sigma * iv.x.array for sigma, iv in
                    zip(sigma_Z, iv_lst)])
                for iv_lst in Xobs_i]
            ).T
        else:
            innov_numpy = np.array([np.concatenate([1./sigma * iv.flatten() for sigma, iv in zip(sigma_Z, iv_lst)])
                 for iv_lst in Xobs_i]).T

        return innov_numpy

    def read_measurement_indices_chkpt(self, measurement) -> list:
        ''' Read measurement indices from checkpoint files.

        Args:
            measurement (dict):     measurement dictionary

        Returns:
            list:   indices
        '''
        if not measurement:
            measurement = self.options['estimation']['measurements'][0]

        root = measurement['file_root']

        path_all = list(Path().glob(root.format(i='[0-9]*')))

        root_split = root.split('{i}')

        assert len(root_split) == 2, 'HDF5 file root must contain {i} once'

        chars_before = len(root_split[0])
        chars_after = len(root_split[1])

        indices = []

        # self.logger.warning(str(path_all))

        for path in path_all:
            #self.logger.warning(str(path))
            index = int(str(path)[chars_before: -chars_after])
            indices.append(index)

        indices = sorted(indices)

        return indices

    def get_measurement_indices(self, measurement_lst) -> list:
        ''' Get measurement indices from checkpoint files or user specified.

        Args:
            measurement_lst (list):    list of measurement dictionaries

        Returns:
            list:   measurement indices
        '''
        indices = self.read_measurement_indices_chkpt(measurement_lst[0])

        self.logger.debug(indices)

        # Done. Now check if indices are consistent

        # compare indices of all measurement data sets
        for measurement in measurement_lst[1:]:
            indices_ = self.read_measurement_indices_chkpt(measurement)

            if not indices_ == indices:
                raise Exception('Indices of measurements must match in '
                                'current implementation.')

        user_indices = self.options['estimation']['measurements'][0]['indices']

        for meas in self.options['estimation']['measurements']:
            if not meas['indices'] == user_indices:
                raise NotImplementedError('Indices/timesteps of all '
                                          'measurements must match.')

        # if indices are specified in options file, use these instead of
        # file_indices; filter file_times accordingly
        if user_indices:
            if not all(i in indices for i in user_indices):
                raise Exception('Given measurement indices don\'t match '
                                'indices found in measurements directory')

            indices = user_indices

        return indices

    def init_cache_measurements(self, Xobs_lst) -> tuple:
        ''' Initialize measurements. Take function space from observations to
        reduce memory footprint.
        Creates a cache dictionary, containing references to the
        measurement functions, indices, time stamps.

        Args:
            Xobs_lst (list of Function):    list of observations

        Returns:
            tuple:  tuple containing:

                * Z_lst (list):     list of 'reusable' measurement functions
                * sigma_Z (list):   list of standard deviation per measurement
                                    set.
        '''
        measurement_lst = self.options['estimation']['measurements']

        indices = self.get_measurement_indices(measurement_lst)

        # load first measurements
        u_next_lst = []
        u_prev_lst = []
        Z_lst = []

        hdf5_root_lst = [meas['file_root'] for meas in measurement_lst]

        for hdf5_root, Xobs in zip(hdf5_root_lst, Xobs_lst):
            if isinstance(Xobs, fem.Function):
                V = Xobs.function_space
                u_next_lst.append(Function(V, name='u_next'))
                u_prev_lst.append(Function(V, name='u_prev'))
                Z_lst.append(Function(V, name='innovation'))

                comm = V.mesh.comm
                t_prev = inout.read_HDF5_data(comm, hdf5_root.format(i=indices[0]),
                                              u_prev_lst[-1], '/u')
                # self.logger.debug('t_prev: ' + str(t_prev))

                t_next = inout.read_HDF5_data(comm, hdf5_root.format(i=indices[1]),
                                              u_next_lst[-1], '/u')
                # self.logger.debug('t_next: ' + str(t_next))
            else:
                #hdf5_root assumed to be npy format
                u_prev_lst.append(np.load(hdf5_root.format(i = indices[0])))
                u_next_lst.append(np.load(hdf5_root.format(i = indices[1])))

                Z_lst.append(np.zeros(Xobs.shape))

                #infer timesteps from indices, since no time data is available
                t_prev = float(indices[0]/1000)
                t_next = float(indices[1]/1000)

        cache_dict = {
            'u_next_lst': u_next_lst,
            'u_prev_lst': u_prev_lst,
            'index': 0,
            't_prev': t_prev,
            't_next': t_next,
            'file_indices': indices,
            }

        return cache_dict, Z_lst

    def init_cache_magnetization(self, Xobs_lst) -> tuple:
        ''' Initialize magnetization measurements.
        Creates a cache dictionary, containing references to the
        time stamps.

        Args:
            Xobs_lst (list of Function):    list of observations

        Returns:
            tuple:  tuple containing:

                * magnetization_cache_dict (dict):  dict with time stamps
                * sigma_Z (list):   list of standard deviation per measurement
                                    set.
        '''
        if not hasattr(self, '_using_magnetization'):
            mag_flag = self.options['estimation']['roukf'].get(
                    'MAG_functional', False)
            self._using_magnetization = mag_flag
            self.logger.info('Initializing measurements with '
                             'magnetization option set {}'
                             .format(self._using_magnetization))

        if not self._using_magnetization:
            return dict(), None

        measurement_lst = self.options['estimation']['measurements']

        indices = self.get_measurement_indices(measurement_lst)

        mod_next_lst = []
        mod_prev_lst = []
        Zmod_lst = []

        hdf5_root_lst_mod = [meas['module_meas_file_root'] for meas in measurement_lst]

        # initializing moduli measurements
        for hdf5_root, Xobs in zip(hdf5_root_lst_mod, Xobs_lst):
            V = Xobs.function_space
            mod_next_lst.append(Function(V, name='mod_next'))
            mod_prev_lst.append(Function(V, name='mod_prev'))
            Zmod_lst.append(Function(V, name='module'))
            comm = V.mesh.comm
            #reading the data but dropping the time 
            _ = inout.read_HDF5_data(comm, hdf5_root.format(i=indices[0]),
                                      mod_prev_lst[-1], '/M')

            _ = inout.read_HDF5_data(comm, hdf5_root.format(i=indices[1]),
                                      mod_next_lst[-1], '/M')

        magnetization_cache_dict = {
            'mod_next_lst': mod_next_lst,
            'mod_prev_lst': mod_prev_lst,
        }

        return magnetization_cache_dict, Zmod_lst

    def init_measurements(self, Xobs_lst) -> tuple:
        ''' Initialize measurements. Take function space from observations to
        reduce memory footprint.
        Creates a self.cache dictionary, calling init_cache_measurements
        and init_cache_magnetization.

        Args:
            Xobs_lst (list of Function):    list of observations

        Returns:
            tuple:  tuple containing:

                * Z_lst (list):     list of 'reusable' measurement functions.
                * sigma_Z (list):   list of standard deviation per measurement
                                    set.
                * Z_lst_mod (list): list of measurement magnetization moduli
                                    functions (default is None).
        '''
        if not hasattr(self, 'cache'):
            self.cache = {'measurements': {}}

        cache_meas, Z_lst = self.init_cache_measurements(Xobs_lst)
        cache_magn, Zmod_lst = self.init_cache_magnetization(Xobs_lst)

        self.cache['measurements'].update(cache_meas)
        self.cache['measurements'].update(cache_magn)

        # Update cache dictionaries in ROUKF
        sigma_Z = [meas['noise_stddev'] for meas in
                    self.options['estimation']['measurements']]

        for k, meas in enumerate(self.options['estimation']['measurements']):

            if sigma_Z[k] == 'initial':
                # Computing sigma from initial measurements
                u_prev_lst = self.cache['measurements']['u_prev_lst']
                if isinstance(u_prev_lst[k], fem.Function):
                    uvec = u_prev_lst[k].x.array
                else:
                    uvec = u_prev_lst[k]

                if not self._using_magnetization:
                    # Classic functional
                    sigma_Z[k] = np.round(np.std(uvec),4)

                else:
                    # Magnetization functional
                    VENC = meas['VENC']
                    mod_prev_lst = self.cache['measurements']['mod_prev_lst']
                    if isinstance(mod_prev_lst[k], fem.Function):
                        Mvec = mod_prev_lst[k].x.array
                    else:
                        Mvec = mod_prev_lst[k]
                    ss = Mvec**2*(1-np.cos(np.pi/VENC*uvec))
                    w_m = 1/len(uvec)*np.sum(ss)
                    sigma_Z[k] = np.round(np.sqrt(w_m),4)

                self.logger.info('setting sigma meas from initial timestep as {}'
                                .format(sigma_Z[k]))

        return Z_lst, sigma_Z, Zmod_lst

    def read_measurements(self, t, Z_lst, Zmod_lst = None):
        ''' Read and interpolate measurements at time t.

        Args:
            t (float):                   current time
            Z_lst (list of Function):    receiving list of measurement functions
            Zmod_lst (list of Function): optional list of magnetization moduli
                                        measurements
        '''
        tol = 1e-12

        file_indices = self.cache['measurements']['file_indices']
        idx = self.cache['measurements']['index']
        u_next_lst = self.cache['measurements']['u_next_lst']
        u_prev_lst = self.cache['measurements']['u_prev_lst']
        t_prev = self.cache['measurements']['t_prev']
        t_next = self.cache['measurements']['t_next']

        interpolation_done = False

        measurement_lst = self.options['estimation']['measurements']

        hdf5_root_lst = [meas['file_root'] for meas in measurement_lst]

        if self._using_magnetization:
            mod_next_lst = self.cache['measurements']['mod_next_lst']
            mod_prev_lst = self.cache['measurements']['mod_prev_lst']
            hdf5_mod_root_lst = [Mmeas['module_meas_file_root'] for Mmeas in measurement_lst]


        self.logger.debug('read_measurement: interval={}, t={}, t_p={}, t_n={}'
                          .format(idx, t, t_prev, t_next))

        # if t > t_next, increment interval index
        if t >= t_next - tol:
            if idx < len(file_indices) - 2:
                idx += 1
                self.logger.debug('read_measurement: increase interval idx,'
                                  ' read file {}'.format(
                                      file_indices[idx + 1]))
                t_prev = t_next

                for i, (u_prev, u_next, froot) in enumerate(zip(u_prev_lst, u_next_lst,
                                                 hdf5_root_lst)):
                    if isinstance(u_prev, fem.Function):
                        u_prev.x.array[:] = u_next.x.array

                        mpi_comm = u_next.function_space.mesh.comm
                        t_next = inout.read_HDF5_data(
                            mpi_comm, froot.format(i=file_indices[idx + 1]),
                            u_next, '/u')
                    else:
                        u_prev *= 0
                        u_prev += u_next
                        t_next = file_indices[idx+1]/1000
                        u_next_lst[i] = np.load(froot.format(i=file_indices[idx+1])) 

                    assert t_prev - tol <= t < t_next - tol

                self.cache['measurements']['index'] = idx
                self.cache['measurements']['t_prev'] = t_prev
                self.cache['measurements']['t_next'] = t_next

                if self._using_magnetization:
                    for i, (mod_prev, mod_next, froot) in enumerate(zip(mod_prev_lst, mod_next_lst, hdf5_mod_root_lst)):
                        if isinstance(mod_prev, fem.Function):
                            mod_prev.x.array[:] = mod_next.x.array
                            mpi_comm = mod_next.function_space.mesh.comm
                            _ = inout.read_HDF5_data(mpi_comm, froot.format(i=file_indices[idx + 1]),mod_next, '/M')
                        else:
                            mod_prev *= 0
                            mod_prev += mod_next
                            mod_next_lst[i] = np.load(froot.format(i=file_indices[idx+1]))


                self.logger.debug(' ===> {}, {}, {}, {}'.format(idx, t,
                                                                t_prev,
                                                                t_next))

            else:
                # reached last measurement, continue using last measurement!
                if isinstance(u_next_lst[0], fem.Function):
                    [Z.x.array.__setitem__(slice(None), u_next.x.array) for Z, u_next in zip(Z_lst, u_next_lst)]
                else:
                    Z_lst = u_next_lst
                self.logger.debug('read_measurement: last file, i={}, t={}'.
                                  format(idx, t_next))

                if Zmod_lst:
                    if isinstance(mod_next_lst[0], fem.Function):
                        [Zm.x.array.__setitem__(slice(None), mod_next.x.array) for Zm, mod_next in zip(Zmod_lst, mod_next_lst)]
                    else:
                        Zmod_lst = mod_next_lst

                interpolation_done = True
                if not np.allclose(t, t_next):
                    self.logger.debug('No measurements for t > {}! const. '
                                      'extrapolation'.format(t_next))

        # current time is between 2 loaded measurements
        if t_prev - tol <= t < t_next - tol:
            # compute weights
            c1 = (t - t_prev)/(t_next - t_prev)
            c0 = 1 - c1
            # self.logger.debug('{}: read_measurement  |u_prev| = {}'
            #                     .format(rank, norm(u_prev)))
            # self.logger.debug('{}: read_measurement  |u_next| = {}'
            #                     .format(rank, norm(u_next)))
            #if c1 > 0.:
            if c1 > tol:
                self.logger.debug('read_measurement: Interpolating! '
                                  'weights = ({}, {})'.format(c0, c1))
                for Z, u_prev, u_next in zip(Z_lst, u_prev_lst, u_next_lst):
                    if isinstance(Z, fem.Function):
                        Z.x.array[:] = c0*u_prev.x.array + c1*u_next.x.array
                        Z.x.scatter_forward()
                    else:
                        Z *= 0
                        Z += c0*u_prev + c1*u_next
                if Zmod_lst:
                    for Zm, mod_prev, mod_next in zip(Zmod_lst, mod_prev_lst, mod_next_lst):
                        if isinstance(Zm, fem.Function):
                            Zm.x.array[:] = c0*mod_prev.x.array + c1*mod_next.x.array
                            Zm.x.scatter_forward()
                        else:
                            Zm *= 0
                            Zm += c0*Z + c1*mod_next

            else:
                self.logger.debug('read_measurement: NO interpolation! '
                                  't = t_p')
                for Z, u_prev in zip(Z_lst, u_prev_lst):
                    if isinstance(Z, fem.Function):
                        Z.x.array[:] = u_prev.x.array
                    else:
                        Z *= 0
                        Z += u_prev

                if Zmod_lst:
                    for Zm, mod_prev in zip(Zmod_lst, mod_prev_lst):
                        if isinstance(Zm, fem.Function):
                            Zm.x.array[:] = mod_prev.x.array
                        else:
                            Zm *= 0
                            Zm += mod_prev

            interpolation_done = True

        elif t < t_prev - tol:
            # this should only happen at the start, if no measurements for the
            # first timestep(s) exist
            # u_next is first measurement (by init_measurements), u_prev = 0
            c0 = t/t_prev
            self.logger.debug('read_measurement: Interpolating t < t_prev: '
                              'weight = {}'.format(c0))
            for Z, u_prev in zip(Z_lst, u_prev_lst):
                if isinstance(Z, fem.Function):
                    Z.x.array[:] = c0*u_prev.x.array
                    Z.x.scatter_forward()
                else:
                    Z *= 0
                    Z += c0*u_prev

            if Zmod_lst:
                for Zm, mod_prev in zip(Zmod_lst, mod_prev_lst):
                    if isinstance(Zm, fem.Function):
                        Zm.x.array[:] = c0*mod_prev.x.array
                        Zm.x.scatter_forward()
                    else:
                        Zm *= 0
                        Zm += c0*mod_prev
            interpolation_done = True

        if not interpolation_done:
            raise Exception('read_measurement terminated without '
                            'interpolating')

    def get_state_functionspaces(self) -> None:
        ''' Return function spaces which are related to state variables of the
        FS problem. These are:
            - monolithic (for reference): X = u_n [u_n-1,..., for higher order
                time schemes]
            - CT standard:  X = u_n tentative
            - IPCS BDF:  X = [u_n, u_n-1, p_n] ?

        If windkessel boundaries are included, they belong to the state.

        Returns:
            list of function spaces
        '''
        if not self.options['timemarching']['fractionalstep']['scheme'] == 'CT':
            raise NotImplementedError(
                'get_state_functionspaces() not supported for {}'.format(
                    self.options['timemarching']['fractionalstep']['scheme']))

        W_lst = [self.u.function_space]

        for bc in self.options['boundary_conditions']:
            type_ = bc.get('type', None)
            prms = bc.get('parameters', dict())
            C_ = prms['C'] if 'C' in prms else None
            if type_ == 'windkessel' and C_:
                # DOLFINx: use DG0 as a surrogate for the legacy 'R' (real) space
                # for the single-DOF windkessel state variable
                R = functionspace(self.u.function_space.mesh, ("DG", 0))
                W_lst.append(R)

        self.logger.warning(
            'State function spaces: {}'.format(
                [W.ufl_element().family() for W in W_lst]
            )
        )

        return W_lst

    def close_xdmf(self) -> None:
        ''' close XDMF Files '''
        if hasattr(self, '_dont_close_xdmf') and self._dont_close_xdmf:
            # a little hacky, I admit
            return

        for attr in ('_xdmf_d', '_xdmf_u', '_xdmf_p', '_xdmf_du',
                     '_u_vis', '_p_vis'):
            if hasattr(self, attr):
                delattr(self, attr)


# TODO: Extend coupled solver for the full ALE case!
class SolverCoupled(Solver):
    def __init__(self, problem, dump_parameters=True):
        self.init_logging()
        self._logging_filehandler = problem._logging_filehandler
        if self._logging_filehandler:
            self.logger.addHandler(self._logging_filehandler)

        self.logger.info('Initializing')
        self.options = problem.options
        self.inputfile = problem.inputfile
        self.t = 0.
        self._t_write = 0.
        self._t_checkpt = 0.
        self.ndim = problem.ndim
        self.bc_dict = problem.bc_dict
        self.u_conv_assigned = problem.u_conv_assigned
        self.p = problem.p
        self.u = problem.u
        self.u0 = problem.u0
        self.u0_mapdd = problem.u0_mapdd
        

        self.ale = problem.ale
        self._using_ale = problem._using_ale
        self._using_wk = problem._using_wk
        self.wk = problem.wk

        self.du = Function(problem.V, name='du')

        self.windkessel_flag = bool(self.bc_dict['p']['windkessel']['id'])
        self.windkessel_flag_C = bool(self.bc_dict['p']['windkessel']['C'])

        self._using_mapdd = problem._using_mapdd
        self._applying_dc_on_update = False

        if 'DC_in_update' in self.options['fem'] and self.options['fem']['DC_in_update']:
            self._applying_dc_on_update = True

        if self.options['timemarching']['fractionalstep']['scheme'] == 'IPCS':
            self.phi = Function(problem.Q, name='phi')
        else:
            self.phi = self.p

        if 'state_velocity' in self.options['fluid']:
            self.state_velocity = self.options['fluid']['state_velocity']
        else:
            # tentative velocity by default
            self.state_velocity = 'tentative'

        self.forms = problem.forms

        self._diverged = False
        # a transpiration/slip optimizer should set this to True
        self._optimizing = False
        self._optimize_robin = False
        self._initialized = False

        self.bnds = problem.bnds

        self.mat = {}
        self.vec = {}

        if self._using_ale:
            self.d = problem.d
            self.d0 = problem.d0

            self.disp_bc = problem.disp_bc
            self.vel_bc = problem.vel_bc
            self.vel_bc_lst = problem.vel_bc_lst

            # FIXME: Used for testing if ALE operators are updated over time
            # self.F, self.F0 = problem.F, problem.F0
            # self.J, self.J0 = problem.J, problem.J0
            # self.j = Function(problem.DG, name='J')
            # self.j0 = Function(problem.DG, name='J0')

            if self.ale['type'] == 'external':
                self.S = problem.S
                self.d_s = problem.d_s
                self.v_s = problem.v_s

            self.mat.update({
                'd': {
                    'diff': None,
                    'div': None}
                })
            self.vec.update({
                'd': {'rhs_const': None}
                })

        self.mat.update({
                    'u': {'mass': None, 'conv': None, 'rhs': None,
                          'pdiv': None, 'gradp': None,
                          'lhs_navslip': [],
                          # 'rhs_navslip': {i: [] for i in range(self.ndim)},
                          'lhs_trans': [],
                          # 'rhs_trans': {i: [] for i in range(self.ndim)},
                          'p_trans': [],
                          'inflow_lhs': None,
                          },
                    # mass_robin: mass matrices on boundary patches with factor
                    # rho/dt
                    # mass_bound: unscaled mass matrices on boundary patches
                    'p': {'rhs_u': None, 'laplacian': None, 'mass_robin': None,
                          'mass_bound': [], 'u_norm_bound': []}
        })
        self.vec.update({
                    'u': {'rhs_const': None, 'inflow_rhs': None, 'mapdd_rhs': None},
                    'p': {'rhs_const': None}
        })

        if dump_parameters:
            self.dump_parameters()

        problem.close_logs()

    def init_assembly(self):
        ''' Initialize, assemble static matrices. '''
        # matrices of velocity component space Vi
        timer = Timer('Z init assembly')
        
        if self._using_ale:
            self.mat['d']['diff'] = _assemble_mat(self.forms['d']['diff'])
            self.mat['d']['div'] = _assemble_mat(self.forms['d']['div'])
            self.vec['d']['rhs_const'] = _assemble_vec(self.forms['d']['rhs_const'])

            for label in ['diff', 'div']:
                _apply_dbc_to_mat(self.mat['d'][label],
                                  self.bc_dict['d']['dirichlet'])

        self.mat['u']['mass'] = _assemble_mat(self.forms['u']['mass'])
        self.mat['u']['diff'] = _assemble_mat(self.forms['u']['diff'])
        # self.mat['u']['mass_diff'] = assemble(self.forms['u']['mass'] +
        #                                    self.forms['u']['diff'])
        self.mat['u']['rhs'] = self.mat['u']['mass'].copy()
        # if ('supg_gradp' in self.forms['u'] and
        #         self.forms['u']['supg_gradp']):
        #     self.mat['u']['supg_gp'] = self.mat['u']['mass'].copy()

        if self.forms['u']['pres']:
            self.mat['u']['pdiv'] = _assemble_mat(self.forms['u']['pres'])
        if self.forms['u']['gradp']:
            self.mat['u']['gradp'] = _assemble_mat(self.forms['u']['gradp'])

        # init convection matrix with its own sparsity (NOT mass.copy())
        self.mat['u']['conv'] = create_matrix(fem_form(self.forms['u']['conv']))

        # assembling inflow matrices
        if self.forms['u']['inflow_lhs']:
            self.mat['u']['inflow_lhs'] = self.mat['u']['mass'].copy()
            _assemble_mat(self.forms['u']['inflow_lhs'],
                          mat=self.mat['u']['inflow_lhs'])

        if self.forms['u']['neumann']:
            self.vec['u']['rhs_const'] = _assemble_vec(self.forms['u']['neumann'])
        else:
            self.vec['u']['rhs_const'] = None

        # inflow RHS
        if self.forms['u']['inflow_rhs']:
            self.vec['u']['inflow_rhs'] = _assemble_vec(self.forms['u']['inflow_rhs'])
        # mapdd RHS
        if self.forms['u']['mapdd_rhs']:
            self.vec['u']['mapdd_rhs'] = _assemble_vec(self.forms['u']['mapdd_rhs'])

        # matrices of pressure space Q
        # right hand side matrix to be multiplied by u.vector()
        # div(u) and in case of transpiration BCs: dot(u, n) term
        self.mat['p']['rhs_u'] = _assemble_mat(self.forms['p']['rhs_u'])
        self.mat['p']['laplacian'] = _assemble_mat(self.forms['p']['laplacian'])

        # apply boundary conditions to pressure Laplacian
        _apply_dbc_to_mat(self.mat['p']['laplacian'],
                          self.bc_dict['p']['dirichlet'])

        if self.forms['p']['neumann']:
            self.vec['p']['rhs_const'] = _assemble_vec(self.forms['p']['neumann'])
        else:
            self.vec['p']['rhs_const'] = None

        # Transpiration and Navier-Slip matrices
        self.init_assembly_robin()

        del timer
        self._init_assembly_done = True

    def init_assembly_robin(self):
        ''' Initialize and assemble matrices related to Robin BCs
        (Navierslip/Transpiration) '''

        sparse_pat = self.mat['u']['mass']

        tmp_form = 0
        for i_bnd in range(len(self.forms['u']['navierslip']['coef'])):
            navslip_forms = self.forms['u']['navierslip']
            if not self._optimizing:
                tmp_form += (navslip_forms['coef'][i_bnd] *
                             navslip_forms['forms'][i_bnd]['implicit'])
            else:
                self.mat['u']['lhs_navslip'].append(
                    _assemble_mat(navslip_forms['forms'][i_bnd]['implicit'],
                                  mat=sparse_pat.copy()))

        # this is bit cumbersome, but close the the uncoupled version
        if not self._optimizing and tmp_form:
            self.mat['u']['lhs_navslip'].append(
                _assemble_mat(tmp_form, mat=sparse_pat.copy()))

        tmp_form_u = 0
        tmp_form_p = 0
        for i_bnd in range(len(self.forms['u']['transpiration']['coef'])):
            trans_forms = self.forms['u']['transpiration']
            if not self._optimizing:
                if trans_forms['forms'][i_bnd]['implicit']:
                    tmp_form_u += (trans_forms['coef'][i_bnd] *
                                   trans_forms['forms'][i_bnd]['implicit'])
            else:
                self.mat['u']['lhs_trans'].append(
                    _assemble_mat(trans_forms['forms'][i_bnd]['implicit'],
                                  mat=sparse_pat.copy()))

            # in the case of CT, add (pn, vn)*ds(i) to boundary form
            if (self.options['timemarching']['fractionalstep']['scheme'] ==
                    'CT'):
                tmp_form_p += trans_forms['forms'][i_bnd]['pressure']

        if not self._optimizing and tmp_form_u:
            self.mat['u']['lhs_trans'].append(
                _assemble_mat(tmp_form_u, mat=sparse_pat.copy()))
        if tmp_form_p:
            self.mat['u']['p_trans'] = _assemble_mat(tmp_form_p)

        # PRESSURE BC
        self.mat['p']['mass_robin'] = [
            _assemble_mat(a, mat=self.mat['p']['laplacian'].copy()) for a in
            self.forms['p']['robin']]

        self.mat['p']['u_norm_bound'] = [
            _assemble_mat(a) for a in
            self.forms['p']['transpiration_dirichlet_u']]
        for a in self.forms['p']['transpiration_dirichlet_p']:
            Atmp = _assemble_mat(a)
            # zero out near-zero diagonal entries so the matrix is invertible
            Atmp.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
            diag = Atmp.getDiagonal()
            diag_arr = diag.getArray()
            diag_arr[diag_arr == 0] = 1.0
            Atmp.setDiagonal(diag)
            self.mat['p']['mass_bound'].append(Atmp)

        # set corresponding rows to zero in Robin mass matrices (if any)
        for mat in self.mat['p']['mass_robin']:
            _zero_mat_rows(mat, self.bc_dict['p']['dirichlet'])

        if not self._optimizing:
            for mat, coef in zip(self.mat['p']['mass_robin'],
                                 self.bc_dict['p']['transpiration']['coef']):
                self.mat['p']['laplacian'].axpy(1./float(coef), mat)
        else:
            # raise Exception('Robin BC optimization not implemented for PPE')
            pass

    def assign_state(self, state):
        ''' ROUKF interface: Update instance solution functions from state
        variable (inverse to update_state).

        Args:
            state       list of state variables
        '''
        if not state:
            return

        self.u.x.array[:] = state[0].x.array

        if (self.options['timemarching']['fractionalstep']['scheme'] ==
                'IPCS'):
            assert True, 'experimental placeholder'
            # should work but not tested
            self.u0.x.array[:] = state[1].x.array
            self.p.x.array[:] = state[2].x.array

    def update_state(self, state):
        ''' ROUKF interface: update state variables from solution functions
        (inverse to assign_state).

        Args:
            state       list of state variables
        '''
        if not state:
            return

        state[0].x.array[:] = self.u.x.array

        if (self.options['timemarching']['fractionalstep']['scheme'] ==
                'IPCS'):
            assert True, 'experimental placeholder'
            assert len(state) == 3, 'expected state space dimension = 3'
            # should work but not tested
            state[1].x.array[:] = self.u0.x.array
            state[2].x.array[:] = self.p.x.array

    def init_solvers(self):
        ''' Initialize linear system solvers '''
        if not (hasattr(self, '_init_assembly_done') and
                self._init_assembly_done):
            raise Exception('init_assembly() must be called before '
                            'init_solvers()')

        self.iterations_ksp = {}
        self.residuals_ksp = {}
        if self._using_ale:
            self.solver_d = PETScSolver(self.options, 'd_',
                                        self._logging_filehandler,
                                        verbose=True)
            A = self.mat['d']['diff']
            A.axpy(1., self.mat['d']['div'])
            self.solver_d.set_operator(A)

            self.iterations_ksp.update({'d': []})
            self.residuals_ksp.update({'d': []})


        self.solver_u_ten = PETScSolver(self.options, 'u_ten_',
                                        self._logging_filehandler,
                                        verbose=True)

        self.solver_p = PETScSolver(self.options, 'p_',
                                    self._logging_filehandler, verbose=True)
        self.solver_p.set_operator(self.mat['p']['laplacian'])

        self.solver_p_mass = []
        for M in self.mat['p']['mass_bound']:
            self.solver_p_mass.append(PETScSolver(self.options, 'p_mass_',
                                                  self._logging_filehandler,
                                                  verbose=True))
            self.solver_p_mass[-1].set_operator(M)

        self.solver_u_upd = PETScSolver(self.options, 'u_upd_',
                                        self._logging_filehandler,
                                        verbose=True)
        self.solver_u_upd.set_operator(self.mat['u']['mass'])

        self.iterations_ksp.update({
                                'u_ten': [],
                                'u_upd': [],
                                'p': []
        })
        self.residuals_ksp.update({
                                'u_ten': [],
                                'u_upd': [],
                                'p': []
        })

        # TODO remove nullspace if no dirichlet BC!

    def assemble_tentative_velocity(self):
        ''' Assemble changing matrices for tentative velocity solve. '''
        # Note: Matrices are stored into mat['u']['conv'] and mat['u']['rhs']
        # A: system matrix. Assemble convection and add mass/diffusion
        if self.options['timemarching']['fractionalstep']['scheme'] == 'IPCS':
            sd_opt = (self.options['fem']['stabilization']
                      ['streamline_diffusion'])
            if (sd_opt['parameter'] in ('standard', 'default', 'klr') or
                    (sd_opt['parameter'] == 'shakib' and
                     'parameter_element_constant' in sd_opt and
                     sd_opt['parameter_element_constant'])):
                with Timer('Z assign conv'):
                    self.u_conv_assigned.x.array[:] = 2*self.u.x.array - self.u0.x.array

        # In ALE mode: recreate conv matrix each step (pre-allocated ghost set
        # becomes invalid after mesh deformation → PETSc "Argument out of range").
        if self._using_ale:
            self.mat['u']['conv'] = _assemble_mat(self.forms['u']['conv'])
        else:
            _assemble_mat(self.forms['u']['conv'], mat=self.mat['u']['conv'])
        A = self.mat['u']['conv']

        if (self.options['timemarching']['fractionalstep']['scheme'] == 'IPCS'
                and self.t > self.options['timemarching']['dt'] + 1e-14):
            cf = 1.5
        else:
            cf = 1

        if self._using_ale:
            _assemble_mat(self.forms['u']['mass'], mat=self.mat['u']['mass'])
            _assemble_mat(self.forms['u']['diff'], mat=self.mat['u']['diff'])
            # mat['u']['rhs'] changes over time when using ALE
            self.mat['u']['rhs'] = self.mat['u']['mass'].copy()

        A.axpy(cf, self.mat['u']['mass'], structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
        A.axpy(1., self.mat['u']['diff'], structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
        # adding inflow lhs term if defined
        if self.mat['u']['inflow_lhs']:
            A.axpy(1., self.mat['u']['inflow_lhs'])

        # assemble time/RHS SUPG matrices if present, add mass matrix
        # mat['u']['rhs'] was initialized to 'mass' and won't be changed
        # otherwise
        if 'supg_time' in self.forms['u'] and self.forms['u']['supg_time']:
            assert True, 'supg_time should not be useable!!'
            _assemble_mat(self.forms['u']['supg_time'],
                          mat=self.mat['u']['rhs'])
            A.axpy(1., self.mat['u']['rhs'])
            self.mat['u']['rhs'].axpy(1., self.mat['u']['mass'])

        _apply_dbc_rows_to_mat(A, self.bc_dict['u']['dirichlet'])

        return A

    def build_rhs_tentative_velocity(self):
        ''' Build RHS vector for tentative velocity solve, for the i'th
        component. '''
        if (self.options['timemarching']['fractionalstep']['scheme'] ==
                'IPCS'):
            # convection matrix was assembled already, so we can safely
            # overwrite u0 (time step k-1) for the RHS
            x = self.u.x.petsc_vec.copy()
            x.scale(2.0)
            x.axpy(-0.5, self.u0.x.petsc_vec)
        else:
            # CT
            if self._using_mapdd:
                x = self.u0.x.petsc_vec
            else:
                x = self.u.x.petsc_vec

        bu = _mat_vec(self.mat['u']['rhs'], x)

        if self.mat['u']['pdiv']:
            bu.axpy(-1.0, _mat_vec(self.mat['u']['pdiv'], self.p.x.petsc_vec))

        if self.vec['u']['rhs_const']:
            bu.axpy(1.0, self.vec['u']['rhs_const'])

        if self.vec['u']['inflow_rhs']:
            self.vec['u']['inflow_rhs'] = _assemble_vec(self.forms['u']['inflow_rhs'])
            bu.axpy(1.0, self.vec['u']['inflow_rhs'])

        if self.vec['u']['mapdd_rhs']:
            self.vec['u']['mapdd_rhs'] = _assemble_vec(self.forms['u']['mapdd_rhs'])
            bu.axpy(1.0, self.vec['u']['mapdd_rhs'])


        return bu

    def assemble_robin_tentative_velocity(self, A, bu):
        ''' Add robin terms to A and bu_i '''
        if not (self.forms['u']['navierslip']['coef'] or
                self.forms['u']['transpiration']['coef']):
            return None

        A_robin = A

        if not self._optimizing:
            assert len(self.mat['u']['lhs_navslip'])
            A_robin.axpy(1., self.mat['u']['lhs_navslip'][0])
            assert len(self.mat['u']['lhs_trans'])
            A_robin.axpy(1., self.mat['u']['lhs_trans'][0])

        else:
            for i_bnd in range(len(self.forms['u']['navierslip']['coef'])):
                coef_ns = float(self.forms['u']['navierslip']['coef']
                                [i_bnd])
                lhs_ns = self.mat['u']['lhs_navslip'][i_bnd]
                A_robin.axpy(coef_ns, lhs_ns)

                coef_t = float(self.forms['u']['transpiration']['coef']
                               [i_bnd])
                lhs_t = self.mat['u']['lhs_trans'][i_bnd]
                A_robin.axpy(coef_t, lhs_t)

        if self.mat['u']['p_trans']:
            assert (self.options['timemarching']['fractionalstep']['scheme'] ==
                    'CT')
            bu.axpy(-1., _mat_vec(self.mat['u']['p_trans'], self.p.x.petsc_vec))

        return A_robin

    def solve_tentative_velocity(self):
        ''' Solve tentative velocity PDE '''
        timer = Timer('Z solve u_ten')
        self.update_velocity_bcs()

        if self._using_mapdd:
            self.u0_mapdd.x.array[:] = self.u0.x.array

        A = self.assemble_tentative_velocity()

        bu = self.build_rhs_tentative_velocity()
        A_robin = self.assemble_robin_tentative_velocity(A, bu)

        if (self.options['timemarching']['fractionalstep']['scheme'] in
                ['IPCS', 'CT']) and not self._using_mapdd:
            # assembly of A, b done; u0_lst is "free"; assign
            # u_i(corrected, u^k) for next iteration
            self.u0.x.array[:] = self.u.x.array

        if A_robin:
            _apply_dbc_rows_to_mat(A_robin, self.bc_dict['u']['dirichlet'])
            set_bc(bu, self.bc_dict['u']['dirichlet'])
            self.solver_u_ten.set_operator(A_robin)

        else:
            _apply_dbc_rows_to_mat(A, self.bc_dict['u']['dirichlet'])
            set_bc(bu, self.bc_dict['u']['dirichlet'])
            self.solver_u_ten.set_operator(A)


        self.solver_u_ten.solve(self.u.x.petsc_vec, bu)

        if self.solver_u_ten.conv_reason < 0:
            self.logger.error('Solver u_ten DIVERGED ({})'.
                              format(self.solver_u_ten.conv_reason))
            if len(self.iterations_ksp['u_ten']) > 0:
                self._diverged = True

        self.iterations_ksp['u_ten'].append(self.solver_u_ten.iterations)
        self.residuals_ksp['u_ten'].append(self.solver_u_ten.residuals)

        timer.stop()

    def build_rhs_velocity_update(self):
        ''' Build RHS vector for tentative velocity solve, for the i'th
            component. '''
        if (self.options['timemarching']['fractionalstep']['scheme'] == 'IPCS'
                and self.t > self.options['timemarching']['dt'] + 1e-14):
            # inverse of 1.5, as multiplied on RHS
            cf = 2./3.
        else:
            cf = 1.
        bu = _mat_vec(self.mat['u']['gradp'], self.phi.x.petsc_vec)
        bu.scale(cf)
        return bu

    def solve_velocity_update(self):
        ''' Solve velocity update PDE '''

        timer = Timer('Z solve u_upd')

        if (self.options['timemarching']['fractionalstep']['scheme'] ==
                'CTp') or self._using_mapdd:
            # CTp: use old tentative velocity in time disc. term on RHS
            self.u0.x.array[:] = self.u.x.array

        bu = self.build_rhs_velocity_update()

        if self._applying_dc_on_update:
            set_bc(bu, self.bc_dict['u']['dirichlet'])

        self.solver_u_upd.solve(self.du.x.petsc_vec, bu)
        self.u.x.petsc_vec.axpy(1.0, self.du.x.petsc_vec)

        if self.solver_u_upd.conv_reason < 0:
            self.logger.error('Solver u_upd DIVERGED ({})'.
                              format(self.solver_u_upd.conv_reason))
            self._diverged = True

        self.iterations_ksp['u_upd'].append(
            self.solver_u_ten.iterations)
        self.residuals_ksp['u_upd'].append(self.solver_u_ten.residuals)

        timer.stop()


class PETScSolver(LoggerBase):
    ''' PETSc preconditioned Krylov solvers. '''
    _initialized = False

    def __init__(self, options, opt_prefix, logging_fh=None, verbose=True):
        super().__init__()
        self._logging_filehandler = logging_fh
        if self._logging_filehandler:
            self.logger.addHandler(self._logging_filehandler)

        self.residuals = []
        self.iterations = None
        self.timing = {}
        self.conv_reason = 0
        self._verbose = verbose

        self.options = options
        self.opt_prefix = opt_prefix

        if opt_prefix not in ('d_', 'u_ten_', 'p_', 'u_upd_', 'p_mass_'):
            raise ValueError('KSP options prefix needs to be one of \'d_\''
                             '\'u_ten_\', \'p_\', \'u_upd\', \'p_mass\'. [{}]'.
                             format(opt_prefix))

        self._setup_from_inputfile = bool(
            'inputfile' in self.options['linear_solver'] and
            self.options['linear_solver']['inputfile'])

        # set global PETSc options only once!
        # the descructor sets this to False
        if not self._initialized:
            type(self)._initialized = True
            self.setup_ksp()
        self.create_ksp()

    def __del__(self):
        ''' Clean up PETScOptions '''
        # super().__del__()
        self.close_logs()
        PETSc.Options().clear()
        type(self)._initialized = False

    def setup_ksp(self):
        if self._setup_from_inputfile:
            inputfile = self.options['linear_solver']['inputfile']
            self.logger.info('Initializing PETSc Krylov solver from file {}'.
                             format(inputfile))
            self.param = inout.read_parameters(inputfile)
            self.dump_parameters(inputfile)
            self._set_options()
        else:
            if not self.options['linear_solver']['method']:
                raise Exception('Specify inbuilt solver (lu, mumps) or input '
                                'file for PETSc!')
            self.logger.info('Initializing {} solvers'.
                             format(self.options['linear_solver']['method']))
            self._set_options_default()

        self.logger.info('  PETScOptions:')
        for k, v in sorted(PETSc.Options().getAll().items()):
            if v is None:
                self.logger.info('    -' + k)
            else:
                self.logger.info('    -{} {}'.format(k, v))

    def create_ksp(self):
        ''' Create KSP instance and set options '''
        self.logger.info('  Creating KSP from options ({})'.
                         format(self.opt_prefix))
        self.ksp = PETSc.KSP().create()
        self.ksp.setOptionsPrefix(self.opt_prefix)
        self.ksp.setFromOptions()
        self.ksp.setConvergenceHistory(reset=True)

    def set_operator(self, A, Ap=None):
        ''' Set operator A of linear problem Ax = b and set-up solver and
        preconditioner matrix Ap.
        Args:
            A:       PETSc.Mat matrix
            Ap:      precondition matrix, PETSc.Mat (optional)
        '''
        if not isinstance(A, PETSc.Mat):
            raise Exception(f"Unknown matrix type: {type(A)}")
        if Ap is not None and not isinstance(Ap, PETSc.Mat):
            raise Exception(f"Unknown preconditioner matrix type: {type(Ap)}")
        self.ksp.setOperators(A, Ap)
        self.ksp.setUp()

    def solve(self, x, b, A=None):
        ''' Solve the system Ax = b. If A is not given, it is supposed that it
        was set beforehand via set_operator(A).
        Saves iteration count, residuals and termination reason.

        Args:
            x           solution PETSc.Vec
            b           rhs PETSc.Vec
            A (opt)     matrix PETSc.Mat
        '''
        if A is not None:
            self.ksp.setOperators(A)
            self.ksp.setUp()

        t0 = Timer('Z PETScSolver SOLVE ')
        self.ksp.solve(b, x)
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                      mode=PETSc.ScatterMode.FORWARD)

        self.timing['ksp'] = t0.stop()
        self.iterations = self.ksp.getIterationNumber()
        self.residuals = self.ksp.getConvergenceHistory()
        self.conv_reason = self.ksp.getConvergedReason()

        if len(self.residuals) > 0 and self._verbose:
            convstr = 'CONVERGED' if self.conv_reason > 0 else 'DIVERGED'
            self.logger.info(
                '{pref}  {c} ({cr}) after {it} iterations. Residual: {res}'.
                format(pref=self.opt_prefix, c=convstr, cr=self.conv_reason,
                       it=self.iterations, res=self.residuals[-1]))

    def _set_options(self):
        # self.pc_name = self.param['config_name']
        self.petsc_options = [s.split(None, 1) for s in
                              self.param['petsc_options']]
        # self.logger.info('  PC: {0}'.format(self.pc_name))
        self.logger.info('  Setting PETScOptions:')
        for popt in self.petsc_options:
            # self.logger.info('  -' + ' '.join(popt))
            PETSc.Options().setValue(*popt)

    def _set_options_default(self):
        ''' Set default options '''
        meth = self.options['linear_solver']['method']
        if meth.lower() == 'mumps':
            PETSc.Options().setValue('d_ksp_type', 'preonly') 
            PETSc.Options().setValue('d_pc_type', 'lu')

            PETSc.Options().setValue('u_ten_ksp_type', 'preonly')
            PETSc.Options().setValue('u_ten_pc_type', 'lu')
            PETSc.Options().setValue('u_ten_pc_factor_mat_solver_package', 'mumps')
            PETSc.Options().setValue('u_ten_mat_mumps_icntl_14', 40)
            PETSc.Options().setValue('p_ksp_type', 'preonly')
            PETSc.Options().setValue('p_pc_type', 'lu')
            PETSc.Options().setValue('p_pc_factor_mat_solver_package', 'mumps')
            PETSc.Options().setValue('p_mat_mumps_icntl_14', 40)
            PETSc.Options().setValue('u_upd_ksp_type', 'preonly')
            PETSc.Options().setValue('u_upd_pc_type', 'lu')
            PETSc.Options().setValue('u_upd_pc_factor_mat_solver_package', 'mumps')
            PETSc.Options().setValue('u_upd_mat_mumps_icntl_14', 40)

            PETSc.Options().setValue('p_mass_ksp_type', 'preonly')
            PETSc.Options().setValue('p_mass_pc_type', 'lu')
            PETSc.Options().setValue('p_mass_pc_factor_mat_solver_package', 'mumps')
            PETSc.Options().setValue('p_mass_mat_mumps_icntl_14', 40)

        elif meth.lower() in ('petsc', 'lu'):
            PETSc.Options().setValue('d_ksp_type', 'preonly') 
            PETSc.Options().setValue('d_pc_type', 'lu')

            PETSc.Options().setValue('u_ten_ksp_type', 'preonly')
            PETSc.Options().setValue('u_ten_pc_type', 'lu')
            PETSc.Options().setValue('p_ksp_type', 'preonly')
            PETSc.Options().setValue('p_pc_type', 'lu')
            PETSc.Options().setValue('u_upd_ksp_type', 'preonly')
            PETSc.Options().setValue('u_upd_pc_type', 'lu')

            PETSc.Options().setValue('p_mass_ksp_type', 'preonly')
            PETSc.Options().setValue('p_mass_pc_type', 'lu')

        elif meth.lower() == 'default':
            PETSc.Options().setValue('d_ksp_type', 'cg') 
            PETSc.Options().setValue('d_ksp_rtol', 1.0e-6)
            
            # BiCGSTAB/Jacobi + CG/GAMG
            PETSc.Options().setValue('u_ten_ksp_type', 'bcgs')
            # PETSc.Options().setValue('u_ten_ksp_converged_reason')
            # PETSc.Options().setValue('u_ten_ksp_monitor_true_residual')
            PETSc.Options().setValue('u_ten_ksp_rtol', 1.0e-6)
            PETSc.Options().setValue('u_ten_ksp_initial_guess_nonzero')
            PETSc.Options().setValue('u_ten_pc_type', 'jacobi')
            # PETSc.Options().setValue('u_ten_pc_type', 'gamg')
            # PETSc.Options().setValue('u_ten_pc_gamg_type', 'agg')
            # PETSc.Options().setValue('u_ten_pc_gamg_threshold', 0.03)
            # PETSc.Options().setValue('u_ten_pc_gamg_square_graph', 10)
            # PETSc.Options().setValue('u_ten_pc_gamg_sym_graph')
            # PETSc.Options().setValue('u_ten_mg_levels_ksp_type', 'richardson')
            # PETSc.Options().setValue('u_ten_mg_levels_pc_type', 'sor')

            PETSc.Options().setValue('p_ksp_type', 'cg')
            # gmres would remove nullspace automatically
            # PETSc.Options().setValue('p_ksp_converged_reason')
            # PETSc.Options().setValue('p_ksp_monitor_true_residual')
            PETSc.Options().setValue('p_ksp_rtol', 1.0e-8)
            PETSc.Options().setValue('p_pc_type', 'gamg')
            PETSc.Options().setValue('p_pc_gamg_type', 'agg')
            PETSc.Options().setValue('p_pc_gamg_threshold', 0.03)
            PETSc.Options().setValue('p_pc_gamg_square_graph', 10)
            PETSc.Options().setValue('p_pc_gamg_sym_graph')     # needed for parallel
            PETSc.Options().setValue('p_mg_levels_ksp_type', 'richardson')
            PETSc.Options().setValue('p_mg_levels_pc_type', 'sor')

            # 3D poisson: taken from Chris Richardson's and Jack Hale's
            # poisson.py
            # PETSc.Options().setValue('p_pc_type', 'hypre')
            # PETSc.Options().setValue('p_pc_hypre_type', 'boomeramg')
            # PETSc.Options().setValue('p_pc_hypre_boomeramg_agg_nl', 4)
            # PETSc.Options().setValue('p_pc_hypre_boomeramg_agg_num_paths', 2)
            # # Truncation factor for interpolation (note: increasing towrds 1
            # # appears to reduce memory useage)
            # PETSc.Options().setValue('p_pc_hypre_boomeramg_truncfactor', 0.9)
            # # Max elements per row for interpolation operator
            # PETSc.Options().setValue('p_pc_hypre_boomeramg_P_max', 5)
            # # PETSc.Options().setValue('p_pc_hypre_boomeramg_max_levels', 10)
            # # Strong threshold (BoomerAMG docs recommend 0.5-0.6 for 3D
            # #   Poisson)
            # PETSc.Options().setValue('p_pc_hypre_boomeramg_strong_threshold', 0.5)

            PETSc.Options().setValue('u_upd_ksp_type', 'cg')
            # PETSc.Options().setValue('u_upd_ksp_converged_reason')
            # PETSc.Options().setValue('u_upd_ksp_monitor_true_residual')
            PETSc.Options().setValue('u_upd_ksp_rtol', 1.0e-8)
            PETSc.Options().setValue('u_upd_ksp_initial_guess_nonzero')
            PETSc.Options().setValue('u_upd_pc_type', 'jacobi')

            PETSc.Options().setValue('p_mass_ksp_type', 'cg')
            # PETSc.Options().setValue('p_mass_ksp_converged_reason')
            # PETSc.Options().setValue('p_mass_ksp_monitor_true_residual')
            PETSc.Options().setValue('p_mass_ksp_rtol', 1.0e-8)
            PETSc.Options().setValue('p_mass_ksp_initial_guess_nonzero')
            PETSc.Options().setValue('p_mass_pc_type', 'jacobi')

        else:
            raise Exception('linear solver method set "{}" unknown'.
                            format(meth))

    @rank0
    def dump_parameters(self, inputfile):
        ''' Write petsc inputfile to results directory. '''
        path = self.options['io']['write_path']
        if not os.path.exists(path):
            os.makedirs(path)
        shutil.copy2(inputfile, path + '/petsc.yaml')


class Mat(object):
    def __init__(self, form, bcs=[], assemble=True):
        ''' '''
        self.form = form
        self.bcs = bcs
        self.mat = None
        self._assembled = False
        self._bcs_applied = False
        if assemble:
            self.assemble()

    def assemble(self):
        ''' '''
        if self.mat is None:
            self.mat = _assemble_mat(self.form)
        else:
            _assemble_mat(self.form, mat=self.mat)
        self._assembled = True
        self._bcs_applied = False

    def apply_bcs(self):
        ''' '''
        assert self._assembled
        _apply_dbc_to_mat(self.mat, self.bcs)
        self._bcs_applied = True

    def axpy(self, x, A):
        ''' '''
        assert self._assembled
        if isinstance(A, self.__class__):
            A = A.mat
        self.mat.axpy(x, A)


def solver(problem):
    opt = problem.options['timemarching']['fractionalstep']
    if 'coupled_velocity' in opt and opt['coupled_velocity']:
        return SolverCoupled(problem)
    else:
        return Solver(problem)
