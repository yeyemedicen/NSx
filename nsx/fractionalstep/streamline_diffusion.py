''' Streamline diffusion (SUPG) stabilization parameters — DOLFINx port

All tau parameters are now pure UFL expressions; the legacy C++ compiled
expressions and the dolfin.Metric helper have been removed.
'''

import numpy as np
import dolfinx.fem as fem
import ufl
from ufl import (
    CellDiameter, JacobianInverse,
    sqrt, inner, dot, transpose, inv,
    conditional, lt, gt,
)
from petsc4py import PETSc
from ..logger.logger import LoggerBase

__all__ = ['SDParameter', 'Metric']


def _C(mesh, val):
    return fem.Constant(mesh, np.array(val, dtype=PETSc.ScalarType))


def _metric_tensor(mesh, F=None):
    # Covariant metric G=(dxi/dx)^T(dxi/dx). Fixed mesh: dxi/dx=Jinv. Under ALE
    # x=X+d so dxi/dx=Jinv.F^-1 and the DEFORMED metric is F^-T (Jinv^T Jinv) F^-1.
    Jinv = JacobianInverse(mesh)
    G = transpose(Jinv) * Jinv
    if F is not None:
        Finv = inv(F)
        G = transpose(Finv) * G * Finv
    return G


class SDParameter(LoggerBase):
    def __init__(self, options, mesh, mu, rho, k, logging_filehandler):
        super().__init__()
        self._logging_filehandler = logging_filehandler
        if self._logging_filehandler:
            self.logger.addHandler(self._logging_filehandler)

        self.options = options
        self.mesh = mesh
        self.mu = mu
        self.rho = rho
        self.k = k
        self._tau_func = None   # DG0 Function for element-constant tau
        self._tau_expr = None   # fem.Expression to interpolate into _tau_func

    def update_tau(self):
        ''' Re-evaluate element-constant tau (call before each conv assembly). '''
        if self._tau_func is not None:
            self._tau_func.interpolate(self._tau_expr)

    def _make_element_constant_tau(self, tau_ufl):
        ''' Project UFL tau expression to DG0 (piecewise-constant per element),
        matching legacy FEniCS CompiledExpression with DG0 element. '''
        from dolfinx.fem import functionspace, Function, Expression as FEMExpression
        V_dg0 = functionspace(self.mesh, ("DG", 0))
        self._tau_func = Function(V_dg0)
        self._tau_expr = FEMExpression(tau_ufl,
                                       V_dg0.element.interpolation_points)
        self.update_tau()
        return self._tau_func

    def stabilization_parameter(self, u_conv, u_conv_assigned=None, F=None, w=None):
        ''' SD stabilization parameter.

        Args:
            u_conv  convecting velocity (vector function or as_vector(u_lst)
                    for component wise treatment)
            u_conv_assigned     convecting velocity vector function (assigned)
                    when using component splitting

        Returns:
            tau     stabilization parameter (ufl expression)
        '''
        parameter = (self.options['fem']['stabilization']
                     ['streamline_diffusion']['parameter'])

        if u_conv_assigned is None:
            u_conv_assigned = u_conv

        if parameter == 'shakib':
            tau = self.tau_shakib(u_conv, u_conv_assigned, F, w)
        elif parameter in ('standard', 'default'):
            tau = self.tau_standard(u_conv, u_conv_assigned)
        elif parameter == 'klr':
            tau = self.tau_klr(u_conv, u_conv_assigned)
        elif parameter == 'codina':
            tau = self.tau_codina(u_conv, u_conv_assigned)
        else:
            raise Exception('streamline diffusion parameter unknown: {}'.
                            format(parameter))

        return tau

    def tau_shakib(self, u_conv, u_conv_assigned, F=None, w=None):
        opt = self.options['fem']['stabilization']['streamline_diffusion']
        alpha = _C(self.mesh, 2.)
        rho = self.rho
        mu = self.mu
        k = self.k
        elem_const = bool(opt.get('parameter_element_constant', False))

        cinv_default = 30. if opt['length_scale'] == 'metric' else 12.
        Cinv = _C(self.mesh, opt['Cinv']) if isinstance(opt['Cinv'], (int, float)) \
            else _C(self.mesh, cinv_default)

        # ALE: convect by mesh-relative velocity (u-w) + deformed metric F^-T G F^-1
        uc = u_conv_assigned if w is None else (u_conv_assigned - w)
        if opt['length_scale'] == 'metric':
            G = _metric_tensor(self.mesh, F)
            tau_ufl = 1. / sqrt(
                (alpha * k)**2
                + dot(uc, G * uc)
                + Cinv * (mu / rho)**2 * inner(G, G)
            )
            len_scale = 'metric'
        elif opt['length_scale'] == 'max':
            h = ufl.MaxCellEdgeLength(self.mesh)
            u_norm2 = dot(u_conv_assigned, u_conv_assigned)
            tau_ufl = 1. / sqrt(
                (alpha * k)**2
                + (2. / h)**2 * u_norm2
                + (Cinv * mu / rho / h**2)**2
            )
            len_scale = 'max'
        else:
            h = CellDiameter(self.mesh)
            u_norm2 = dot(u_conv_assigned, u_conv_assigned)
            tau_ufl = 1. / sqrt(
                (alpha * k)**2
                + (2. / h)**2 * u_norm2
                + (Cinv * mu / rho / h**2)**2
            )
            len_scale = 'average'

        self.logger.info('SD parameter: shakib\n\tlength scale: {ls}\n'
                         '\tC_inv: {c}'.format(ls=len_scale, c=float(np.asarray(Cinv.value).flat[0])))

        if elem_const:
            return self._make_element_constant_tau(tau_ufl)
        return tau_ufl

    def tau_standard(self, u_conv, u_conv_assigned):
        opt = self.options['fem']['stabilization']['streamline_diffusion']
        rho = self.rho
        mu = self.mu
        eps = 1e-10
        elem_const = bool(opt.get('parameter_element_constant', False))

        if opt['length_scale'] == 'metric':
            G = _metric_tensor(self.mesh)
            u_inner_metric = sqrt(dot(u_conv_assigned, G * u_conv_assigned))
            u_norm2 = dot(u_conv_assigned, u_conv_assigned)
            Pe = u_norm2 / (u_inner_metric + eps) * rho / mu
            xi = conditional(lt(Pe, 3.), Pe / 3., ufl.as_ufl(1.))
            tau_ufl = conditional(gt(u_inner_metric, eps),
                                  xi / (u_inner_metric + eps),
                                  ufl.as_ufl(0.))
            len_scale = 'metric'
        else:
            h = CellDiameter(self.mesh)
            u_norm = sqrt(dot(u_conv_assigned, u_conv_assigned))
            Pe = 0.5 * u_norm * h * rho / mu
            xi = conditional(lt(Pe, 3.), Pe / 3., ufl.as_ufl(1.))
            tau_ufl = conditional(gt(u_norm, eps),
                                  0.5 * h / (u_norm + eps) * xi,
                                  ufl.as_ufl(0.))
            len_scale = 'average'

        self.logger.info('SD parameter: standard\n\tlength scale: {ls}'
                         .format(ls=len_scale))

        if elem_const:
            return self._make_element_constant_tau(tau_ufl)
        return tau_ufl

    def tau_klr(self, u_conv, u_conv_assigned):
        h = CellDiameter(self.mesh)
        mu = self.mu
        rho = self.rho
        k = self.k
        deg = self.options['fem']['velocity_space']
        pk_val = float(''.join(c for c in str(deg) if c.isdigit()) or '1') ** 2
        pk = _C(self.mesh, pk_val)
        eps = 1e-10
        u_norm = sqrt(dot(u_conv_assigned, u_conv_assigned)) + eps
        a = 0.5 * h / u_norm
        b = 2. / 3. / k
        c0 = h**2 / (pk * mu / rho)
        tau = conditional(lt(a, conditional(lt(b, c0), b, c0)),
                          a,
                          conditional(lt(b, c0), b, c0))

        self.logger.info('SD parameter: KLR')
        return tau

    def tau_codina(self, u_conv, u_conv_assigned):
        h = CellDiameter(self.mesh)
        rho = self.rho
        mu = self.mu
        k = self.k
        tau = 1. / (4 * mu / rho / h**2 + sqrt(inner(u_conv, u_conv)) / h + 1.5 * k)

        self.logger.info('SD parameter: Codina')
        return tau


class Metric:
    ''' Metric tensor G = J^{-T} J^{-1} as a pure UFL expression. '''

    def __init__(self, mesh):
        self.mesh = mesh

    def create(self):
        return _metric_tensor(self.mesh)
