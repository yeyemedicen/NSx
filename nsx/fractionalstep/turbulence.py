'''Subgrid-scale turbulence models — eddy-viscosity closures.

WHY THIS EXISTS. On the CoAo phantom the model reproduces the jet (rigid, no
streamline diffusion: 372 cm/s forward against 349 measured) yet recovers only
12.5 mmHg of transcoarctation drop against a catheter 33.0 and a Borda-Carnot
estimate of 35.4. The rigid wall IS the mu -> infinity limit, so no wall
stiffness can close that gap: the missing ~20 mmHg is IRRECOVERABLE loss, i.e.
the turbulent destruction of the jet's kinetic energy in the post-stenotic
shear layer, and nothing in a laminar Navier-Stokes discretisation supplies it.

The models here add an eddy viscosity mu_t to the molecular mu, so the existing
viscous form is used unchanged with mu -> mu + mu_t.

CHOICE OF MODEL, and why WALE is the default. In an FSI where the WALL MOTION
is the measurement, a closure that leaves a spurious mu_t at the wall corrupts
the interface traction and therefore the stiffness being estimated. Ranked by
that criterion:

  wale       mu_t -> 0 like d^3 at a wall and vanishes in pure shear, so it does
             not pollute the WSS. Same cost as Smagorinsky. DEFAULT.
  vreman     also vanishes in laminar/near-wall flow; cheap; fewer hemodynamic
             reference constants than WALE.
  sigma      best-behaved of the algebraic closures (vanishes for ALL 2D and
             axisymmetric-expansion flows) but needs an SVD of the velocity
             gradient per quadrature point.
  smagorinsky  simplest and unconditionally dissipative, but mu_t != 0 in pure
             shear, so it damps the near-wall layer and biases WSS. Provided
             for comparison, NOT recommended when a solid is coupled.

All are evaluated on the CONVECTING velocity (the extrapolated u_conv of the
semi-implicit tentative step), so mu_t is a known coefficient and the tentative
solve stays LINEAR.

ALE. Gradients are taken in the DEFORMED configuration, grad(u).inv(F), and the
filter width uses the deformed cell volume J*CellVolume, so the model does not
drift as the mesh moves.
'''

import numpy as np
import dolfinx.fem as fem
import ufl
from ufl import (
    sym, grad, inner, dot, inv, tr, Identity, sqrt, CellVolume,
    conditional, lt,
)
from petsc4py import PETSc

__all__ = ['eddy_viscosity', 'default_options', 'describe']

# Literature constants. Nicoud & Ducros (1999) propose Cw in 0.5-0.6 for their
# tests; 0.325 is the value most used in the hemodynamic LES literature and is
# the conservative choice here (less dissipation).
_DEFAULTS = {
    'enabled': False,
    'model': 'wale',
    'Cw': 0.325,        # WALE
    'Cs': 0.17,         # Smagorinsky
    'Cv': 0.07,         # Vreman
    'Csig': 1.35,       # sigma
}


def default_options():
    return dict(_DEFAULTS)


def describe(opt):
    '''One-line human-readable summary for the log.'''
    m = opt.get('model', 'wale').lower()
    c = {'wale': 'Cw', 'smagorinsky': 'Cs', 'vreman': 'Cv',
         'sigma': 'Csig'}.get(m, 'Cw')
    return ('turbulence: %s subgrid model ENABLED (%s = %g, filter width = '
            '(J*CellVolume)^(1/3))' % (m.upper(), c, opt.get(c, _DEFAULTS[c])))


def _filter_width(mesh, J):
    '''Cubic-root of the DEFORMED cell volume.'''
    return (J * CellVolume(mesh)) ** (1.0 / 3.0)


def eddy_viscosity(mesh, u_conv, rho, opt, F=None, J=None):
    '''Return the eddy viscosity mu_t as a UFL expression (dynamic, not kinematic).

    Args:
        mesh    : the fluid mesh
        u_conv  : convecting velocity (known -> keeps the tentative step linear)
        rho     : density (Constant or float)
        opt     : the `fem > turbulence` options dict
        F, J    : ALE deformation gradient and its determinant (None = fixed mesh)

    Returns:
        UFL expression for mu_t, or None when disabled.
    '''
    if not opt.get('enabled', False):
        return None
    model = opt.get('model', 'wale').lower()
    if J is None:
        J = 1.0
    delta = _filter_width(mesh, J)
    eps = 1.0e-14

    # velocity gradient in the DEFORMED configuration
    g = grad(u_conv) if F is None else dot(grad(u_conv), inv(F))
    S = sym(g)
    SS = inner(S, S)

    if model == 'smagorinsky':
        Cs = opt.get('Cs', _DEFAULTS['Cs'])
        return rho * (Cs * delta) ** 2 * sqrt(2.0 * SS)

    if model == 'wale':
        Cw = opt.get('Cw', _DEFAULTS['Cw'])
        d = mesh.geometry.dim
        g2 = dot(g, g)
        Sd = sym(g2) - (1.0 / 3.0) * tr(g2) * Identity(d)
        SdSd = inner(Sd, Sd)
        num = SdSd ** 1.5
        den = SS ** 2.5 + SdSd ** 1.25
        # den -> 0 only where the velocity gradient vanishes, and num -> 0
        # faster there; the conditional keeps it finite at machine level.
        return rho * (Cw * delta) ** 2 * conditional(lt(den, eps), 0.0,
                                                     num / (den + eps))

    if model == 'vreman':
        Cv = opt.get('Cv', _DEFAULTS['Cv'])
        d = mesh.geometry.dim
        b = dot(g.T, g) * delta ** 2
        aa = inner(g, g)
        Bb = (b[0, 0] * b[1, 1] - b[0, 1] * b[1, 0]
              + b[0, 0] * b[2, 2] - b[0, 2] * b[2, 0]
              + b[1, 1] * b[2, 2] - b[1, 2] * b[2, 1]) if d == 3 else \
             (b[0, 0] * b[1, 1] - b[0, 1] * b[1, 0])
        return rho * Cv * conditional(lt(aa, eps), 0.0,
                                      sqrt(abs(Bb) / (aa + eps)))

    raise ValueError(
        "turbulence model %r not supported. Use one of: "
        "wale (default), smagorinsky, vreman." % model)
