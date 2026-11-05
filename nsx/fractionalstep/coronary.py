''' Lumped coronary bed of Kim et al. 2010 (Ann. Biomed. Eng. 38:3195) on an
outlet, as used by the 'coronary' boundary condition (problem._coronary,
Solver.solve_coronary / sync_coronary). Pure python, no dolfinx, so the 0D
pre-flight (angio/tools/heart0d_preflight.py) runs the identical code.

Circuit, from the outlet face (pressure p, outward flux Q) to the veins:

    p --R_a-- P1 --R_am-- P2 --R_v-- P_v
              |           |
             C_a         C_im
              |           |
            ground      P_im(t) = lambda * P_LV(t)

    p = P1 + R_a Q
    C_a  dP1/dt          = Q - (P1 - P2)/R_am
    C_im d(P2 - P_im)/dt = (P1 - P2)/R_am - (P2 - P_v)/R_v

Backward Euler over one step dt, with x = P1^{n+1}, y = P2^{n+1},
a = C_a/dt, c = C_im/dt, g = 1/R_am, gv = 1/R_v, dP_im = P_im^{n+1} - P_im^n:

    (a + g) x - g y            = Q + a P1^n
    -g x + (c + g + gv) y      = c (P2^n + dP_im) + gv P_v

so x, and therefore p = x + R_a Q, is AFFINE in Q^{n+1}:
p = gamma Q + b. That is the condensed form NSx solves inside the projection.

Units: whatever the caller uses, consistently (NSx: CGS).
'''


def _system(prm, dt, P1, P2, dP_im):
    a = prm['C_a']/dt
    c = prm['C_im']/dt
    g = 1.0/prm['R_am']
    gv = 1.0/prm['R_v']
    r1 = a*P1                                   # + Q
    r2 = c*(P2 + dP_im) + gv*prm['P_v']
    A11, A12, A22 = a + g, -g, c + g + gv
    D = A11*A22 - A12*A12
    return A11, A12, A22, D, r1, r2


def condensed(prm, dt, P1, P2, dP_im):
    ''' (gamma, b) with p^{n+1} = gamma*Q^{n+1} + b at the outlet face. '''
    A11, A12, A22, D, r1, r2 = _system(prm, dt, P1, P2, dP_im)
    # x = (A22 (Q + r1) - A12 r2)/D
    gamma = prm['R_a'] + A22/D
    b = (A22*r1 - A12*r2)/D
    return gamma, b


def update(prm, dt, P1, P2, dP_im, Q):
    ''' (P1, P2) at t^{n+1} given the outward flux Q^{n+1}. '''
    A11, A12, A22, D, r1, r2 = _system(prm, dt, P1, P2, dP_im)
    x = (A22*(Q + r1) - A12*r2)/D
    y = (A11*r2 - A12*(Q + r1))/D
    return x, y


def gamma_only(prm, dt):
    ''' gamma does not depend on the state: R_a + (c + g + gv)/D. '''
    return condensed(prm, dt, 0.0, 0.0, 0.0)[0]
