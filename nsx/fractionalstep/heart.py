''' 0D left heart for the "heart" inlet boundary condition.

Circuit (all CGS: dyn/cm2, cm3, s; R in dyn s/cm5, L in dyn s2/cm5,
E in dyn/cm5, C in cm5/dyn):

    P_LA --[R_MV, L_MV, diode]--> LV: P_LV = E(t)*(V - V_0)
         --[R_AV, L_AV, diode]--> node P_ao (compliance C_ao) --Q_3D--> 3D inlet

    dV/dt        = Q_MV - Q_AV
    L_MV dQ_MV/dt = P_LA - P_LV - R_MV*Q_MV,   Q_MV >= 0  (mitral valve)
    L_AV dQ_AV/dt = P_LV - P_ao - R_AV*Q_AV,   Q_AV >= 0  (aortic valve)
    C_ao dP_ao/dt = Q_AV - Q_3D

A valve with L = 0 is a pure resistive diode, Q = max(dP, 0)/R. The left
atrium is a constant pressure P_LA. P_ao is what the 3D sees: it is imposed as
a pressure Dirichlet on the inlet, like the explicit Windkessel's Pl.

E(t) = (E_max - E_min)*En(t_c/t_max) + E_min, t_c = t mod T, with the
normalised double-Hill elastance of Stergiopulos et al. (1996),
    En(x) = k * (x/a1)^n1/(1 + (x/a1)^n1) * 1/(1 + (x/a2)^n2),
k chosen so max En = 1 (computed once, numerically).

The coupling is EXPLICIT (as the explicit Windkessel): Q_3D is measured from
the tentative velocity and held fixed over the step, the 0D is advanced over
dt in `substeps` forward-Euler substeps, and the new P_ao drives the pressure
projection.
'''
import numpy as np

# Stergiopulos double-Hill shape constants
HILL = {'a1': 0.7, 'a2': 1.17, 'n1': 1.9, 'n2': 21.9}


def double_hill(x, a1=HILL['a1'], a2=HILL['a2'], n1=HILL['n1'], n2=HILL['n2']):
    ''' Un-normalised double-Hill shape, x = t_c/t_max. '''
    x = np.maximum(x, 0.0)
    g1 = (x/a1)**n1
    g2 = (x/a2)**n2
    return g1/(1 + g1) * 1/(1 + g2)


def hill_normalisation():
    ''' k such that max_x k*double_hill(x) = 1. '''
    x = np.linspace(0, 3, 300001)
    return 1.0/double_hill(x).max()


def elastance(t, prm):
    ''' LV elastance E(t) [dyn/cm5]. prm: dict of floats incl. 'k_hill'. '''
    tc = np.mod(t, prm['T'])
    En = prm['k_hill']*double_hill(tc/prm['t_max'])
    return (prm['E_max'] - prm['E_min'])*En + prm['E_min']


def _valve(Q, dP, R, L, h):
    ''' One substep of a diode valve. Closed valves do not regurgitate. '''
    if L > 0:
        Q = Q + h/L*(dP - R*Q)
        return max(Q, 0.0)
    return max(dP, 0.0)/R


def advance(st, prm, t0, dt, Q3d, nsub):
    ''' Advance the heart state from t0 to t0 + dt with Q_3D held fixed.

    Args:
        st   (dict): {'V', 'Q_MV', 'Q_AV', 'P_ao'}, floats, updated copy returned
        prm  (dict): parameters (floats), see module docstring
        t0   (float): start time of the step
        dt   (float): step
        Q3d  (float): flow INTO the 3D domain through the inlet [cm3/s]
        nsub (int):   number of forward-Euler substeps

    Returns:
        dict: new state, plus 'P_LV' and 'E' at t0 + dt and 'V_AV', the
        volume ejected through the aortic valve over the step
    '''
    s = dict(st)
    s['V_AV'] = 0.0
    h = dt/nsub
    for k in range(nsub):
        t = t0 + k*h
        E = elastance(t, prm)
        P_LV = E*(s['V'] - prm['V_0'])
        Q_MV = _valve(s['Q_MV'], prm['P_LA'] - P_LV, prm['R_MV'], prm['L_MV'], h)
        Q_AV = _valve(s['Q_AV'], P_LV - s['P_ao'], prm['R_AV'], prm['L_AV'], h)
        s['V'] += h*(Q_MV - Q_AV)
        s['V_AV'] += h*Q_AV
        s['P_ao'] += h/prm['C_ao']*(Q_AV - Q3d)
        s['Q_MV'], s['Q_AV'] = Q_MV, Q_AV
    s['E'] = elastance(t0 + dt, prm)
    s['P_LV'] = s['E']*(s['V'] - prm['V_0'])
    return s


def stability_timescale(prm):
    ''' Shortest explicit time scale of the 0D circuit [s]: the substep must
    stay well below it (forward Euler on dx/dt = -x/tau needs h < 2*tau). '''
    taus = [prm['R_AV']*prm['C_ao'],            # AV open, node charging
            prm['R_AV']/prm['E_max'],           # LV ejecting through AV
            prm['R_MV']/prm['E_max']]           # filling (worst case E)
    for R, L in ((prm['R_MV'], prm['L_MV']), (prm['R_AV'], prm['L_AV'])):
        if L > 0:
            taus.append(L/R)
    if prm['L_AV'] > 0:
        taus.append(np.sqrt(prm['L_AV']*prm['C_ao']))
    return min(taus)
