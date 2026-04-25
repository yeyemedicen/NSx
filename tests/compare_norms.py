"""
Compare velocity and pressure norms between legacy FEniCS and DOLFINx checkpoints.

Pressure gauge: both fields are pinned to the same reference DOF (the mesh node
closest to the origin, which is guaranteed to exist in both after coordinate
alignment).  At each timestep the gauge constant C(t) = p_dol[ref] - p_leg[ref]
is removed before computing norms, because p and p+C(t) are both valid NS
solutions for a given u.

Requires both legacy_fields.npz and dolfinx_fields.npz (produced by the two
extract_*.py scripts).  Runs in any Python environment with numpy + matplotlib.

Usage:
    python tests/compare_norms.py [--plot]
"""
import sys
import numpy as np

LEGACY  = '/Users/yeye/NSx/tests/results/legacy_fields.npz'
DOLFINX = '/Users/yeye/NSx/tests/results/dolfinx_fields.npz'
DT      = 0.001   # time step


def _sort_index(coords):
    """Stable sort by (x, y) so both meshes are aligned by geometry."""
    return np.lexsort((coords[:, 1], coords[:, 0]))


def _ref_dof(coords_sorted):
    """
    Return the index (in the already-sorted array) of the reference DOF used
    to fix the pressure gauge.

    We pick the node on boundary 3 (right wall, x=1.5) closest to the
    bottom-right corner.  Boundary 3 carries the Neumann outflow BC which
    becomes p=0 Dirichlet in the pressure Poisson step, so both solvers
    enforce p=0 there — making C(t)=0 identically.
    """
    x_max = coords_sorted[:, 0].max()
    on_b3 = np.where(np.isclose(coords_sorted[:, 0], x_max))[0]
    # among boundary-3 nodes choose the one with minimum y
    return int(on_b3[np.argmin(coords_sorted[on_b3, 1])])


def main():
    plot = '--plot' in sys.argv

    leg = np.load(LEGACY,  allow_pickle=False)
    dol = np.load(DOLFINX, allow_pickle=False)

    steps_leg = leg['steps'].tolist()
    steps_dol = dol['steps'].tolist()
    if steps_leg != steps_dol:
        raise ValueError(f'Step lists differ:\n  legacy:  {steps_leg}\n  dolfinx: {steps_dol}')

    # sort-indices to align vertices by coordinate
    idx_leg = _sort_index(leg['coords'])
    idx_dol = _sort_index(dol['coords'])

    coords_leg = leg['coords'][idx_leg]
    coords_dol = dol['coords'][idx_dol]
    coord_err = np.max(np.abs(coords_leg - coords_dol))
    if coord_err > 1e-10:
        raise ValueError(f'Mesh coordinates do not match after sorting (max diff = {coord_err:.2e}).')
    print(f'Mesh coordinate alignment: max diff = {coord_err:.2e}  ✓')

    # reference DOF for pressure gauge
    ref = _ref_dof(coords_leg)
    print(f'Pressure reference DOF: sorted index {ref},  coords = {coords_leg[ref]}\n')

    times = np.array(steps_leg) * DT

    u_norms_leg, u_norms_dol = [], []
    p_norms_leg, p_norms_dol = [], []
    u_rel_err  = []
    p_rel_err  = []
    p_gauge    = []   # C(t) = p_dol[ref] - p_leg[ref]

    header = (f"{'step':>5}  {'t':>6}  "
              f"{'|u| leg':>12}  {'|u| dol':>12}  {'|Δu|/|u|':>10}  "
              f"{'|p-p0| leg':>12}  {'|p-p0| dol':>12}  {'|Δp|/|p|':>10}  "
              f"{'C(t)':>12}")
    print(header)
    print('-' * len(header))

    for step in steps_leg:
        u_l = leg[f'u_{step}'][idx_leg]   # (N, 2)
        u_d = dol[f'u_{step}'][idx_dol]

        p_l = leg[f'p_{step}'][idx_leg]   # (N,)
        p_d = dol[f'p_{step}'][idx_dol]

        # ── velocity ─────────────────────────────────────────────────────────
        nl = np.linalg.norm(u_l)
        nd = np.linalg.norm(u_d)
        du = np.linalg.norm(u_d - u_l)
        u_rel = du / nl if nl > 1e-15 else float('nan')

        # ── pressure: pin to reference DOF ───────────────────────────────────
        # C(t) is the time-dependent gauge constant between the two solutions
        Ct = p_d[ref] - p_l[ref]
        p_d_gauged = p_d - Ct           # now p_d_gauged[ref] == p_l[ref]

        # norms relative to the reference DOF value (p - p[ref])
        p_l_0 = p_l - p_l[ref]
        p_d_0 = p_d_gauged - p_d_gauged[ref]   # == p_d - Ct - p_l[ref] = p_l_0 - corrected

        pl0 = np.linalg.norm(p_l_0)
        pd0 = np.linalg.norm(p_d_0)
        dp  = np.linalg.norm(p_d_0 - p_l_0)
        p_rel = dp / pl0 if pl0 > 1e-15 else float('nan')

        u_norms_leg.append(nl);  u_norms_dol.append(nd)
        p_norms_leg.append(pl0); p_norms_dol.append(pd0)
        u_rel_err.append(u_rel)
        p_rel_err.append(p_rel)
        p_gauge.append(Ct)

        t = step * DT
        print(f'{step:5d}  {t:6.3f}  '
              f'{nl:12.6e}  {nd:12.6e}  {u_rel:10.3e}  '
              f'{pl0:12.6e}  {pd0:12.6e}  {p_rel:10.3e}  '
              f'{Ct:12.6e}')

    u_rel_err = np.array(u_rel_err)
    p_rel_err = np.array(p_rel_err)
    p_gauge   = np.array(p_gauge)

    print()
    print(f'Velocity  — max rel error: {np.nanmax(u_rel_err):.3e}  '
          f'  mean rel error: {np.nanmean(u_rel_err):.3e}')
    print(f'Pressure  — max rel error: {np.nanmax(p_rel_err[1:]):.3e}  '
          f'  mean rel error: {np.nanmean(p_rel_err[1:]):.3e}')
    print(f'Gauge C(t) — max |C|: {np.max(np.abs(p_gauge)):.3e}  '
          f'  final C: {p_gauge[-1]:.6e}')

    if plot:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        fig.suptitle('Legacy FEniCS vs DOLFINx — checkpoint comparison')

        ax = axes[0, 0]
        ax.plot(times, u_norms_leg, label='legacy')
        ax.plot(times, u_norms_dol, '--', label='dolfinX')
        ax.set_title('‖u‖ (DOF L2 norm)')
        ax.set_xlabel('t'); ax.legend()

        ax = axes[0, 1]
        ax.semilogy(times[1:], u_rel_err[1:])
        ax.set_title('Velocity relative error  ‖Δu‖/‖u‖')
        ax.set_xlabel('t')

        ax = axes[0, 2]
        ax.plot(times, p_gauge)
        ax.set_title('Gauge constant  C(t) = p_dol[ref] − p_leg[ref]')
        ax.set_xlabel('t')

        ax = axes[1, 0]
        ax.plot(times, p_norms_leg, label='legacy  (p − p[ref])')
        ax.plot(times, p_norms_dol, '--', label='dolfinX (p − p[ref])')
        ax.set_title('‖p − p[ref]‖ (DOF L2 norm, gauge-fixed)')
        ax.set_xlabel('t'); ax.legend()

        ax = axes[1, 1]
        ax.semilogy(times[1:], p_rel_err[1:])
        ax.set_title('Pressure relative error  ‖Δp‖/‖p‖  (gauge-fixed)')
        ax.set_xlabel('t')

        ax = axes[1, 2]
        ax.plot(times[1:], u_rel_err[1:],  label='velocity')
        ax.plot(times[1:], p_rel_err[1:], label='pressure (gauge-fixed)')
        ax.set_yscale('log')
        ax.set_title('Relative errors  (log scale)')
        ax.set_xlabel('t'); ax.legend()

        plt.tight_layout()
        out_fig = '/Users/yeye/NSx/tests/results/comparison.png'
        plt.savefig(out_fig, dpi=150)
        print(f'\nPlot saved → {out_fig}')
        plt.show()


if __name__ == '__main__':
    main()
