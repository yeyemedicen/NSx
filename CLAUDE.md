# NSx — Navier-Stokes DOLFINx Solver


> **Working practice — shared by all six repos.** How to report, how to debug, how to
> run and watchdog long MPI jobs, how to publish results, and the tooling-hygiene rules
> (no hardcoded values; a constant measured on one dataset is a HYPOTHESIS for the next)
> live in **`ROUKF/docs/agent_practice.md`**, with the rules in `ROUKF/CLAUDE.md` under
> "WORKING PRACTICE". Read them before running anything.

## Project layout

- `nsx/fractionalstep/solver.py` — main solver (`Solver`, `SolverCoupled`, `PETScSolver`)
- `nsx/fractionalstep/problem.py` — variational forms and BC setup (`Problem`, `ProblemCoupled`)
- `nsx/fractionalstep/streamline_diffusion.py` — SUPG/SD stabilization parameters
- `nsx/solver.py` — entry-point dispatcher (fractionalstep or monolithic)
- `tests/` — 2D rectangle regression test with comparison scripts

## Dirichlet BC application strategy (critical)

**Velocity matrices must use rows-only zeroing (`_apply_dbc_rows_to_mat`), not rows+columns.**

Legacy FEniCS `bc.apply(A)` zeros rows only. DOLFINx `zeroRowsColumnsLocal` zeros both rows AND columns. When columns are zeroed, the coupling term `A[j, k_dbc] * u[k_dbc]` disappears from the equation for non-Dirichlet DOF j, requiring `apply_lifting` to correct the RHS. Since `apply_lifting` is not applied in the `same_dbc_boundaries=True` path, columns must NOT be zeroed for velocity matrices.

- `_apply_dbc_rows_to_mat` → `mat.zeroRowsLocal(dofs, diag=1.0)` — use for ALL velocity matrices
- `_apply_dbc_to_mat` → `mat.zeroRowsColumnsLocal(dofs, diag=1.0)` — use ONLY for pressure matrices

Affected locations: `Solver.assemble_tentative_velocity`, `Solver.init_solvers` (Aupd), `SolverCoupled.assemble_tentative_velocity`, `SolverCoupled.solve_tentative_velocity`.

**Why it matters:** With a non-zero inlet BC (e.g. u = -10), zeroing columns without lifting causes a systematic error near the inlet that convects downstream, appearing as "increased velocity at the outlet."

## SUPG stabilization parameter (DG0 element-constant)

Legacy FEniCS `tau_standard` and `tau_shakib` always return a `CompiledExpression` with `element=DG0` — the parameter is **element-constant** (piecewise-constant per cell), evaluated dynamically at cell centroids.

DOLFINx port: when `parameter_element_constant: True`, `tau_standard`/`tau_shakib` project the UFL tau formula to a DG0 `Function` using `fem.Expression(tau_ufl, V_dg0.element.interpolation_points)`. The function is updated each timestep via `SDParameter.update_tau()`, called in `Solver.assemble_tentative_velocity()` before the convection matrix is re-assembled.

`SDParameter._tau_func` / `_tau_expr` store the DG0 function and expression. The problem stores `self._sd_param = sd`; the solver picks it up as `self._sd_param = getattr(problem, '_sd_param', None)`.

Note: `V_dg0.element.interpolation_points` is a **property** (no parentheses) in this DOLFINx version.

## RHS BC application pattern (`solve_tentative_velocity`)

The BC application block must be a single `if/elif/elif/else` chain:

```
if A_robin:   apply rows-only to A_robin + set_bc + set_operator
elif A_fnv:   apply rows-only to A_fnv  + set_bc + set_operator
elif same_dbc_boundaries:  set_bc only (rows already zeroed in assemble step)
else:          copy A, apply rows-only + set_bc + set_operator(copy)
```

Two independent `if` blocks caused a redundant `set_bc` when Robin+same_dbc were both true, and silently overrode the Robin operator with the plain matrix when `same_dbc=False`.

## Pressure RHS apply_lifting

In `build_rhs_pressure`, `apply_lifting` should precede `set_bc` for the pressure Dirichlet BCs:

```python
apply_lifting(bp, [fem_form(self.forms['p']['laplacian'])],
              [self.bc_dict['p']['dirichlet']])
set_bc(bp, self.bc_dict['p']['dirichlet'])
```

For homogeneous `p=0` Dirichlet (the typical outlet BC), this is a no-op. For non-zero outlet pressure it matters.

## DOLFINx vs legacy FEniCS comparison (3D cylinder, CT, P1-P1)

After all fixes, the residual differences are:

| Run | Velocity error | Pressure error |
|-----|---------------|----------------|
| Stokes (no convection, no SUPG) | ~2.6e-14 | ~5e-13 |
| Full NS, no SUPG | ~7.6% | ~17% |
| Full NS, SUPG + DG0 tau | ~7.1% | ~30% |

**Stokes gives machine-precision agreement** — all linear terms (mass, diffusion, pressure Laplacian) assemble identically between FFCx (DOLFINx) and FFC (legacy).

The ~7% velocity / 17% pressure residual with full NS is entirely from the **nonlinear convection term**: FFCx and FFC produce different quadrature point orderings and floating-point accumulation for `rho*dot(u_conv, grad(u))*v*dx`, converging to slightly different steady states. SUPG amplifies this to ~30% pressure because it adds a term proportional to the velocity field difference.

This residual is **irreducible** — it is not a bug.

## Key DOLFINx API notes

- `V.dofmap.index_map_bs` — block size of a vector function space (e.g. 3 for 3D P1 vector)
- `_split_vec_to_lst` / `_merge_lst_to_vec` use `u.x.array[i::bs]` slicing to split/merge components; relies on the blocked DOF ordering of the vector space
- `DirichletBC.dof_indices()[0]` returns the local DOF indices array
- `fem.Expression(ufl_expr, points)` — `points` is `V.element.interpolation_points` (property, not method)
- `mat.zeroRowsLocal(dofs, diag)` — zeros rows, sets diagonal; `mat.zeroRowsColumnsLocal(dofs, diag)` — also zeros columns

## Windkessel distal pressure `P_d` (added 2026-10-28, portalvein D0)
Optional BC parameter `P_d` (default 0): a constant pressure at the far end of R_d,
C dπ/dt = Q − (π − P_d)/R_d, steady state π = R_d·Q + P_d (e.g. hepatic-vein pressure).
π, p0 and Pl stay ABSOLUTE. Backward-Euler step: π ← απ + βQ + (1−α)P_d. Implicit operator:
Pl = γQ + απ + (1−α)P_d, i.e. `rhs_p` gains `delta_d*P_d` with delta_d = (1−α)/γ
(`_wk_update_deltas`: 0 on a restart step with C>0, where Pl = R_p·Q + π already holds
P_d; 1/γ when C=0). Initial seed Pl(0) = α·p0 + (1−α)P_d. With P_d = 0 every added term
is zero. Verified: D0 rigid run with P_d = 4 mmHg equals the P_d = 0 run shifted by P_d
(to 1e-12 mmHg). Files: `fractionalstep/problem.py::_windkessel`,
`fractionalstep/solver.py::solve_windkessel`, `::_wk_update_deltas`.

**Time-dependent `P_d(t)`** (added 2026-10-29): `P_d` may be a waveform string in `t`, with named
constants in `P_d_parameters`, e.g.
`P_d: "Pd0 + dPd*min(max(t - t0, 0.0)/Tr, 1.0)"`, `P_d_parameters: {Pd0: 5332.88, dPd: 5332.88, t0: 1, Tr: 2}`.
It is compiled with `_make_time_expression` and stored in `bc_dict['p']['windkessel']['P_d_func'][bid]`.
`solve_windkessel` sets `prm['P_d'].value = f(self.t)` at the new time before each update (backward
Euler; `rhs_p` holds the same Constant). p0/Pl(0) use P_d(0). A numeric `P_d` takes the old path
unchanged. Verified on the D0 rigid mesh (2 ranks): `"Pd0"` vs numeric equal to 5e-11 dyn/cm²; a +4 mmHg
ramp over 50 ms (τ = 0.02 s) follows the analytic first-order lag (2.540 vs 2.531 mmHg mid-ramp,
3.988 at 5τ), with outlet flows unchanged. Cases in `portalvein/D0/windkessel/results/tPd_*`.

**Compliance reference `P_ext(t)`** (added 2026-10-31, portalvein D0 breathing): optional BC parameter
`P_ext` (string in `t`, constants in `P_ext_parameters`; absent = grounded capacitor, unchanged). The
capacitor sees the pressure outside it, e.g. the abdominal pressure on the hepatic bed:
C d(π − P_ext)/dt = Q − (π − P_d)/R_d. This is exactly the grounded form with
P_d_eff = P_d + R_d·C·dP_ext/dt, which `solve_windkessel` applies (backward difference over one step,
built from the un-shifted P_d: `P_d_func` or `prm['P_d_base']`, so the two calls per step do not
accumulate). Only dP_ext/dt enters, so a constant offset has no effect. Example:
`P_ext: "-dPa*0.5*(1 - cos(2*pi*max(t - t0, 0.0)/Tr))"`, `P_ext_parameters: {dPa: 3999.66, t0: 2.5, Tr: 4}`.
Verified on the D0 rigid mesh (8 ranks): with P_d following the same swing (f = 1), π − P_ext is constant
to 0.01 mmHg for a 8 mmHg swing (grounded C: 6.4 mmHg lagging 0.5 s). Case `D0/windkessel/results/resp_UE`.
## `coronary` outlet BC — Kim et al. 2010 bed (added 2026-11-03, angio)

`type: 'coronary'`: p --R_a-- P1 (C_a to ground) --R_am-- P2 (C_im to P_im) --R_v-- P_v, with
P_im = `lambda`*P_LV read every step from the `heart` BC (`heart_id`, default the only one; `lambda: 0`
needs no heart). Algebra in `nsx/fractionalstep/coronary.py` (pure python, shared with the angio 0D
pre-flight): backward Euler gives p = gamma*Q + b, condensed into the projection as a WK entry with
`kind: 'coronary'` (alpha = 1, beta = gamma; `_wk_coefficients`). `solve_coronary` sets b (as `pi`)
after `solve_heart`; `sync_coronary` reads Pl and updates P1, P2. Needs windkessel implicit+condensed.
Verified: steady state p = P_v + Q(R_a+R_am+R_v) exactly; condensed form = update to round-off.
State (P1, P2, P_im) rides in checkpoints via the `wk` scalar group; ROUKF restart steps refused.

## `heart` inlet BC — 0D left heart (added 2026-11-03, angio)

> **`coupling: 'condensed'` is the default and the only mass-conserving one (2026-11-03).**
> The node `C_ao` is registered as a condensed implicit Windkessel with R_p = 0, R_d -> inf
> (a pure capacitor, `_wk_coefficients` in solver.py): `solve_heart` sets its reservoir to
> `P_ao^n + V_AV/C_ao` before the projection, `sync_heart` copies back `P_ao = Pl` and
> `Q_3D = C_ao (pi_pre - Pl)/dt`. Needs `windkessel: {implicit: True, condensed: True}`.
> `coupling: 'dirichlet'` (the first version) imposes P_ao as a pressure Dirichlet; the
> projection then never tests div u against a q nonzero on the inlet and the outlets
> received **+35 % more volume per cycle than the heart ejected** (angio heart_wk,
> MEASURED), peak P_ao +14 mmHg above the 0D. Condensed: heart Q_3D volume 58.59 mL vs
> outlet volume 58.58 mL over 0-0.25 s, peak P_ao 113.8 vs 112.1 mmHg in the 0D.
> OPEN: the SAVED velocity's inlet-face flux still reads ~12 % below the outflow in
> systole, with the compensating div u in the first inlet cell layer; zeroing of the rim
> nodes by `DC_in_update` explains at most ~40 % of it (measured). Cause unknown.

## Restart contract: heart + coronary (MEASURED 2026-11-06, angio, 4 ranks)
- A checkpoint carries the full 0D state: `heart_*` attrs (V, Q_AV, Q_MV, P_ao, P_LV, E, Q) and
  `wk_*` for each bed (P1, P2, P_im, ...). The plain floats NOT saved (`b`, `dP_im`, `P_im_new`,
  heart `pi_pre`) are rewritten in `solve_coronary`/`solve_heart` before they are read.
- Test: a continuous run from t = 0.4 vs a restart from its own 0.41 checkpoint, 20 steps.
  heart CSV identical, u 8e-13, p 8e-14.
- **FIXED, NON-ALE CT: stale velocity ghosts (`solver.py`, velocity update).**
  - Defect: `u_i.x.petsc_vec.axpy(1, du)` updates OWNED dofs only, with no `scatter_forward`. The
    ghosts kept u*, the next step convected with it, and u0_lst copied it into the history.
  - A restart scatters the ghosts, so before the fix restart and continuous runs differed
    (Q_3D 9e-7, u 5e-4). Every differing dof was a ghost (found by snapshotting all Functions).
  - Effect of the fix on the continuous run, 20 steps: u L2 9e-4, max 1.3e-2 at 6 nodes
    (partition interfaces in fast flow), p 5e-5, Q_3D 5e-6, P_ao 4e-8.
  - So every multi-rank rigid (non-ALE) run before 2026-11-06 carries this partition-dependent
    LOCAL velocity error; global flows and pressures are unaffected at the 1e-5 level.
  - The ALE branch already scattered.
- Minor:
  - a restart into a NEW write_path appends `heart_<id>.csv` without a header;
  - checkpoint dirs are numbered from the restart step (0, 20, ...), not globally;
  - the "Restored Windkessel state bid 3: rt_..." log line is the heart matched by suffix
    (cosmetic).
