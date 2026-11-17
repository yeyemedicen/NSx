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

## Consistent SUPG (added 2026-11-13, angio kim5; LOCAL, uncommitted)

Input: `fem: stabilization: streamline_diffusion: {enabled: true, consistent: true, parameter: shakib, ...}`.
Without `consistent` (default) the term is the MINIMAL residual `tau (u.grad v)(rho u.grad u)` = streamline
diffusion, as before (kim3 smoke over 20 steps reproduces results_kim3/heart_3.csv BIT-FOR-BIT after this change).
With `consistent: true` (split solver `Problem`/`Solver` only; `ProblemCoupled` still raises) the weight
`tau (u.grad v)` multiplies the FULL momentum residual of component i:
- time: `supg_time = tau (u.grad v) rho/dt u_i` -> LHS `cf*S`, and the RHS matrix that multiplies the
  old-velocity vector becomes `mass + S` (rebuilt from mass each step; S assembled fresh, own sparsity);
- convection (+ `-mu lap u` for P2+): `supg_convdiff`, unchanged;
- pressure: `supg_gradp[i] = -tau (u.grad v) dp/dx_i` with the CURRENT `self.p` (= p^n when the tentative step
  runs; under CT `phi is p`, ghosts updated by the pressure KSP), linear forms compiled once
  (`Solver._supg_gradp_c`) and assembled per component per step in `build_rhs_tentative_velocity`.
  Consistent up to the lagged pressure, i.e. the splitting error O(dt). P1: the viscous part of the residual is 0
  elementwise (standard P1 SUPG caveat).
- tau should carry the time scale: `parameter: shakib` (1/sqrt((2/dt)^2 + u.G.u + Cinv (nu)^2 G:G)); a warning is
  logged for `standard` (no dt term, tau/dt can be >> 1 at low speed).
- The old `assert True, 'supg_time should not be useable!!'` block (which assembled S INTO the rhs matrix and
  added only S, not cf*S, to A) is replaced in the split solver; the coupled one is unreachable (raises first).
Smoke (angio coronary_v2, heart + 23 Kim beds, 4 ranks, 20 steps): runs, ~20 % slower per step than kim3; u at
t = 0.01 differs 4.1 % rel L2 from kim3 (max |u| 22.7 vs 21.8 cm/s), P_ao/P_LV identical to 1e-5.
**MEMORY LEAK, FIXED 2026-11-13 (MEASURED).** The first version allocated a NEW S (`_assemble_mat` with no mat), a
NEW rhs (`mass.copy()`) and a NEW gradp vector every step: +7 MB/step/rank footprint on coronary_v2 (4 ranks,
`top` MEM; RSS hides it, macOS compresses). kim5 reached ~9 GB/rank at step 1400 and panicked an 8 GB Mac (watchdog,
compressor full). Each allocation alone does NOT leak in a unit-cube test (1 and 2 ranks), so which one leaked is not
known. Fix: `_supg_time_c`/`_supg_S`/`_supg_rhs`/`_supg_gradp_vec` are persistent, re-assembled in place (rhs =
mass copied with SUBSET pattern + S). Footprint flat over 60 steps (~450 MB/rank, kim3 ~430); heart CSV
byte-identical to the leaking version. Rule: never allocate a PETSc Mat/Vec per step; keep it and assemble in place.

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

## `nsx/transport` — passive scalar (contrast) transport (added 2026-11-11, angio; LOCAL, uncommitted)
- `ScalarTransport(mesh, facet_tags, options)` (`scalar.py`): dc/dt + u.grad c - div(D grad c) = s
  (`form: advective`, default) or the conservative form; P1, theta scheme, SUPG. It owns NO velocity:
  the caller sets `self.u` (P1 vector, t^{n+theta}) before `step(dt)`, so the same class serves the
  OFFLINE driver and, later, an in-solver sub-step.
- Inflow imposed WEAKLY and UPWIND on open ids (acts only where u.n < 0, so a closed valve imports
  nothing); outlets natural or `backflow: zero`; wall zero-flux. `sources`: Gaussian (or ball)
  catheter tip, solute without volume.
- `balance()` + the driver's `balance.csv`: M, solute flux per tag (as the scheme exchanges it),
  int c div u, int s, residual. The residual is ~1e-10: the bookkeeping is exact, so any mass
  change from a non-solenoidal u shows up in `c_div_u`, not hidden.
- `StoredVelocity` (`velocity.py`): a run's u.h5, nodes matched by coordinates (KD-tree, any rank
  count), LINEAR interpolation in time (default whatever dt), `replay: {t0, T}` replays one stored cycle.
- Offline: `python -m nsx.transport.offline config.yaml` (doc in the module). Writes c.xdmf, balance.csv.
- **CT velocity (2026-11-14, angio)**: `velocity: ct_correction: {pressure: p.h5, dt, rho, dirichlet: [1]}`
  advects with w = u* - (dt/rho) grad p (P1 + P0), the field CT makes weakly divergence-free.
  `CTVelocity` (`velocity.py`) rebuilds u* from the SAVED UPDATE u and p with no flow re-run:
  M_D du = -(dt/rho)(grad p, v) (consistent P1 mass, du = 0 on velocity-Dirichlet facets = what
  `solve_velocity_update` does with `DC_in_update: true`), u* = u - du (CG/jacobi, 27 its).
  MEASURED vs a run that saved u* (angio kim4, 4 ranks): max err 1.1e-11. The saved u is the
  L2 projection of w on nodal P1: at a pressure-driven (condensed) inlet the inflow lives in the
  P0 grad p of the first cell layer, which nodal u cannot hold (inlet flux ~40 % low, alternating
  div u in the first layers). `ScalarTransport` option `pressure_correction: k` (set by the driver)
  uses `self.vel = self.u - k grad(self.p)`; advective form only. Its `balance()` books the
  boundary exchange NODALLY (d_j = int w.grad phi_j; flux of tag t = sum_{j in t} c_j d_j + the weak
  inflow term), c_div_u = creation on non-open nodes. MEASURED angio kim4 LCA, 0.4 s: creation
  1e-14 (saved u: +0.27 % of the injection), residual 2e-9. The generic c_div_u is now written
  int_dOmega c u.n - int u.grad c (identical for continuous u; channel verification unchanged).
- **Inline / two-way contrast (2026-11-15, angio; `nsx/transport/inline.py`)**: options block `contrast:`
  (JellyFSI: under `fluid:`; doc in the module). `Solver.init` builds `InlineContrast` after
  read_checkpoint; `Solver.timestep` calls `solve_contrast()` after the velocity update (JellyFSI FSI
  paths do NOT call it yet; fixed mesh + CT + DC_in_update only). Advects with w^{n+1/2} = mean of
  u* - (dt/rho) grad p of the two steps (u* = u - du in memory). Two-way: `injection: volume: true` ->
  `build_rhs_pressure` adds rho/dt (s_v, q) (div w = s_v weakly) and the transport gets the dilution
  -c s_v with the SAME s_v (`ScalarTransport` sources `volume`/`c`; source rate = mean of the two
  projections' rates, first step = the new rate). `mixture:` (problem.py form_velocity_tentative):
  mu(c) log mixing (viscous matrix refreshed every step via `_turb_active`), Boussinesq body force
  (rho_c - rho_b) c g in `build_rhs_tentative_velocity` (forms['u']['contrast_body'], None per zero
  g component); c lagged one step (`problem.contrast_c`), clipped to [0, 1]. Output
  <write_path>/contrast/{c.xdmf,balance.csv}; checkpoint <ckpt>/c.h5 (by coordinates; a flow-only
  checkpoint starts c = 0; `t0` defaults to the first step's flow time minus dt).
  Catheter jet (2026-11-16): `injection: momentum: {enabled, direction, catheter_diameter}` adds
  rho s_v (u_inj - c_u u^n) to the tentative RHS (`InlineContrast.momentum_source`, hook in
  build_rhs_tentative_velocity), u_inj = q(t)/A along `direction`, c_u = 1/2 with skew convection
  (cancels the +rho/2 (div u) u* the skew form adds when div u = s_v), 1 otherwise; rate at t^{n+1}.
  MEASURED (angio kim4 restarted at 1.6 s, 4 ranks, `cfd/heart_wk/inj_smoke/`): one-way inline =
  offline at the same dt (c_max 0.07110 vs 0.07108 at 5 ms; u* in memory = rebuilt to 3e-12);
  creation 1e-15 per step; volume on: sum of weak open-boundary fluxes 0.9998 vs Q_inj 1 (0.0002
  without); restart from a mid-run checkpoint bit-identical (contrast + heart); 2.1-2.5 s/step.
- Verification `python -m nsx.transport.verify_channel` (offline path end to end; box, u = U0(1+a t)
  written to u.h5 every 0.05 s, read back): MEASURED 2026-11-11, h 0.025, dt 5 ms, D 1e-3:
  front 1.2492 vs 1.2500, RMS vs erfc 7.0e-3, c in [-0.032, 1.035], mass exact, residual 3e-11;
  advective = conservative (div u = 0). With `shock_capturing: 0.2`: monotone [0, 1], front +4.9e-3,
  RMS 4.4e-2. C = 0.7 smears the front to ~0.5 (unusable); default is 0 (off).
- Lessons (MEASURED on coronary_v2): (1) cap the quadrature (`quadrature_degree: 2` default) --
  UFL's estimate for tau's sqrt/powers made it 7.3 s/step, 0.69 with the cap, identical results;
  (2) the flux forms must use the GLOBAL tag set (a rank may miss a tag; mismatched reductions);
  (3) the shock-capturing residual must include the (lagged) time derivative, or it diffuses every
  moving front; (4) a step c = 1 at the valve-plane inlet in diastole overshoots (c 1.6 / -0.2) --
  inject at a catheter tip instead.
