# NSx — Navier-Stokes DOLFINx Solver

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
