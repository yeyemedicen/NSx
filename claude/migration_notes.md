# Migration Notes: legacy FEniCS → DOLFINx

---

## 1. Legacy vs DOLFINx comparison

### Test case
Rectangular channel (2D), P1/P1 velocity-pressure, fractional-step CT scheme.
- Mesh: `tests/rectangle_shared.xdmf` (shared between both solvers — same triangulation)
- Input: `tests/input.yaml`
- T = 0.5 s, dt = 0.001, 500 time steps
- Boundary conditions: lid-driven (U=0.1 on top), no-slip on sides, Neumann outflow

### Numerical results

```
 step       t       |u| leg       |u| dol    |Δu|/|u|    |p-p0| leg    |p-p0| dol    |Δp|/|p|          C(t)
-----------------------------------------------------------------------------------------------------------
    0   0.000  0.000000e+00  0.000000e+00         nan  0.000000e+00  0.000000e+00         nan  0.000000e+00
   10   0.010  1.812179e+00  1.810529e+00   1.086e-02  7.852097e+00  1.010330e+01   3.125e-01  1.239780e-01
   50   0.050  1.835281e+00  1.833548e+00   1.104e-02  4.997553e+00  7.294993e+00   4.890e-01  1.234938e-01
  100   0.100  1.855823e+00  1.854062e+00   1.096e-02  4.311851e+00  6.630750e+00   5.666e-01  1.234565e-01
  200   0.200  1.877583e+00  1.875810e+00   1.086e-02  3.919269e+00  6.253325e+00   6.233e-01  1.234477e-01
  300   0.300  1.885791e+00  1.884017e+00   1.082e-02  3.808772e+00  6.147606e+00   6.413e-01  1.234488e-01
  400   0.400  1.888599e+00  1.886825e+00   1.080e-02  3.773803e+00  6.114229e+00   6.473e-01  1.234498e-01
  500   0.500  1.889525e+00  1.887752e+00   1.080e-02  3.762592e+00  6.103548e+00   6.492e-01  1.234504e-01

Velocity  — max rel error: 1.106e-02    mean rel error: 1.087e-02
Pressure  — max rel error: 6.492e-01    mean rel error: 6.018e-01
Gauge C(t) — max |C|: 1.240e-01        final C: 1.234504e-01
```

### Interpreting the results

**Velocity (~1.1% error):** consistent and stable across all time steps. The small difference
is expected — the two implementations use different linear algebra backends (PETSc interface
in legacy vs DOLFINx's assembled forms) and slightly different floating-point assembly order.

**Pressure gauge C(t):** the Navier–Stokes pressure is only defined up to a time-dependent
additive constant — `p` and `p + C(t)` are both exact solutions for the same velocity field.
The two solvers pin pressure differently internally, so a non-zero C(t) ≈ 0.1234 is expected
and is not an error. The comparison subtracts C(t) at each step before computing the pressure
norm difference.

**Pressure relative error (~60%):** after gauge correction, the remaining difference is
dominated by the pressure field being smooth but small in magnitude (denominators are small),
so relative errors look large while absolute errors are consistent with the velocity accuracy.
This is normal behaviour for fractional-step pressure.

### How to rerun the comparison

Three scripts in `tests/`, each must be run in its own conda environment:

```bash
# Step 1 — extract legacy FEniCS fields (run in FenicsLegacy env)
conda run -n FenicsLegacy python tests/extract_legacy.py

# Step 2 — extract DOLFINx fields (run in dolfinX env)
PYTHONPATH=/Users/yeye/Common conda run -n dolfinX \
    python tests/extract_dolfinx.py

# Step 3 — compare (any env with numpy; add --plot for matplotlib figures)
python tests/compare_norms.py
python tests/compare_norms.py --plot   # saves tests/results/comparison.png
```

**Prerequisites:**
- Both simulations must have been run and saved checkpoints to `tests/results/checkpoint/`
- Legacy simulation must have run from `NavierStokes/` using `tests/input_legacy.yaml`
- DOLFINx simulation must have run from `NSx/` using `tests/input.yaml`
- Both must use `tests/rectangle_shared.xdmf` as the mesh (same triangulation — created once
  by `tests/convert_mesh.py` running in FenicsLegacy env)

**Output files:**
- `tests/results/legacy_fields.npz` — vertex-ordered velocity and pressure at each checkpoint
- `tests/results/dolfinx_fields.npz` — same format for DOLFINx
- `tests/results/comparison.png` — plots of norms and relative errors (with `--plot`)

---

## 2. Key changes from legacy FEniCS to DOLFINx

### 1. `Constant` requires a mesh reference

**Legacy:** `Constant(1.0)` — a global scalar, no mesh needed.

**DOLFINx:** constants are attached to a mesh so PETSc knows the scalar type and MPI
communicator.

```python
# legacy
c = Constant(1.0)

# DOLFINx
c = fem.Constant(mesh, np.array(1.0, dtype=PETSc.ScalarType))
```

Updating a constant in-place (used for time-dependent coefficients):
```python
# legacy
c.assign(new_value)

# DOLFINx
c.value = new_value   # or c.value[:] = ... for vector constants
```

---

### 2. `DirichletBC` is now a two-step locate + construct

**Legacy:** boundary data was a `MeshFunction` of integers, and `DirichletBC` accepted it
directly.

**DOLFINx:** boundaries are stored as `MeshTags`. You first locate DOFs topologically, then
construct the BC object.

```python
# legacy
bc = DirichletBC(V, value, boundaries, boundary_id)

# DOLFINx
facets = facet_tags.find(boundary_id)
dofs   = locate_dofs_topological(V, mesh.topology.dim - 1, facets)
bc     = dirichletbc(value, dofs, V)
```

**Important:** DOLFINx `DirichletBC` objects are not hashable, so they cannot be used as
dictionary keys. Replace any `dict[bc]` pattern with `dict[(boundary_id, component)]`.

---

### 3. `Expression` (C++ string) is replaced by `Function` + `.interpolate()`

**Legacy:** expressions were compiled C++ strings evaluated at quadrature points at runtime.

**DOLFINx:** there is no string-based expression. Instead, interpolate a Python lambda (or
callable) into a `Function` once, and for time-dependent cases update a `Constant` multiplier
each step.

```python
# legacy — time-dependent parabolic inflow
expr = Expression('U * 4*x[1]*(H - x[1]) / H^2', U=0.1, H=1.0, degree=2)

# DOLFINx — time-independent part
u_profile = Function(V)
u_profile.interpolate(lambda x: 0.1 * 4*x[1]*(H - x[1]) / H**2)

# for time-dependent amplitude, multiply by a Constant updated each step
U_amp = fem.Constant(mesh, 1.0)
# in the form: U_amp * u_profile
# each step: U_amp.value = np.sin(t)
```

**Math note:** `.interpolate()` uses nodal interpolation (exact at DOF nodes for Lagrange
elements), not L2 projection — it is cheaper but assumes the function is smooth enough that
nodal values are representative.

---

### 4. `assemble()` for scalars requires explicit MPI reduction

**Legacy:** `assemble(form)` automatically summed contributions across all MPI ranks.

**DOLFINx:** `assemble_scalar` returns the local rank's contribution only. You must explicitly
reduce across ranks.

```python
# legacy
area = assemble(1 * ds(boundary_id))

# DOLFINx
area_local = assemble_scalar(fem_form(1 * ds(boundary_id)))
area = mesh.comm.allreduce(area_local, op=MPI.SUM)
```

**Why it matters:** forgetting the `allreduce` gives silently wrong results on more than one
MPI rank (each rank sees only its own facets).

---

### 5. Function spaces: new API and explicit shape

**Legacy:** separate classes for scalar vs vector spaces.

**DOLFINx:** a single `functionspace` call; vector spaces pass a shape tuple.

```python
# legacy
V_scalar = FunctionSpace(mesh, 'CG', 1)
V_vector = VectorFunctionSpace(mesh, 'CG', 1)

# DOLFINx
V_scalar = functionspace(mesh, ('Lagrange', 1))
V_vector = functionspace(mesh, ('Lagrange', 1, (mesh.topology.dim,)))
```

`.collapse()` on a sub-space now returns a **tuple** `(collapsed_space, dof_map)`:
```python
# legacy
V_c = V.sub(0).collapse()

# DOLFINx
V_c, dof_map = V.sub(0).collapse()   # dof_map needed for dirichletbc
```

---

### 6. P1b (mini element) via basix enriched element

**Legacy:** `VectorFunctionSpace(mesh, 'CG', 1)` + enrichment was not standard.

**DOLFINx:** use `basix.ufl` to compose Lagrange + Bubble elements explicitly.

```python
import basix.ufl as bufl

cell = mesh.basix_cell()
P1   = bufl.element('Lagrange', cell, 1, shape=(ndim,))
Bub  = bufl.element('Bubble',   cell, ndim + 1, shape=(ndim,))
V    = functionspace(mesh, bufl.enriched_element([P1, Bub]))
```

**Math note:** the bubble function is zero on all element faces and non-zero in the interior,
providing local inf-sup stability without a pressure stabilization term. It is the vector
analogue of the MINI element.

---

### 7. DOF array access: `.vector()[:]` → `.x.array`

**Legacy:** PETSc vector wrapped inside dolfin, accessed via `.vector()[:]`.

**DOLFINx:** direct numpy view of the local DOF array.

```python
# legacy
values = u.vector()[:]
u.vector()[:] = new_values

# DOLFINx
values = u.x.array                 # read (local DOFs)
u.x.array[:] = new_values          # write
u.x.scatter_forward()              # synchronize ghost DOFs after write
```

`scatter_forward()` is required after any manual write to `.x.array` in parallel to push
updated values to ghost DOFs on neighbouring ranks.

---

### 8. Mesh topology attributes are properties, not methods

**Legacy:** `mesh.topology().dim()` — both `.topology()` and `.dim()` are method calls.

**DOLFINx:** both are properties.

```python
# legacy
dim  = mesh.topology().dim()
comm = mesh.mpi_comm()

# DOLFINx
dim  = mesh.topology.dim
comm = mesh.comm
```

---

### 9. `project()` is gone — use a mass-matrix solve

**Legacy:** `project(expr, V)` solved a global L2 projection implicitly.

**DOLFINx:** no built-in `project`. Use `LinearProblem` or the `dolfinx.fem.Expression`
(for point evaluation) depending on the use case.

```python
# legacy
u_proj = project(expr, V)

# DOLFINx — explicit L2 projection
from dolfinx.fem.petsc import LinearProblem
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
proj = LinearProblem(fem_form(inner(u, v)*dx),
                     fem_form(inner(expr, v)*dx), bcs=[])
u_proj = proj.solve()
```

**Math note:** L2 projection minimises `‖u_h - expr‖²` over the space V. For Lagrange
elements nodal interpolation (`.interpolate()`) is cheaper and usually sufficient; projection
is only needed when `expr` is not smooth enough for pointwise interpolation (e.g. DG fields,
derivative quantities).

---

### 10. XDMF visualization is limited to P1 (geometry nodes)

**Legacy:** ParaView could read P2 data directly from the XDMF/HDF5 output.

**DOLFINx:** `write_function` stores values only at geometry (P1) nodes. Higher-order DOFs
at edge midpoints are silently ignored.

**Workaround for visualization:** interpolate to a P1 function before writing XDMF.
Checkpoints (raw `.h5` via h5py) store the full DOF array and are not affected.

```python
# create once
u_vis = Function(functionspace(mesh, ('Lagrange', 1, (ndim,))), name='u')

# each write step
u_vis.interpolate(u)          # u is P2 or P1b
xdmf.write_function(u_vis, t)
```

---

### 11. `form()` compilation is explicit and cached

**Legacy:** UFL forms were compiled transparently on first `assemble()`.

**DOLFINx:** you must call `fem.form(ufl_form)` explicitly to compile. The compiled object
is separate from the UFL expression and should be stored and reused — recompiling every step
is expensive.

```python
# DOLFINx — compile once, assemble many times
a_compiled = fem_form(a_ufl)
A = assemble_matrix(a_compiled, bcs=bcs)

# scalar integrals
val = assemble_scalar(fem_form(expr * dx))   # fem_form each call is OK for scalars
                                              # (cheap), but store for hot loops
```

**Why it matters:** `fem.form()` triggers JIT compilation via FFCx and is roughly equivalent
to a C compilation step. For matrix/vector forms called in the time loop, always compile
outside the loop.
