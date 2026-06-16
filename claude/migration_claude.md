# Migration Plan: `problem.py` — legacy FEniCS → DOLFINx

## Context

`problem.py` is the variational-form / BC setup module for the fractional-step Navier-Stokes solver. It was written for FEniCS 2019 (dolfin). The goal is to port it to DOLFINx (≥ 0.7) while preserving all class structure (Problem, BoundaryConditions, ProblemCoupled, BoundaryConditionsCoupled) and all physics (ALE, MAPDD, Windkessel, Robin BCs, SUPG stabilization, etc.).

## Critical files

| File | Role |
|------|------|
| `/Users/yeye/NSx/nsx/fractionalstep/problem.py` | **Primary target** |
| `/Users/yeye/Common/common/inout.py` | Mesh & HDF5 I/O — needs parallel updates (separate repo) |
| `/Users/yeye/Common/common/utils.py` | `is_Expression`, `is_Constant`, `is_enriched` helpers |
| `/Users/yeye/NSx/nsx/fractionalstep/streamline_diffusion.py` | Out of scope here; imports `from dolfin import *` |

---

## Step 1 — Imports  *(mechanical)*

Replace the three legacy import lines (14–19) with:

```python
from mpi4py import MPI
import dolfinx
import dolfinx.fem as fem
from dolfinx.fem import (
    Function, functionspace, dirichletbc,
    locate_dofs_topological, locate_dofs_geometrical,
    form as fem_form, assemble_scalar,
)
import basix
import basix.ufl as bufl
import ufl
from ufl import (
    TrialFunction, TestFunction,
    as_vector, inner, sym, grad, dx, dot, div,
    FacetNormal, Measure, Identity, inv, det, sqrt, CellVolume,
)
from petsc4py import PETSc
from common import inout, utils
```

Drop `TensorFunctionSpace`, `project`, `Expression`, `DirichletBC` from imports (handled below).

---

## Step 2 — MPI  *(mechanical, 5 sites)*

| Legacy | DOLFINx |
|--------|---------|
| `MPI.rank(MPI.comm_world)` | `MPI.COMM_WORLD.rank` |
| `MPI.size(MPI.comm_world)` | `MPI.COMM_WORLD.size` |
| `MPI.barrier(MPI.comm_world)` | `MPI.COMM_WORLD.Barrier()` |

Update the `rank0` decorator at line 26 to use `MPI.COMM_WORLD.rank == 0`.

---

## Step 3 — Version check  *(mechanical, 1 site)*

```python
# line 97
if dolfinx.__version__ < '0.7':
    raise Exception('DOLFINx version 0.7 or higher required!')
```

---

## Step 4 — Mesh attribute access  *(mechanical)*

| Legacy | DOLFINx |
|--------|---------|
| `mesh.topology().dim()` | `mesh.topology.dim` |
| `V.mesh().topology().dim()` | `self.mesh.topology.dim` |
| `FacetNormal(V.mesh())` | `FacetNormal(self.mesh)` |
| `mesh.mpi_comm()` | `mesh.comm` |

~20 sites total (mesh dim at lines 178, 855, 1923, 2346; FacetNormal at ~15 BC method sites).

---

## Step 5 — `Constant`  *(logic change — needs mesh ref, ~35 sites)*

Add a private helper to **both** `Problem` and `BoundaryConditions` (after `self.mesh` is set):

```python
def _C(self, val):
    import numpy as np
    return fem.Constant(self.mesh, np.array(val, dtype=PETSc.ScalarType))
```

Replace every `Constant(x)` → `self._C(x)`.

The one `Constant.assign(val)` call (line ~1438) becomes:
```python
prm['pi'].value = pi0    # in-place mutation
```

---

## Step 6 — Function spaces  *(mostly mechanical, P1b is logic)*

### 6a. Standard spaces
```python
FunctionSpace(mesh, 'CG', 1)          → functionspace(mesh, ("Lagrange", 1))
VectorFunctionSpace(mesh, 'CG', deg)   → functionspace(mesh, ("Lagrange", deg, (ndim,)))
FunctionSpace(mesh, 'DG', 0)           → functionspace(mesh, ("DG", 0))
FunctionSpace(mesh, 'DG', 1)           → functionspace(mesh, ("DG", 1))
```

### 6b. P1b enriched element (lines 269–272, 2006–2008)
```python
cell = mesh.basix_cell()
P1  = bufl.element("Lagrange", cell, deg)
B   = bufl.element("Bubble",   cell, deg + self.ndim)
self.Vi = functionspace(mesh, bufl.enriched_element([P1, B]))
self.V  = functionspace(mesh, bufl.enriched_element([
    bufl.element("Lagrange", cell, deg, shape=(self.ndim,)),
    bufl.element("Bubble",   cell, deg + self.ndim, shape=(self.ndim,)),
]))
```

### 6c. DOF count helper (6 sites)
```python
def _dof_count(V):
    return V.dofmap.index_map.size_global * V.dofmap.index_map_bs
# replace V.dim() → _dof_count(V)
```

### 6d. Sub-space queries (mechanical)
```python
V.num_sub_spaces()  →  V.num_sub_elements
V.ufl_element().degree()  →  V.ufl_element.degree
```

### 6e. `.collapse()` return-value change
```python
# before: V_c = V.sub(i).collapse()
# after:
V_c, dof_map = V.sub(i).collapse()   # dof_map needed for dirichletbc
```

---

## Step 7 — `DirichletBC`  *(logic change, ~15 sites — most complex)*

The legacy pattern `DirichletBC(V, val, bnds, id)` becomes a two-step locate + construct.

**Requires**: `inout.read_mesh` returns `(mesh, subdomains, facet_tags)` where `facet_tags` is a `dolfinx.mesh.MeshTags` object (see Step 12).

```python
# Store facet_tags on self:
self.mesh, self.subdomains, self.facet_tags = inout.read_mesh(...)
self.bnds = self.facet_tags   # preserve name used downstream

# Component-wise sub-space BC (most common pattern):
facets    = self.facet_tags.find(bc['id'])
V_sub, _  = self.V.sub(i).collapse()
dofs      = locate_dofs_topological((self.V.sub(i), V_sub),
                                     self.mesh.topology.dim - 1, facets)
dbc       = dirichletbc(self._C(scalar_val), dofs, self.V.sub(i))

# Scalar space BC (pressure, scalar Vi):
dofs = locate_dofs_topological(self.Q, self.mesh.topology.dim - 1, facets)
dbc  = dirichletbc(self._C(val), dofs, self.Q)

# Function-valued BC (inflow profile, external ALE displacement):
dbc = dirichletbc(func, dofs)   # no space arg — inferred from func
```

**Point pressure BC** (`method='pointwise'`, line 1927):
```python
dofs = locate_dofs_geometrical(self.Q, lambda x: np.isclose(x[0], px) & ...)
dbc  = dirichletbc(self._C(0.0), dofs, self.Q)
```

**bc_dict key change**: DOLFINx `DirichletBC` objects are not hashable — replace `bc_dict['u']['dbc_expressions'][dbc]` with `bc_dict['u']['dbc_expressions'][(bc['id'], i)]`. Update all read sites in `solver.py` accordingly (mark as interface change).

---

## Step 8 — `Expression`  *(logic change, ~12 sites)*

String-based C++ `Expression` is gone. Replacement strategy per usage type:

| Usage | Replacement |
|-------|-------------|
| Time-independent spatial profile (parabola) | `fem.Function(Vi)` + `.interpolate(lambda x: ...)` |
| Time-dependent (sine-parabola, waveform) | spatial `fem.Function` × scalar `fem.Constant` (updated by solver each step) |
| Windkessel explicit pressure | `fem.Constant` stored in `prm['Pl_const']`; solver updates `.value` |
| Displacement / velocity string Dirichlet | `fem.Function` + `.interpolate(lambda x: numpy_eval(...))` |

For time-dependent cases store a `time_state = {'t': 0.0}` dict in `bc_dict` so the solver can write `time_state['t'] = t` and call `func.interpolate(...)`.

`isinstance(val, Expression)` checks → `isinstance(val, fem.Function)`.
`dolfin.function.expression.Expression` type refs → `fem.Function`.

---

## Step 9 — `project`  *(logic change, 2 sites)*

Add module-level helper:
```python
def _project(expr, V):
    from dolfinx.fem.petsc import LinearProblem
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    problem = LinearProblem(fem_form(inner(u, v)*dx),
                            fem_form(inner(expr, v)*dx), bcs=[])
    return problem.solve()
```

Lines 2514, 2553: `project(val, V.collapse())` → `_project(val, V_collapsed)` (with `.collapse()` returning a tuple — take first element).

---

## Step 10 — `assemble(scalar_form)`  *(mechanical + MPI allreduce, ~10 sites)*

```python
# before
area = assemble(elem * self.ds(bc['id']))

# after
area_loc = assemble_scalar(fem_form(elem * self.ds(bc['id'])))
area = self.mesh.comm.allreduce(area_loc, op=MPI.SUM)
```

Sites: lines 1253, 1254, 1263, 1278, 1279, 1399, 2607, 2608, 2617, 2632, 2633.

---

## Step 11 — `Measure` / `ds`  *(mechanical, 3 explicit sites)*

```python
# before
ds = Measure('ds', domain=self.mesh, subdomain_data=self.bnds)
# after (identical syntax; self.bnds is now a MeshTags object from Step 7)
ds = ufl.Measure('ds', domain=self.mesh, subdomain_data=self.facet_tags)
```

No other change — `dx` and `ds` used as plain integrals are unchanged.

---

## Step 12 — `inout.py` TODO  *(separate repo — `Common/common/inout.py`)*

Required changes in `inout.py` to unblock `problem.py`:

1. `read_mesh` must return `(dolfinx.mesh.Mesh, dolfinx.mesh.MeshTags, dolfinx.mesh.MeshTags)` — replace `dolfin.HDF5File` / `XDMFFile` mesh reading with `dolfinx.io.XDMFFile`.
2. `read_HDF5_data` / `write_HDF5_data` — replace `dolfin.HDF5File` with `dolfinx.io.HDF5File`; `mpi_comm` arg becomes `comm` (`mpi4py.MPI.Comm`).
3. `utils.is_Expression` → `isinstance(obj, fem.Function)`; `utils.is_Constant` → `isinstance(obj, fem.Constant)`; `utils.is_enriched` → check `hasattr(V.ufl_element(), '_elements')`.

Until these are updated, stub them in tests with a DOLFINx-native mesh.

---

## Step 13 — `uprofile.vector()[:]` access  *(mechanical, 3 sites)*

```python
uprofile[i].vector()[:]  →  uprofile[i].x.array
```

---

## Execution order

1. Steps 1–4 (imports, MPI, version, mesh attrs) → module imports cleanly
2. Steps 5–6a (Constant helper, standard spaces) → `create_functionspaces` runs
3. Steps 10–11 (scalar assemble, Measure) → inflow area computations work
4. Step 7 (DirichletBC) → BCs can be constructed (most work, do last)
5. Steps 6b, 6d–6e (P1b, sub-space) → enriched element path works
6. Steps 8–9, 13 (Expression, project, vector access) → full BC setup
7. Step 12 (inout.py) → real mesh test end-to-end

---

## Verification

```bash
# 1. Import smoke test
python -c "from nsx.fractionalstep.problem import Problem, BoundaryConditions"

# 2. Unit cube function spaces
python tests/test_problem_spaces.py   # create unit cube mesh, call create_functionspaces for p1/p2/p1b

# 3. Scalar assemble
python -c "
from dolfinx.mesh import create_unit_cube, CellType
from mpi4py import MPI
import ufl, dolfinx.fem as fem
mesh = create_unit_cube(MPI.COMM_WORLD, 4,4,4,CellType.tetrahedron)
vol = fem.assemble_scalar(fem.form(ufl.dx(domain=mesh)))
assert abs(vol - 1.0) < 1e-12
"

# 4. DirichletBC topology test — tag one face, apply no-slip, check dof count > 0

# 5. Full YAML simulation — run 1 time step with a simple pipe mesh, compare
#    u/p norms vs. legacy FEniCS output stored in reference checkpoints
```
