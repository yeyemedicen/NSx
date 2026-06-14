# Context: Automatic Parabolic Dirichlet BC for FSI Solver (DOLFINx)

## Goal

Implement a fully automatic parabolic (fully-developed) velocity profile as a
Dirichlet boundary condition for the fluid inlet in a DOLFINx-based FSI solver.
The method must work for **any inlet shape and any spatial orientation** — no
hard-coded normals, radii, or axis assumptions.

---

## Core Mathematical Idea

Given a set of inlet DOF coordinates in 3D, the method:

1. Computes an **orthonormal local frame** (two tangent vectors + normal) directly
   from the point cloud using SVD — no geometric assumptions needed.
2. **Projects** all inlet points into the local 2D frame.
3. Computes a **bounding box** in that local frame to get effective half-extents
   `R1` and `R2`.
4. Evaluates an **elliptic paraboloid** in the local coordinates:

   ```
   f(s1, s2) = clip(1 - (s1/R1)^2 - (s2/R2)^2, 0, None)
   ```

5. The final velocity field is `u = U_max * f * n_hat`, where `n_hat` is the
   inward unit normal from the SVD.

The `clip(..., 0)` handles non-elliptic shapes (e.g. square inlets) where corner
DOFs fall outside the inscribed ellipse and would otherwise produce negative values.

---

## Implementation

### Step 1 — Local frame from SVD

```python
import numpy as np

def compute_inlet_local_frame(inlet_coords: np.ndarray):
    """
    inlet_coords : (N, 3) float array of inlet DOF positions.

    Returns
    -------
    centroid : (3,) array
    t1       : (3,) principal tangent  (max variance direction)
    t2       : (3,) secondary tangent
    n        : (3,) unit normal  (min variance direction)
    R1       : half-extent along t1
    R2       : half-extent along t2
    """
    centroid = inlet_coords.mean(axis=0)
    centered = inlet_coords - centroid

    # SVD: rows of Vt are right singular vectors, ordered by descending variance
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)

    t1 = Vt[0]   # max variance  -> tangent 1
    t2 = Vt[1]   # mid variance  -> tangent 2
    n  = Vt[2]   # min variance  -> normal (already unit length from SVD)

    s1 = centered @ t1
    s2 = centered @ t2

    R1 = (s1.max() - s1.min()) / 2.0
    R2 = (s2.max() - s2.min()) / 2.0

    return centroid, t1, t2, n, R1, R2
```

> **Note on normal orientation:** SVD gives an arbitrary sign for `n`. Make sure
> it points *inward* (into the fluid domain) before using it. A simple check:
> pick an interior point of the mesh, compute `(interior - centroid) · n`, and
> flip `n` if the dot product is negative.

---

### Step 2 — Parabolic profile interpolation function

```python
def make_inlet_velocity(inlet_dof_coords: np.ndarray, U_max: float,
                        flip_normal: bool = False):
    """
    Returns a callable suitable for Function.interpolate() in DOLFINx.

    inlet_dof_coords : (N, 3) coordinates of DOFs on the inlet face.
    U_max            : peak velocity magnitude at the centroid.
    flip_normal      : set True if the SVD normal points outward.
    """
    centroid, t1, t2, n, R1, R2 = compute_inlet_local_frame(inlet_dof_coords)
    if flip_normal:
        n = -n

    def inlet_velocity(x):
        # x has shape (3, N) in DOLFINx convention
        pts = x.T - centroid          # (N, 3)
        s1  = pts @ t1                # (N,)
        s2  = pts @ t2                # (N,)

        profile = np.clip(1.0 - (s1 / R1)**2 - (s2 / R2)**2, 0.0, None)  # (N,)

        # Velocity vector = profile * U_max * n  (broadcast over components)
        values = np.zeros((3, x.shape[1]))
        for i in range(3):
            values[i] = U_max * profile * n[i]

        return values

    return inlet_velocity
```

---

### Step 3 — Wiring into DOLFINx

```python
from dolfinx import fem, mesh
from dolfinx.fem import functionspace, Function

# --- Locate inlet facets (adapt the marker to your geometry) ---
fdim = domain.topology.dim - 1
inlet_facets = mesh.locate_entities_boundary(
    domain, fdim,
    marker=lambda x: np.isclose(x[0], x_inlet_value)   # example: x=0 plane
)

# --- Build vector function space (Taylor-Hood P2/P1 or similar) ---
V = functionspace(domain, ("Lagrange", 2, (domain.geometry.dim,)))

# --- Get inlet DOF coordinates ---
inlet_dofs_all = fem.locate_dofs_topological(V, fdim, inlet_facets)
all_coords     = V.tabulate_dof_coordinates()            # (total_dofs, 3)
inlet_coords   = all_coords[inlet_dofs_all]              # (N_inlet, 3)

# --- Create and apply the BC ---
u_inlet = Function(V)
u_inlet.interpolate(make_inlet_velocity(inlet_coords, U_max=1.0, flip_normal=False))

bc_inlet = fem.dirichletbc(u_inlet, inlet_dofs_all)
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| SVD on centered point cloud | Gives orthonormal frame for any orientation; no axis assumptions |
| Bounding box in local frame | Automatic scale; no manual `R` needed; adapts to any convex shape |
| Elliptic paraboloid | Exact for circles/ellipses; reasonable approximation for polygons |
| `clip(..., 0)` | Prevents negative velocities at corners of non-elliptic inlets |
| `u = f * n_hat` | Velocity aligned with inlet normal — correct for any tilt |

---

## Limitations and When to Use Approach 3 Instead

The bounding-box elliptic paraboloid is **not exactly zero on non-elliptic walls**
(e.g. square or triangular inlets). For FSI problems where the inlet shape matters
physically, consider the Laplacian-based approach instead:

- Solve `-∇²φ = 1` on a submesh of the inlet face with `φ = 0` on its boundary.
- Normalize: `u_inlet = U_max * φ / max(φ) * n_hat`.
- This gives the exact fully-developed Stokes profile for any cross-section.

Use the SVD/bounding-box method when:
- The inlet is roughly circular or elliptic.
- You want a quick, zero-dependency implementation.
- Approximate profiles are acceptable (e.g. as initial conditions).

Use the Laplacian submesh method when:
- The inlet has a sharp polygonal cross-section.
- You need the physically correct Hagen-Poiseuille or Stokes profile.

---

## Dependencies

- `numpy` (SVD, array ops)
- `dolfinx` + `petsc4py` (standard FSI stack)
- No additional packages required for the SVD/bounding-box method.

---

## Files to Create / Integrate

- `boundary_conditions.py` — house `compute_inlet_local_frame` and
  `make_inlet_velocity` as standalone utilities.
- Import into the main solver script and call during BC setup, before
  `fem.dirichletbc`.
- For time-dependent FSI, the profile shape is fixed; only `U_max` may vary
  (e.g. ramp-up). Re-interpolate `u_inlet` each time step if `U_max = U_max(t)`.
