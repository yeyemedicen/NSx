"""
Convert legacy FEniCS rectangle.h5 → DOLFINx-compatible rectangle_shared.xdmf + .h5

Run with:
    conda run -n FenicsLegacy python tests/convert_mesh.py
"""
import sys
sys.path.insert(0, '/Users/yeye/Common')

from dolfin import Mesh, HDF5File, MeshFunction, MPI
import numpy as np
import h5py

LEGACY_H5  = '/Users/yeye/NavierStokes/tests/rectangle.h5'
OUT_H5     = '/Users/yeye/NSx/tests/rectangle_shared.h5'
OUT_XDMF   = '/Users/yeye/NSx/tests/rectangle_shared.xdmf'

# ── Read legacy mesh ──────────────────────────────────────────────────────────
mesh = Mesh()
with HDF5File(MPI.comm_world, LEGACY_H5, 'r') as hdf:
    hdf.read(mesh, '/mesh', False)
    bnds = MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
    hdf.read(bnds, '/boundaries')

coords = mesh.coordinates().copy()          # (N, 2) float64
cells  = mesh.cells().copy().astype('int64') # (T, 3) int64
N = len(coords)
T = len(cells)

# boundary facets: filter to non-zero markers
raw_vals  = bnds.array()                     # all facets
from dolfin import facets as dolfin_facets
raw_topo  = np.zeros((len(raw_vals), 2), dtype='int64')
for i, f in enumerate(dolfin_facets(mesh)):
    verts = f.entities(0).tolist()
    raw_topo[i] = sorted(verts)

mask        = raw_vals > 0
facet_topo  = raw_topo[mask].astype('int64')   # (F, 2)
facet_vals  = raw_vals[mask].astype('int32')    # (F,)
F = len(facet_topo)

print(f'Mesh: {N} nodes, {T} cells, {F} marked boundary facets')
print(f'Boundary IDs: {np.unique(facet_vals)}')

# ── Write DOLFINx-format HDF5 ─────────────────────────────────────────────────
with h5py.File(OUT_H5, 'w') as f:
    f.create_dataset('Mesh/mesh/geometry', data=coords)
    f.create_dataset('Mesh/mesh/topology', data=cells)
    f.create_dataset('MeshTags/facets/topology', data=facet_topo)
    f.create_dataset('MeshTags/facets/Values',   data=facet_vals.reshape(-1, 1))

# ── Write XDMF header ─────────────────────────────────────────────────────────
h5name = 'rectangle_shared.h5'
xdmf = f"""<?xml version="1.0"?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="3.0" xmlns:xi="https://www.w3.org/2001/XInclude">
  <Domain>
    <Grid Name="mesh" GridType="Uniform">
      <Topology TopologyType="Triangle" NumberOfElements="{T}" NodesPerElement="3">
        <DataItem Dimensions="{T} 3" NumberType="Int" Format="HDF">{h5name}:/Mesh/mesh/topology</DataItem>
      </Topology>
      <Geometry GeometryType="XY">
        <DataItem Dimensions="{N} 2" Format="HDF">{h5name}:/Mesh/mesh/geometry</DataItem>
      </Geometry>
    </Grid>
    <Grid Name="facets" GridType="Uniform">
      <xi:include xpointer="xpointer(/Xdmf/Domain/Grid/Geometry)" />
      <Topology TopologyType="PolyLine" NumberOfElements="{F}" NodesPerElement="2">
        <DataItem Dimensions="{F} 2" NumberType="Int" Format="HDF">{h5name}:/MeshTags/facets/topology</DataItem>
      </Topology>
      <Attribute Name="facets" AttributeType="Scalar" Center="Cell">
        <DataItem Dimensions="{F} 1" Format="HDF">{h5name}:/MeshTags/facets/Values</DataItem>
      </Attribute>
    </Grid>
  </Domain>
</Xdmf>
"""
with open(OUT_XDMF, 'w') as f:
    f.write(xdmf)

print(f'Written: {OUT_H5}')
print(f'Written: {OUT_XDMF}')
