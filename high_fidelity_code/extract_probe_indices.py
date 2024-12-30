from fenics import *
from mshr import *
import numpy as np

# this code is used the extract the indices in the velocity snapshots corresponding to the probe locations listed below

# locations of the target probes
target_probe_1 = [0.40, 0.20]
target_probe_2 = [0.60, 0.20]
target_probe_3 = [1.00, 0.20]

# tolerances used to assess the distance between the prescribed target probe locations and the underlying mesh points
eps1 = 0.01
eps2 = 0.01
eps3 = 0.01


channel     = Rectangle(Point(0, 0), Point(2.2, 0.41))
cylinder    = Circle(Point(0.2, 0.2), 0.05)
domain      = channel - cylinder
mesh        = generate_mesh(domain, 256)

V       = VectorFunctionSpace(mesh, 'P', 2)
element = V.element()
dofmap  = V.dofmap()

min_probe_1_dist    = []
min_probe_1_ind     = []

min_probe_2_dist    = []
min_probe_2_ind     = []

min_probe_3_dist    = []
min_probe_3_ind     = []

for cell in cells(mesh):
    elemn   = element.tabulate_dof_coordinates(cell)
    ind     = dofmap.cell_dofs(cell.index())

    for i, el in enumerate(elemn):
        dist1 = np.sqrt((el[0] - target_probe_1[0])**2 + (el[1] - target_probe_1[1])**2)
        dist2 = np.sqrt((el[0] - target_probe_2[0])**2 + (el[1] - target_probe_2[1])**2)
        dist3 = np.sqrt((el[0] - target_probe_3[0])**2 + (el[1] - target_probe_3[1])**2)

        if dist1 < eps1:
            min_probe_1_dist.append(dist1) 
            min_probe_1_ind.append(ind[i])

        if dist2 < eps2:
            min_probe_2_dist.append(dist2) 
            min_probe_2_ind.append(ind[i])

        if dist3 < eps3:
            min_probe_3_dist.append(dist3) 
            min_probe_3_ind.append(ind[i])

ind1 = np.argmin(min_probe_1_dist)
ind2 = np.argmin(min_probe_2_dist)
ind3 = np.argmin(min_probe_3_dist)

print('\033[1m The indices for the target probes are:  \033[0m')
print(target_probe_1, ': \t', min_probe_1_ind[ind1])
print(target_probe_2, ': \t', min_probe_2_ind[ind2])
print(target_probe_3, ': \t', min_probe_3_ind[ind3])