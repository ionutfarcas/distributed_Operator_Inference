import numpy as np
import h5py as h5
from mpi4py import MPI
from itertools import product
from sys import argv
from time import time

# DoF setup
ns 	= 2
n 	= 292678 
nx  = int(n/ns)

# number of training snapshots
nt = 600

# state variable names
state_variables = ['u_x', 'u_y']

# path to the HDF5 file containing the training snapshots
H5_snapshots_all 		= 'navier_stokes_benchmark/velocity_snapshots_full.h5'
H5_training_snapshots 	= 'navier_stokes_benchmark/velocity_training_snapshots.h5' 

# number of time instants over the time domain of interest (training + prediction)
nt_p = 1200

# define target retained energy for the OpInf ROM
target_ret_energy = 0.9996

# ranges for the regularization parameter pairs
B1 = np.logspace(-10., 0., num=8)
B2 = np.logspace(-4., 4., num=8)

# maximum variance of reduced training data for optimal regularization parameter selection
max_growth = 1.2

# flag to determine whether the (transformed) training data are centered with respect to the temporal mean 
CENTERING = True

# flag to determine whether the (transformed) training data are scaled by the maximum absolute value of each state variable
SCALING = False

# flag to determine whether we postprocess the OpInf reduced solution
POSTPROC = True

# flag to determine whether we compute the ROM approximate solution in the original cooridnates in the full domain 
# at user specified time instants (specified in target_time_instants)
POSTPROC_FULL_DOM_SOL = False

# flag to determine whether we compute the ROM approximate solution in the original coordinates
# at user specified probe locations (specified in target_probe_indices)
POSTPROC_PROBES = True

# time instants at which to save the OpInf approximate solutions mappend to the original coordinates
target_time_instants 	= [-1]
# indices for probe locations (0.40, 0.20), (0.60, 0.20) and (1.00, 0.20)
target_probe_indices 	= [48250, 77502, 130722]