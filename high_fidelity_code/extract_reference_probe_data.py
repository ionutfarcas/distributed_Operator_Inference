import numpy as np
import h5py as h5

nt_p 				= 1200
H5_snapshots_all 	= '../navier_stokes_benchmark/velocity_snapshots_full.h5'

if __name__ == '__main__':

	ref_var_probe = np.zeros(nt_p)

	try:
		with h5.File(H5_snapshots_all, 'r') as file:

			group 					= file['/Vector']
			snapshot_indices 		= [str(index) for index in range(800, 2000)]
			
			for target_var_index in range(ns):
				for k, probe_index in enumerate(target_probe_indices):
					for j, key in enumerate(snapshot_indices):
						ref_var_probe[j] = group[key][probe_index + target_var_index*nx]

					np.save('../postprocessing/ref_data/ref_probe_' + str(k + 1) + '_var_' + str(target_var_index + 1) + '.npy', ref_var_probe)

		file.close()

	except FileNotFoundError:
		print("File containing the full dataset doesn't exist!")
		print("To generate the full dataset, you can run the\033[1m generate_high_fidelity_data.py \033[0mscript")
