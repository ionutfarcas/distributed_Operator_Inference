from config.config import *
from utils.utils import *

if __name__ == '__main__':

	######## INITIALIZATION ########
	# MPI initialization
	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	size = comm.Get_size()

	# this flag determines whether the CPU time is saved to disk
	# this is useful when performing scaling studies, for example
	SAVE_CPUTIME = False
	if len(argv) != 2:

		if rank == 0:
			print('\033[1m If you wish to perform multiple measurements of the total CPU time (e.g., for scaling studies) \033[0m')
			print('\033[1m Run the code as: ' + argv[0] + ' iteration #  \033[0m')

	else:
		SAVE_CPUTIME 	= True
		iteration 		= int(sys.argv[1])

	# compute the Cartesian product of all regularization pairs (beta1, beta2 )
	reg_pairs_global 	= list(product(B1, B2))
	n_reg_global 		= len(reg_pairs_global)

	# distribute the regularization pairs among the p MPI ranks
	start_ind_reg_params, end_ind_reg_params 	= distribute_reg_pairs(rank, n_reg_global, size)
	reg_pairs_rank 								= reg_pairs_global[start_ind_reg_params : end_ind_reg_params]

	# the start and end indices, and the total number of snapshots for each MPI rank
	nx_i_start, nx_i_end, nx_i = distribute_nx(rank, nx, size)
	###### INITIALIZATION END ######

	start_time_global 		= MPI.Wtime()
	start_time_data_loading = MPI.Wtime()
	
	######## STEP I: DISTRIBUTED TRAINING DATA LOADING ########
	# allocate memory for the snapshot data corresponding to each MPI rank
	# note that the full snapshot data has been saved to disk in HDF5 format
	Q_rank = np.zeros((ns * nx_i, nt))
	with h5.File(H5_training_snapshots, 'r') as file:

		for j in range(ns):
			Q_rank[j*nx_i : (j + 1)*nx_i, :] = \
			    file[state_variables[j]][nx_i_start : nx_i_end, :]

	file.close()

	end_time_data_loading 	= MPI.Wtime()
	data_loading_time 		= end_time_data_loading - start_time_data_loading
	#################### STEP I END ###########################
	

	######## STEP II: DISTRIBUTED DATA TRANSFORMATIONS ########
	compute_time 					= 0	
	communication_time 				= 0
	start_time_data_transformations = MPI.Wtime()

	if CENTERING:
		# compute the temporal mean of each variable on each rank
		temporal_mean_rank  = np.mean(Q_rank, axis=1)
		# center (in place) each variable with respect to its temporal mean on each rank
		Q_rank              -= temporal_mean_rank[:, np.newaxis]

	if SCALING:
		# scale the centered stated variables by their global maximum absolute value
		# this ensures that the centered and scaled variables do not exceed [-1, 1]
		for j in range(ns):

		    # determine the local maximum absolute value of each centered variable on each rank
		    min_centered_var_rank   = np.min(Q_rank[j*nx_i : (j + 1)*nx_i, :])
		    max_centered_var_rank   = np.max(Q_rank[j*nx_i : (j + 1)*nx_i, :])
		    scaling_param_rank 	    = np.maximum(np.abs(min_centered_var_rank), \
		                                         np.abs(max_centered_var_rank))

		    # determine the global maximum absolute value via a parallel reduction
		    # since all ranks require the global scaling parameters, the reduction results are also broadcasted to all ranks
		    scaling_param_global = np.zeros_like(scaling_param_rank)
		    comm.Allreduce(scaling_param_rank, scaling_param_global, op=MPI.MAX)

		    # scale each centered variable by its corresponding global scaling parameter
		    Q_rank[j*nx_i : (j + 1)*nx_i, :] /= scaling_param_global

	end_time_data_transformations 	= MPI.Wtime()
	compute_time					+= end_time_data_transformations - start_time_data_transformations
	#################### STEP II END ##########################

	
	######## STEP III: DISTRIBUTED DIMENSIONALITY REDUCTION ########
	start_time_matmul 	= MPI.Wtime()
	# compute the local Gram matrices on each rank
	D_rank  			= np.matmul(Q_rank.T, Q_rank)
	end_time_matmul 	= MPI.Wtime()
	compute_time		+= end_time_matmul - start_time_matmul
	
	start_time_reduction = MPI.Wtime()

	# aggregate local Gram matrices to form global Gram matrix and distribute the result to all ranks
	D_global = np.zeros_like(D_rank)
	comm.Allreduce(D_rank, D_global, op=MPI.SUM)

	end_time_reduction 		= MPI.Wtime()
	communication_time		+= end_time_reduction - start_time_reduction

	start_time = MPI.Wtime()

	# compute the eigendecomposition of the positive, semi-definite global Gram matrix
	eigs, eigv = np.linalg.eigh(D_global)

	# order eigenpairs by increasing eigenvalue magnitude
	sorted_indices 	= np.argsort(eigs)[::-1]
	eigs 			= eigs[sorted_indices]
	eigv 			= eigv[:, sorted_indices]

	if rank == 0:
		np.save('postprocessing/dOpInf_postprocessing/Sigma_sq_global.npy', eigs)

	# compute retained energy for r bteween 1 and nt
	ret_energy 	= np.cumsum(eigs)/np.sum(eigs)
	# select reduced dimension r for that the retained energy exceeds the prescribed threshold
	r 			= np.argmax(ret_energy > target_ret_energy) + 1

	# compute the auxiliary Tr matrix
	Tr_global 	= np.matmul(eigv[:, :r], np.diag(eigs[:r]**(-0.5)))
	# compute the low-dimensional representation of the high-dimensional transformed snapshot data
	Qhat_global = np.matmul(Tr_global.T, D_global)

	end_time 		= MPI.Wtime()
	compute_time	+= end_time - start_time
	##################### STEP III END #############################


	######## STEP IV: DISTRIBUTED REDUCED OPERATOR INFERENCE ########
	learning_time_grid_search_total = 0
	compute_time_grid_search_total 	= 0	
	start_time_grid_search_total 	= MPI.Wtime()

	# extract left and right shifted reduced data matrices for the discrete OpInf learning problem
	Qhat_1 = Qhat_global.T[:-1, :]
	Qhat_2 = Qhat_global.T[1:, :]

	# column dimension of the data matrix Dhat used in the discrete OpInf learning problem
	s = int(r*(r + 1)/2)
	d = r + s + 1

	# compute the non-redundant quadratic terms of Qhat_1 squared
	Qhat_1_sq = compute_Qhat_sq(Qhat_1)

	# define the constant part (due to mean shifting) in the discrete OpInf learning problem
	K 		= Qhat_1.shape[0]
	Ehat 	= np.ones((K, 1))

	# assemble the data matrix Dhat for the discrete OpInf learning problem
	Dhat   = np.concatenate((Qhat_1, Qhat_1_sq, Ehat), axis=1)
	# compute Dhat.T @ Dhat for the normal equations to solve the OpInf least squares minimization
	Dhat_2 = Dhat.T @ Dhat

	# compute the temporal mean and maximum deviation of the reduced training data
	mean_Qhat_train   	 	= np.mean(Qhat_global.T, axis=0)
	max_diff_Qhat_train 	= np.max(np.abs(Qhat_global.T - mean_Qhat_train), axis=0)

	# dictionary to store the regularization pair for each training error
	opt_train_err_reg_pair 			= {}
	# dictionary to store the reduced solutions over the target time horizon for each training error
	Qtilde_dOpInf_reg_pair 			= {}
	# dictionary to store the OpInf reduced model learning time for each training error
	OpInf_wtime_learning_reg_pair 	= {}
	# dictionary to store the OpInf reduced model evaluation time for each training error
	OpInf_ROM_wtime_reg_pair 		= {}

	# loop over the regularization pairs corresponding to each MPI rank
	for pair in reg_pairs_rank:

		# extract beta1 and beta2 from each candidate regularization pair
		beta1 = pair[0]
		beta2 = pair[1]

		start_time_OpInf_learning = MPI.Wtime()

		# regularize the linear and constant reduced operators using beta1, and the quadratic operator using beta2
		regg            = np.zeros(d)
		regg[:r]        = beta1
		regg[r : r + s] = beta2
		regg[r + s:]    = beta1
		regularizer     = np.diag(regg)
		Dhat_2_reg 		= Dhat_2 + regularizer

		# solve the OpInf learning problem by solving the regularized normal equations
		Ohat = np.linalg.solve(Dhat_2_reg, np.dot(Dhat.T, Qhat_2)).T

		# extract the linear, quadratic, and constant reduced model operators
		Ahat = Ohat[:, :r]
		Fhat = Ohat[:, r:r + s]
		chat = Ohat[:, r + s]

		end_time_OpInf_learning = MPI.Wtime()

		# define the OpInf reduced model 
		dOpInf_red_model 	= lambda x: Ahat @ x + Fhat @ compute_Qhat_sq(x) + chat
		# extract the reduced initial condition from Qhat_1
		qhat0 				= Qhat_1[0, :]
		
		# compute the reduced solution over the trial time horizon, which here is the same as the target time horizon
		start_time_OpInf_eval 			= MPI.Wtime()
		contains_nans, Qtilde_dOpInf 	= solve_opinf_difference_model(qhat0, nt_p, dOpInf_red_model)
		end_time_OpInf_eval 			= MPI.Wtime()

		time_OpInf_learning = end_time_OpInf_learning - start_time_OpInf_learning
		time_OpInf_eval 	= end_time_OpInf_eval - start_time_OpInf_eval

		learning_time_grid_search_total += time_OpInf_learning
		
		# for each candidate regulairzation pair, we compute the training error 
		# we also save the corresponding reduced solution, learning time and ROM evaluation time
 		# and compute the ratio of maximum coefficient growth in the trial period to that in the training period
		if contains_nans == False:
			train_err     			= compute_train_err(Qhat_global.T[:nt, :], Qtilde_dOpInf[:nt, :])
			max_diff_Qhat_trial  	= np.max(np.abs(Qtilde_dOpInf - mean_Qhat_train), axis=0)			
			max_growth_trial  		= np.max(max_diff_Qhat_trial)/np.max(max_diff_Qhat_train)

			if max_growth_trial < max_growth:
				opt_train_err 									= train_err
				opt_train_err_reg_pair[opt_train_err] 			= pair
				Qtilde_dOpInf_reg_pair[opt_train_err] 			= Qtilde_dOpInf
				OpInf_wtime_learning_reg_pair[opt_train_err]	= time_OpInf_learning
				OpInf_ROM_wtime_reg_pair[opt_train_err]			= time_OpInf_eval

		else:
			opt_train_err 									= 1e20
			opt_train_err_reg_pair[opt_train_err] 			= pair
			Qtilde_dOpInf_reg_pair[opt_train_err] 			= Qtilde_dOpInf
			OpInf_wtime_learning_reg_pair[opt_train_err]	= time_OpInf_learning
			OpInf_ROM_wtime_reg_pair[opt_train_err]			= time_OpInf_eval

	end_time_grid_search_total 		= MPI.Wtime()
	compute_time_grid_search_total	+= end_time_grid_search_total - start_time_grid_search_total
	
	# minimize global training error by reducing local results on each rank, subject to the bound constraint on inferred reduced coefficients
	opt_key_rank 	= np.min(list(opt_train_err_reg_pair.keys()))
	opt_key_rank 	= np.array([opt_key_rank])
	opt_key_global 	= np.zeros_like(opt_key_rank)

	start_time_reduction = MPI.Wtime()
	comm.Allreduce(opt_key_rank, opt_key_global, op=MPI.MIN)
	end_time_reduction 		= MPI.Wtime()
	# communication_time		+= end_time_reduction - start_time_reduction
	
	start_time_extract_opt_sol = MPI.Wtime() 
	
	opt_key_global = opt_key_global[0]

	# extract optimal hyperparameters, and corresponding reduced model and inference time
	if opt_key_rank == opt_key_global:

		rank_reg_opt 		= rank
		Qtilde_dOpInf_opt 	= Qtilde_dOpInf_reg_pair[opt_key_global]

		reg_pair_opt = opt_train_err_reg_pair[opt_key_global]

		beta1_opt = reg_pair_opt[0]
		beta2_opt = reg_pair_opt[1]

		OpInf_wtime_learning_opt 	= OpInf_wtime_learning_reg_pair[opt_key_global]
		OpInf_ROM_wtime_opt 		= OpInf_ROM_wtime_reg_pair[opt_key_global]
	else:
		rank_reg_opt	 = -1
		Qtilde_dOpInf_opt = None

	end_time_extract_opt_sol 		= MPI.Wtime()
	compute_time_grid_search_total	+= end_time_extract_opt_sol - start_time_extract_opt_sol
	####################### STEP IV END #############################

	if POSTPROC:
		######## POSTPROCESSING ########
		# compute the components of the POD basis vectors on each rank
		start_time_postproc = MPI.Wtime()

		# broadcast the dOpInf reduced solution from the rank containing the optimal regularization pair to all ranks
		if rank == rank_reg_opt:
			for i in range(size):
				if i != rank_reg_opt:
					comm.send(Qtilde_dOpInf_opt, dest=i)
		else:
			Qtilde_dOpInf_opt = comm.recv(Qtilde_dOpInf_opt, source=rank_reg_opt)

		if POSTPROC_FULL_DOM_SOL:
			# compute the components of the POD basis vectors on each rank
			Phir_rank = np.matmul(Q_rank, Tr_global)

			# extract and save to disk the approximate full state at the time instants specified in target_time_instants 
			for target_var_index in range(ns):
				Phir_full_state 			= Phir_rank[target_var_index*nx_i : (target_var_index + 1)*nx_i, :]
				temporal_mean_full_state 	= temporal_mean_rank[target_var_index*nx_i : (target_var_index + 1)*nx_i]

				full_state_rec = Phir_full_state @ Qtilde_dOpInf_opt.T[:, target_time_instants] + temporal_mean_full_state[:, np.newaxis]
				full_state_rec = comm.gather(full_state_rec, root=0)

				if rank == 0:
					full_state_rec 	= np.vstack((full_state_rec))

					np.save('postprocessing/dOpInf_postprocessing/dOpInf_full_state_var_' + str(target_var_index + 1) + '.npy', full_state_rec)

		if POSTPROC_PROBES:
			# extract and save to disk the approximate solutions at the probe locations with indices specified in target_probe_indices
			for target_var_index in range(ns):

				for j, probe_index in enumerate(target_probe_indices):

					# map the global probe indices to local indices
					if probe_index >= nx_i_start and probe_index < nx_i_end:
						probe_index -= nx_i_start			

						# compute the components of the POD basis vectors corresponding to the target probe locations on each rank
						Phir_probe 			= np.matmul(Q_rank[probe_index + target_var_index*nx_i, :], Tr_global)
						 # same for the temporal mean used for centering
						temporal_mean_probe = temporal_mean_rank[probe_index + target_var_index*nx_i]

						var_probe_prediction = Phir_probe @ Qtilde_dOpInf_opt.T + temporal_mean_probe

						np.save('postprocessing/dOpInf_postprocessing/dOpInf_probe_' + str(j + 1) + '_var_' + str(target_var_index + 1) + '.npy', var_probe_prediction)

		end_time_postproc = MPI.Wtime()

		runtime_postproc = end_time_postproc - start_time_postproc
		###### POSTPROCESSING END ######
	
	end_time_global = MPI.Wtime()
	runtime_global 	= end_time_global - start_time_global	
		

	if rank == rank_reg_opt:

		print('\033[1m The reduced dimension that satisfies the target retained energy is: {} \033[0m'.format(r))
		print('*************************')

		print('\033[1m The optimal regularization pair was found on rank: {} \033[0m'.format(rank_reg_opt))
		print('*************************')

		print('\033[1m The optimal regularization pair is: ({}, {}) \033[0m'.format(beta1_opt, beta2_opt))
		print('*************************')

		print("\033[1m Runtime for data loading: {} seconds \033[0m".format(data_loading_time))
		print("\033[1m Runtime for computations: {} seconds \033[0m".format(compute_time))
		print("\033[1m Runtime for communication: {} seconds \033[0m".format(communication_time))
		print('*************************')

		print("\033[1m Runtime for grid search computations (total): {} seconds \033[0m".format(compute_time_grid_search_total))
		print("\033[1m Runtime for grid search learning (total): {} seconds \033[0m".format(learning_time_grid_search_total))
		print("\033[1m Runtime for learning: {} seconds \033[0m".format(OpInf_wtime_learning_opt))
		print("\033[1m Runtime for ROM eval: {} seconds \033[0m".format(OpInf_ROM_wtime_opt))
		print('*************************')

		if POSTPROC:
			print("\033[1m Runtime for postprocessing: {} seconds \033[0m".format(runtime_postproc))
			print('*************************')

		print("\033[1m Global runtime: {} seconds \033[0m".format(runtime_global))
		print('*************************')

		if SAVE_CPUTIME:
			runtime_dOpInf = [data_loading_time, compute_time, communication_time, compute_time_grid_search_total, learning_time_grid_search_total, OpInf_wtime_learning_opt, OpInf_ROM_wtime_opt, runtime_global]
				
			dOpInf_runtime_strong_scaling = lambda iteration: 'runtimes/dOpInf_runtime_using_' + str(size) + '_cores_' + str(iteration) + '.npy'
			np.save(dOpInf_runtime_strong_scaling(iteration), runtime_dOpInf)
	
	# Terminate MPI
	MPI.Finalize()