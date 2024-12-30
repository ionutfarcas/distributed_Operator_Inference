from config.config import *
from utils.utils import *

if __name__ == '__main__':

	# this flag determines whether the CPU time is saved to disk
	# this is useful when performing scaling studies, for example
	SAVE_CPUTIME = False
	if len(argv) != 2:
		print('\033[1m If you wish to perform multiple measurements of the total CPU time (e.g., for scaling studies) \033[0m')
		print('\033[1m Run the code as: ' + argv[0] + ' iteration #  \033[0m')

	else:
		SAVE_CPUTIME 	= True
		iteration 		= int(sys.argv[1])

	######## INITIALIZATION ########
	# compute the Cartesian product of all regularization pairs (beta1, beta2 )
	reg_pairs_global 	= list(product(B1, B2))
	n_reg_global 		= len(reg_pairs_global)
	###### INITIALIZATION END ######

	start_time_global 		= time()
	start_time_data_loading = time()


	######## STEP I: SEQUENTIAL TRAINING DATA LOADING ########
	# allocate memory for the global snapshot data, which has been saved to disk in HDF5 format
	Q_global = np.zeros((n, nt))
	with h5.File(H5_training_snapshots, 'r') as file:

		for j in range(ns):
			Q_global[j*nx : (j + 1)*nx, :] = file[state_variables[j]][:]

	file.close()

	end_time_data_loading 	= time()
	data_loading_time 		= end_time_data_loading - start_time_data_loading
	#################### STEP I END ###########################
	

	######## STEP II: SEQUENTIAL DATA TRANSFORMATIONS ########
	compute_time 					= 0	
	communication_time 				= 0
	start_time_data_transformations = time()

	if CENTERING:
		# compute the global temporal mean of each variable
		temporal_mean_global 	= np.mean(Q_global, axis=1)
		# center (in place) each variable with respect to its global temporal mean
		Q_global 				-= temporal_mean_global[:, np.newaxis]

	if SCALING:
		# scale the centered stated variables by their global maximum absolute value
		# this ensures that the centered and scaled variables do not exceed [-1, 1]
		for j in range(ns):

			# determine the global maximum absolute value of each centered variable on each rank
			min_centered_var_global   = np.min(Q_global[j*nx : (j + 1)*nx, :])
			max_centered_var_global   = np.max(Q_global[j*nx : (j + 1)*nx, :])
			scaling_param_global 	  = np.maximum(np.abs(min_centered_var_global), \
			                                     np.abs(max_centered_var_global))

			# scale each centered variable by its corresponding global scaling parameter
			Q_global[j*nx : (j + 1)*nx, :] /= scaling_param_global

	end_time_data_transformations 	= time()
	compute_time					+= end_time_data_transformations - start_time_data_transformations
	#################### STEP II END ##########################

	
	######## STEP III: SEQUENTIAL DIMENSIONALITY REDUCTION ########
	start_time_matmul 	= time()
	# compute the local Gram matrices on each rank
	D_global  			= np.matmul(Q_global.T, Q_global)
	end_time_matmul 	= time()
	compute_time		+= end_time_matmul - start_time_matmul
	
	start_time = time()
	
	# compute the eigendecomposition of the positive, semi-definite global Gram matrix
	eigs, eigv = np.linalg.eigh(D_global)

	# order eigenpairs by increasing eigenvalue magnitude
	sorted_indices 	= np.argsort(eigs)[::-1]
	eigs 			= eigs[sorted_indices]
	eigv 			= eigv[:, sorted_indices]

	# compute retained energy for r bteween 1 and nt
	ret_energy 	= np.cumsum(eigs)/np.sum(eigs)
	# select reduced dimension r for that the retained energy exceeds the prescribed threshold
	r 			= np.argmax(ret_energy > target_ret_energy) + 1

	# compute the auxiliary Tr matrix
	Tr_global 	= np.matmul(eigv[:, :r], np.diag(eigs[:r]**(-0.5)))
	# compute the low-dimensional representation of the high-dimensional transformed snapshot data
	Qhat_global = np.matmul(Tr_global.T, D_global)

	end_time 		= time()
	compute_time	+= end_time - start_time
	##################### STEP III END #############################


	######## STEP IV: SEQUENTIAL REDUCED OPERATOR INFERENCE ########
	learning_time_grid_search_total = 0	
	start_time_grid_search_total 	= time()

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
	# training error corresponding to the optimal regularization hyperparameters
	opt_train_err 			= 1e20

	# loop over the regularization pairs corresponding to each MPI rank
	for pair in reg_pairs_global:

		# extract beta1 and beta2 from each candidate regularization pair
		beta1 = pair[0]
		beta2 = pair[1]

		start_time_OpInf_learning = time()

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

		end_time_OpInf_learning = time()

		# define the OpInf reduced model 
		dOpInf_red_model 	= lambda x: Ahat @ x + Fhat @ compute_Qhat_sq(x) + chat
		# extract the reduced initial condition from Qhat_1
		qhat0 				= Qhat_1[0, :]
		
		# compute the reduced solution over the trial time horizon, which here is the same as the target time horizon
		start_time_OpInf_eval 			= time()
		contains_nans, Qtilde_OpInf 	= solve_opinf_difference_model(qhat0, nt_p, dOpInf_red_model)
		end_time_OpInf_eval 			= time()

		time_OpInf_learning = end_time_OpInf_learning - start_time_OpInf_learning
		time_OpInf_eval 	= end_time_OpInf_eval - start_time_OpInf_eval

		learning_time_grid_search_total += time_OpInf_learning
		
		# for each candidate regulairzation pair, we compute the training error 
		# we also save the corresponding reduced solution, learning time and ROM evaluation time
 		# and compute the ratio of maximum coefficient growth in the trial period to that in the training period
		if contains_nans == False:
			train_err     			= compute_train_err(Qhat_global.T[:nt, :], Qtilde_OpInf[:nt, :])
			max_diff_Qhat_trial  	= np.max(np.abs(Qtilde_OpInf - mean_Qhat_train), axis=0)			
			max_growth_trial  		= np.max(max_diff_Qhat_trial)/np.max(max_diff_Qhat_train)

			if max_growth_trial < max_growth:

				if train_err < opt_train_err:
					opt_train_err 				= train_err
					Qtilde_OpInf_opt 			= Qtilde_OpInf
					OpInf_wtime_learning_opt	= time_OpInf_learning
					OpInf_ROM_wtime_opt			= time_OpInf_eval

					beta1_opt = pair[0]
					beta2_opt = pair[1]

	end_time_grid_search_total 		= time()
	compute_time_grid_search_total	= end_time_grid_search_total - start_time_grid_search_total
	####################### STEP IV END #############################

	if POSTPROC:
		######## POSTPROCESSING ########
		start_time_postproc = time()

		if POSTPROC_FULL_DOM_SOL:
			# compute the global POD basis vectors
			Phir_global = np.matmul(Q_global, Tr_global)

			# extract and save to disk the approximate full state at the time instants specified in target_time_instants 
			for target_var_index in range(ns):
				Phir_full_state 			= Phir_global[target_var_index*nx : (target_var_index + 1)*nx, :]
				temporal_mean_full_state 	= temporal_mean_global[target_var_index*nx : (target_var_index + 1)*nx]

				full_state_rec = Phir_full_state @ Qtilde_OpInf_opt.T[:, target_time_instants] + temporal_mean_full_state[:, np.newaxis]
				
				np.save('postprocessing/sOpInf_postprocessing/sOpInf_full_state_var_' + str(target_var_index + 1) + '.npy', full_state_rec)

		# extract and save to disk the approximate solutions at the probe locations specified in target_probe_indices
		for target_var_index in range(ns):

			for j, probe_index in enumerate(target_probe_indices):

				# compute the components of the POD basis vectors corresponding to the target probe locations
				Phir_probe 			= np.matmul(Q_global[probe_index + target_var_index*nx, :], Tr_global)
				temporal_mean_probe = temporal_mean_global[probe_index + target_var_index*nx]

				var_probe_prediction = Phir_probe @ Qtilde_OpInf_opt.T + temporal_mean_probe

				np.save('postprocessing/sOpInf_postprocessing/sOpInf_probe_' + str(j + 1) + '_var_' + str(target_var_index + 1) + '.npy', var_probe_prediction)

		end_time_postproc = time()

		runtime_postproc 	= end_time_postproc - start_time_postproc
		###### POSTPROCESSING END ######

	end_time_global = time()
	runtime_global 	= end_time_global - start_time_global	

	print('\033[1m The reduced dimension that satisfies the target retained energy is: {} \033[0m'.format(r))
	print('*************************')

	print('\033[1m Optimal regularization pair is: ({}, {}) \033[0m'.format(beta1_opt, beta2_opt))
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
		runtime_OpInf = [data_loading_time, compute_time, compute_time_grid_search_total, learning_time_grid_search_total, OpInf_wtime_learning_opt, OpInf_ROM_wtime_opt, runtime_global]
			
		sOpInf_runtime_strong_scaling = lambda iteration: 'runtimes/sOpInf_runtime_iteration_' + str(iteration) + '.npy'
		np.save(sOpInf_runtime_strong_scaling(iteration), runtime_OpInf)