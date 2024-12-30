import numpy as np

def distribute_nx(rank, nx, size):
	"""
 	distribute_nx distributes the spatial DoF nx into chunks of size nxi such that 
 	\sum_{i=0}^{p-1} nx_i = nx where p is the number of used compute cores

 	:rank: 	the MPI rank 0, 1, ... ,p-1 that will run this function
 	:n_x: 	number of DoF used for spatial discretization
 	:size: 	size of the MPI communicator (p in our case)
 	
 	:return: the start and end index of the local DoF, and the number of local DoF for each rank
 	"""

	nx_i_equal = int(nx/size)

	nx_i_start = rank * nx_i_equal
	nx_i_end   = (rank + 1) * nx_i_equal

	if rank == size - 1 and nx_i_end != nx:
		nx_i_end += nx - size*nx_i_equal

	nx_i = nx_i_end - nx_i_start

	return nx_i_start, nx_i_end, nx_i

def distribute_reg_pairs(rank, n_reg, size):
	"""
 	get_reg_params_per_rank returns the index of the first and last regularization pair for each MPI rank
 
 	:rank: 		MPI rank 0, 1, ... ,p-1 
 	:n_reg: 	total number of regularization parameter pairs
 	:size: 		size of the MPI communicator (p in our case)

 	:return: the start and end indices, and the total number of snapshots for each MPI rank
 	"""

	nreg_i_equal = int(n_reg/size)

	start = rank * nreg_i_equal
	end   = (rank + 1) * nreg_i_equal

	if rank == size - 1 and end != n_reg:
		end += n_reg - size*nreg_i_equal

	return start, end

def compute_Qhat_sq(Qhat):
	"""
	compute_Qhat_sq returns the non-redundant terms in Qhat squared

	:Qhat: reduced data

	:return: Qhat_sq containing the non-redundant in Qhat squared
	"""

	if len(np.shape(Qhat)) == 1:

	    r 		= np.size(Qhat)
	    prods 	= []
	    for i in range(r):
	        temp = Qhat[i]*Qhat[i:]
	        prods.append(temp)

	    Qhat_sq = np.concatenate(tuple(prods))

	elif len(np.shape(Qhat)) == 2:
	    K, r 	= np.shape(Qhat)    
	    prods 	= []
	    
	    for i in range(r):
	        temp = np.transpose(np.broadcast_to(Qhat[:, i], (r - i, K)))*Qhat[:, i:]
	        prods.append(temp)
	    
	    Qhat_sq = np.concatenate(tuple(prods), axis=1)

	else:
	    print('invalid input!')

	return Qhat_sq

def compute_train_err(Qhat_train, Qtilde_train):
	"""
	compute_train_err computes the OpInf training error

	:Qhat_train: 	Qhat_trainerence data
	:Qtilde_train: 	Qtilde_train data

	:return: train_err containing the value of the training error
	"""
	train_err = np.max(np.sqrt(np.sum( (Qtilde_train - Qhat_train)**2, axis=1) / np.sum(Qhat_train**2, axis=1)))

	return train_err

def solve_opinf_difference_model(qhat0, n_steps_pred, dOpInf_red_model):
	"""
	solve_opinf_difference_model solves the discrete OpInf ROM for n_steps_pred over the target time horizon (training + prediction)

	:qhat0: 			reduced initial condition Qtilde0=np.matmul (Vr.T, q[:, 0]
	:n_steps_pred: 		number of steps over the target time horizon to solve the OpInf reduced model
	:dOpInf_red_model: 	dOpInf ROM

	:return: contains_nan flag indicating NaN presence in in the Qtilde_train reduced solution, Qtilde
	"""

	Qtilde    		= np.zeros((np.size(qhat0), n_steps_pred))
	contains_nans  	= False

	Qtilde[:, 0] = qhat0
	for i in range(n_steps_pred - 1):
	    Qtilde[:, i + 1] = dOpInf_red_model(Qtilde[:, i])

	if np.any(np.isnan(Qtilde)):
	    contains_nans = True

	return contains_nans, Qtilde.T