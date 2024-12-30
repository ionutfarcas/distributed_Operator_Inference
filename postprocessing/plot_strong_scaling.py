import numpy as np
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import Axes3D

import math

def truncate(number, digits) -> float:
    # Improve accuracy with floating point operations, to avoid truncate(16.4, 2) = 16.39 or truncate(-1.13, 2) = -1.12
    nbDecimals = len(str(number).split('.')[1]) 
    if nbDecimals <= digits:
        return number
    stepper = 10.0 ** digits

    return math.trunc(stepper * number) / stepper


def get_runtime_serial(file):

	data = np.load(file)

	data_loading_time 				= data[0]
	compute_time 					= data[1]
	compute_time_grid_search_total 	= data[2] 
	learning_time_grid_search_total = data[3]  
	OpInf_wtime_learning_opt 		= data[4]  
	OpInf_ROM_wtime_opt 			= data[5]

	return data_loading_time, compute_time, compute_time_grid_search_total, learning_time_grid_search_total, OpInf_wtime_learning_opt, OpInf_ROM_wtime_opt


def get_runtimes_distributed(file):

	data = np.load(file)

	data_loading_time 				= data[0]
	compute_time 					= data[1]
	communication_time 				= data[2] 
	compute_time_grid_search_total 	= data[3] 
	learning_time_grid_search_total = data[4]  
	OpInf_wtime_learning_opt 		= data[5]  
	OpInf_ROM_wtime_opt 			= data[6]

	return data_loading_time, compute_time, communication_time, compute_time_grid_search_total, learning_time_grid_search_total, OpInf_wtime_learning_opt, OpInf_ROM_wtime_opt

if __name__ == '__main__':

	#####################
	p_ss 		= np.array([1, 2, 4, 8])
	p_ss_labels = [str(p) for p in p_ss]

	n_reps = 100

	p_ss_labels = [str(p) for p in p_ss]


	data_loading_time_ss 	= np.zeros((len(p_ss), n_reps))
	compute_time_ss 		= np.zeros((len(p_ss), n_reps))
	communication_time_ss 	= np.zeros((len(p_ss), n_reps))
	compute_time_grid_ss 	= np.zeros((len(p_ss), n_reps))
	learning_time_grid_ss 	= np.zeros((len(p_ss), n_reps))

	OpInf_runtime = np.zeros((len(p_ss), n_reps)) 

	for i, p in enumerate(p_ss):

		
		for j in range(n_reps):
			if p == 1:
				file = 'runtimes/sOpInf_runtime_iteration_' + str(j + 1) + '.npy'
				data_loading_time_ss[i, j], compute_time_ss[i, j], compute_time_grid_ss[i, j], learning_time_grid_ss[i, j], _, OpInf_runtime[i, j] = get_runtime_serial(file)

			else:
				file = 'runtimes/dOpInf_runtime_using_' + str(p) + '_cores_' + str(j + 1) + '.npy'

				data_loading_time_ss[i, j], compute_time_ss[i, j], communication_time_ss[i, j], compute_time_grid_ss[i, j], learning_time_grid_ss[i, j], _, OpInf_runtime[i, j] = get_runtimes_distributed(file)

	
	data_loading_time_ss_mean 	= np.mean(data_loading_time_ss, axis=1)
	compute_time_ss_mean 		= np.mean(compute_time_ss, axis=1)
	communication_time_ss_mean 	= np.mean(communication_time_ss, axis=1)
	compute_time_grid_ss_mean 	= np.mean(compute_time_grid_ss, axis=1)


	print(data_loading_time_ss_mean)
	print(np.std(data_loading_time_ss, ddof=1, axis=1))
	print('***********')

	print(compute_time_ss_mean)
	print(np.std(compute_time_ss, ddof=1, axis=1))
	print('***********')

	print(communication_time_ss_mean)
	print(np.std(communication_time_ss, ddof=1, axis=1))
	print('***********')

	print(compute_time_grid_ss_mean)
	print(np.std(compute_time_grid_ss, ddof=1, axis=1))
	print('***********')

	print(np.mean(OpInf_runtime, axis=1))
	print(np.std(OpInf_runtime, ddof=1, axis=1))
	print('***********')


	total_runtime_ss = data_loading_time_ss_mean + compute_time_ss_mean + communication_time_ss_mean + compute_time_grid_ss_mean

	total_runtime_ss_all = data_loading_time_ss + compute_time_ss + communication_time_ss + compute_time_grid_ss

	print(total_runtime_ss)
	print(np.std(total_runtime_ss_all, ddof=1, axis=1))


	obtained_ss = np.zeros_like(total_runtime_ss)
	ideal_ss 	= np.zeros_like(total_runtime_ss)

	for i, Tp in enumerate(total_runtime_ss):

		obtained_ss[i] 	= total_runtime_ss[0]/total_runtime_ss[i]
		ideal_ss[i] 	= p_ss[i]/p_ss[0]


	perc_data_loading_time_ss_mean 	= 100*np.array([data_loading_time_ss_mean[i]/total_runtime_ss[i] for i in range(p_ss.shape[0])])
	perc_compute_time_ss_mean 		= 100*np.array([compute_time_ss_mean[i]/total_runtime_ss[i] for i in range(p_ss.shape[0])])
	perc_communication_time_ss_mean = 100*np.array([communication_time_ss_mean[i]/total_runtime_ss[i] for i in range(p_ss.shape[0])])
	perc_compute_time_grid_ss_mean 	= 100*np.array([compute_time_grid_ss_mean[i]/total_runtime_ss[i] for i in range(p_ss.shape[0])]) 

	data = np.vstack((perc_data_loading_time_ss_mean, perc_compute_time_ss_mean, perc_communication_time_ss_mean, perc_compute_time_grid_ss_mean))

	print(ideal_ss)
	print(obtained_ss)

	print(data_loading_time_ss)
	print(compute_time_ss)
	print(communication_time_ss)
	print(total_runtime_ss[0], total_runtime_ss[-1])

	print('*****************************')	
	############################


	fontsize=12

	rc("figure", dpi=400)           # High-quality figure ("dots-per-inch")
	rc("text", usetex=True)         # Crisp axis ticks
	rc("font", family="sans-serif")      # Crisp axis labels
	rc("legend", edgecolor='none')  # No boxes around legends
	rc('text.latex', preamble=r'\usepackage{amsfonts}')
	rcParams["figure.figsize"] = (9, 4)
	rcParams.update({'font.size': fontsize})
	rcParams["figure.autolayout"] = True
	# rcParams['figure.constrained_layout.use'] = True

	rcParams['grid.color'] = (0.5, 0.5, 0.5, 0.2)

	charcoal    = [0.1, 0.1, 0.1]
	colors = ['#41b6c4', '#1d91c0', '#225ea8', '#253494', '#081d58']
	color1 = '#009E73'
	color2 = '#D55E00'
	color3 = '#0072B2'


	fig 	= figure()
	ax11 	= fig.add_subplot(121)
	ax12 	= fig.add_subplot(122)


	ax11.spines['right'].set_visible(False)
	ax11.spines['top'].set_visible(False)
	ax11.yaxis.set_ticks_position('left')
	ax11.xaxis.set_ticks_position('bottom')

	ax12.spines['right'].set_visible(False)
	ax12.spines['top'].set_visible(False)
	ax12.yaxis.set_ticks_position('left')
	ax12.xaxis.set_ticks_position('bottom')


	##############
	p1 = ax11.loglog(p_ss, ideal_ss, linestyle='-', marker='o', lw=1.05, ms=3, color=charcoal)
	p2 = ax11.loglog(p_ss, obtained_ss, linestyle='--', marker='*', lw=1.05, ms=5, color=color2)

	for i, p in enumerate(p_ss):
		if i == 0:
			# ax11.text(0.80*p, obtained_ss[i]*1.07, ideal_ss[i], color=charcoal, rotation=20)
			# ax11.text(1.00*p, obtained_ss[i]*0.65, truncate(obtained_ss[i], 2), color=color2, rotation=20)
			pass

		elif i == 3:
			ax11.text(0.88*p, ideal_ss[i]*1.0, ideal_ss[i], color=charcoal, rotation=39)
			ax11.text(0.95*p, obtained_ss[i]*0.80, truncate(obtained_ss[i], 2), color=color2, rotation=12)

		else:
			ax11.text(0.88*p, ideal_ss[i]*1.0, ideal_ss[i], color=charcoal, rotation=39)
			ax11.text(0.95*p, obtained_ss[i]*0.80, truncate(obtained_ss[i], 2), color=color2, rotation=30)

	ax11.text(2, 2.9, 'ideal speed-up', color=charcoal, rotation=38)
	ax11.text(2, 1.3, 'obtained speed-up', color=color2, rotation=30)	


	fig.supxlabel('number of compute cores ' + r'$p$')
	ax11.set_xticks(p_ss)
	ax11.set_xticklabels(p_ss_labels)

	ax11.set_ylabel('speed-up')
	ax11.grid(True, linestyle='--', lw=0.3)

	ax11.set_xticks([1, 2, 3, 4, 6, 8])
	ax11.set_xticklabels([1, 2, '', 4, '', 8])

	ax11.set_yticks([1, 2, 3, 4, 6, 8])
	ax11.set_yticklabels([1, 2, '', 4, '', 8])

	ax11.set_xlim([0.9, 8.2])
	ax11.set_ylim([0.9, 8.2])

	# ax11.set_yticks([1, 2, 4, 8, 16, 32, 64])
	# ax11.set_yticklabels([1, 2, 4, 8, 16, 32, 64])
	###################
	

	

	##################
	color1 = '#a6cee3'
	color3 = '#1f78b4'
	color2 = '#b2df8a'
	color4 = '#33a02c'

	colors = [color1, color2, color3, color4]

	ny 	= len(data[0])
	ind = list(range(ny))

	axes 		= []
	cum_size 	= np.zeros(ny)

	axes 		= []
	cum_size 	= np.zeros(ny)

	ax12.set_yscale('log')
	ax12.set_ylim((5e0,100))

	bars = []

	for i, row_data in enumerate(data):
		bars.append(ax12.bar(ind, row_data, bottom=cum_size, label='i',
		                    color=colors[i]))

		axes.append(ax12.bar(ind, row_data, bottom=cum_size, label='i',
		                    color=colors[i]))
		cum_size += row_data

	for axis in axes:
	    for bar in axis:
	        w, h = bar.get_width(), bar.get_height()
	        if h > 0:
	            string = r'$\mathsf {{{:.2f}}} $ \%'.format(h)

	            ax12.text(bar.get_x() + w/2, bar.get_y() + h/2,
	                     string, ha="center",
	                     va="center", weight='bold')

	ax12.set_ylabel( 'percentage CPU time ' + '(' + r'$\%$' + ')')



	y_pos_all   = np.array([100, 70, 30, 18, 5])
	labels      = [r'$100$', r'$70$', r'$30$', r'$20$', r'$5$']

	ax12.set_yticks(y_pos_all)
	ax12.set_yticklabels(labels)

	x_pos_all   = np.array([0, 1, 2, 3])
	labels      = np.array([1, 2, 4, 5])

	ax12.set_xticks(x_pos_all)
	ax12.set_xticklabels(labels)

	ax12.legend((bars[0][0], bars[1][0], bars[2][0], bars[3][0]), \
				('I/O', 'computations', 'communication', 'dOpInf learning'), loc=(0.05, 1.01), ncol=2, frameon=True, shadow=False, edgecolor='white')

	##################


	tight_layout()

	fig_name = 'figures/NS_example_strong_scaling.png'

	savefig(fig_name)
	close()