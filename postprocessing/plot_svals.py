import numpy as np 
import matplotlib as mpl 
from matplotlib.lines import Line2D	
mpl.use('TkAgg')
from matplotlib.pyplot import *

if __name__ == '__main__':

	svals = np.load('dOpInf_postprocessing/Sigma_sq_global.npy')

	no_kept_svals_global 	= 300
	no_kept_svals_energy 	= 30
	no_svals_global 		= range(1, no_kept_svals_global + 1)
	no_svals_energy 		= range(1, no_kept_svals_energy + 1)

	retained_energy 	= np.cumsum(svals)/np.sum(svals)
	target_ret_energy 	= 0.9996

	r 			= np.argmax(retained_energy > target_ret_energy) + 1	
	ret_energy 	= retained_energy[r]
	
	rcParams['lines.linewidth'] = 0
	rc("figure", dpi=400)
	rc("font", family="serif")
	rc("legend", edgecolor='none')
	rcParams["figure.figsize"] = (7, 3)
	rcParams.update({'font.size': 8})

	charcoal    = [0.1, 0.1, 0.1]
	color1      = '#D55E00'
	color2      = '#0072B2'

	fig 		= figure()
	ax1 		= fig.add_subplot(121)
	ax2 		= fig.add_subplot(122)
	
	rc("figure",facecolor='w')
	rc("axes",facecolor='w',edgecolor='k',labelcolor='k')
	rc("savefig",facecolor='w')
	rc("text",color='k')
	rc("xtick",color='k')
	rc("ytick",color='k')

	ax1.spines['right'].set_visible(False)
	ax1.spines['top'].set_visible(False)
	ax1.yaxis.set_ticks_position('left')
	ax1.xaxis.set_ticks_position('bottom')

	ax2.spines['right'].set_visible(False)
	ax2.spines['top'].set_visible(False)
	ax2.yaxis.set_ticks_position('left')
	ax2.xaxis.set_ticks_position('bottom')

	## plot
	ax1.semilogy(no_svals_global, np.sqrt(svals)[:no_kept_svals_global]/np.sqrt(svals[0]), linestyle='-', lw=1.25, color=color1)
	ax1.set_xlabel('index')
	ax1.set_ylabel('singular values transformed data')
	

	ax2.plot(no_svals_energy, retained_energy[:no_kept_svals_energy], linestyle='-', lw=1.25, color=color1)
	ax2.set_xlabel('reduced dimension')
	ax2.set_ylabel('% energy retained')	
	ax2.plot([r, r], [0, retained_energy[r]], linestyle='--', lw=0.5, color=charcoal)
	ax2.plot([0, r], [retained_energy[r], retained_energy[r]], linestyle='--', lw=0.5, color=charcoal)
	##

	##
	xlim = ax1.get_xlim()
	# ax1.set_xlim([0, training_end])
	ax1.set_ylim([1e-10, 1.02e0])

	x_pos_all 	= np.array([0, 99, 199, 299])
	labels 		= np.array([1, 100, 200, 300])
	ax1.set_xticks(x_pos_all)
	ax1.set_xticklabels(labels)

	ax2.set_xlim([0, 30])
	ax2.set_ylim([0.4, 1.001])

	ax2.set_xticks([1, r, 20, 30])
	ax2.set_yticks([0.5, 0.75, ret_energy , 1])
	ax2.set_yticklabels([r'$50\%$', r'$75\%$', r'$99.98\%$', ''])
	###

	tight_layout()

	savefig('figures/NS_example_POD_svals_and_ret_energy.png', pad_inches=3)
	close()