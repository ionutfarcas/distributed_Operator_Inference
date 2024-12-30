import numpy as np 
import matplotlib as mpl 
from matplotlib.lines import Line2D	
mpl.use('TkAgg')
from matplotlib.pyplot import *

if __name__ == '__main__':

	t_start = 4
	t_end 	= 10
	t_train = 7
	dt 		= 5e-3 

	t = np.arange(t_start, t_end, dt)

	ref_data_var1_loc1 = np.load('ref_data/ref_probe_1_var_1.npy')
	ref_data_var1_loc2 = np.load('ref_data/ref_probe_2_var_1.npy')
	ref_data_var1_loc3 = np.load('ref_data/ref_probe_3_var_1.npy')

	ref_data_var2_loc1 = np.load('ref_data/ref_probe_1_var_2.npy')
	ref_data_var2_loc2 = np.load('ref_data/ref_probe_2_var_2.npy')
	ref_data_var2_loc3 = np.load('ref_data/ref_probe_3_var_2.npy')
	
	dOpInf_data_var1_loc1 = np.load('dOpInf_postprocessing/dOpInf_probe_1_var_1.npy')
	dOpInf_data_var1_loc2 = np.load('dOpInf_postprocessing/dOpInf_probe_2_var_1.npy')
	dOpInf_data_var1_loc3 = np.load('dOpInf_postprocessing/dOpInf_probe_3_var_1.npy')
	
	dOpInf_data_var2_loc1 = np.load('dOpInf_postprocessing/dOpInf_probe_1_var_2.npy')
	dOpInf_data_var2_loc2 = np.load('dOpInf_postprocessing/dOpInf_probe_2_var_2.npy')
	dOpInf_data_var2_loc3 = np.load('dOpInf_postprocessing/dOpInf_probe_3_var_2.npy')


	rcParams['lines.linewidth'] = 0
	rc("figure", dpi=400)           # High-quality figure ("dots-per-inch")
	# rc("text", usetex=True)         # Crisp ax1is ticks
	rc("font", family="serif")      # Crisp ax1is labels
	rc("legend", edgecolor='none')  # No boxes around legends
	rcParams["figure.figsize"] = (8, 4)
	rcParams.update({'font.size': 8})

	charcoal    = [0.1, 0.1, 0.1]
	color1      = '#D55E00'
	color2      = '#0072B2'

	fig 		= figure()
	ax11 		= fig.add_subplot(231)
	ax12 		= fig.add_subplot(232, sharex=ax11, sharey=ax11)
	ax13 		= fig.add_subplot(233, sharex=ax11, sharey=ax11)
	ax21 		= fig.add_subplot(234)
	ax22 		= fig.add_subplot(235, sharex=ax21, sharey=ax21)
	ax23 		= fig.add_subplot(236, sharex=ax21, sharey=ax21)
	
	rc("figure",facecolor='w')
	rc("axes",facecolor='w',edgecolor='k',labelcolor='k')
	rc("savefig",facecolor='w')
	rc("text",color='k')
	rc("xtick",color='k')
	rc("ytick",color='k')


	ax11.spines['right'].set_visible(False)
	ax11.spines['top'].set_visible(False)
	ax11.yaxis.set_ticks_position('left')
	ax11.xaxis.set_ticks_position('bottom')

	ax12.spines['right'].set_visible(False)
	ax12.spines['top'].set_visible(False)
	ax12.yaxis.set_ticks_position('left')
	ax12.xaxis.set_ticks_position('bottom')

	ax21.spines['right'].set_visible(False)
	ax21.spines['top'].set_visible(False)
	ax21.yaxis.set_ticks_position('left')
	ax21.xaxis.set_ticks_position('bottom')

	ax22.spines['right'].set_visible(False)
	ax22.spines['top'].set_visible(False)
	ax22.yaxis.set_ticks_position('left')
	ax22.xaxis.set_ticks_position('bottom')

	ax13.spines['right'].set_visible(False)
	ax13.spines['top'].set_visible(False)
	ax13.yaxis.set_ticks_position('left')
	ax13.xaxis.set_ticks_position('bottom')

	ax23.spines['right'].set_visible(False)
	ax23.spines['top'].set_visible(False)
	ax23.yaxis.set_ticks_position('left')
	ax23.xaxis.set_ticks_position('bottom')

	
	## plot
	ax11.plot(t, ref_data_var1_loc1, linestyle='-', lw=1.00, color=charcoal)
	ax11.plot(t, dOpInf_data_var1_loc1, linestyle='--', lw=1.00, color=color1)

	ax12.plot(t, ref_data_var1_loc2, linestyle='-', lw=1.00, color=charcoal)
	ax12.plot(t, dOpInf_data_var1_loc2, linestyle='--', lw=1.00, color=color1)

	ax13.plot(t, ref_data_var1_loc3, linestyle='-', lw=1.00, color=charcoal)
	ax13.plot(t, dOpInf_data_var1_loc3, linestyle='--', lw=1.00, color=color1)
	
	
	ax21.plot(t, ref_data_var2_loc1, linestyle='-', lw=1.00, color=charcoal)
	ax21.plot(t, dOpInf_data_var2_loc1, linestyle='--', lw=1.00, color=color1)

	ax22.plot(t, ref_data_var2_loc2, linestyle='-', lw=1.00, color=charcoal)
	ax22.plot(t, dOpInf_data_var2_loc2, linestyle='--', lw=1.00, color=color1)

	ax23.plot(t, ref_data_var2_loc3, linestyle='-', lw=1.00, color=charcoal)
	ax23.plot(t, dOpInf_data_var2_loc3, linestyle='--', lw=1.00, color=color1)
	##

	ax11.axvline(x=t_train, lw=1.00, linestyle='--', color='gray')
	ax12.axvline(x=t_train, lw=1.00, linestyle='--', color='gray')
	ax13.axvline(x=t_train, lw=1.00, linestyle='--', color='gray')
	ax21.axvline(x=t_train, lw=1.00, linestyle='--', color='gray')
	ax22.axvline(x=t_train, lw=1.00, linestyle='--', color='gray')
	ax23.axvline(x=t_train, lw=1.00, linestyle='--', color='gray')

	ax11.set_title('probe 1 (0.4, 0.2)')
	ax12.set_title('probe 2 (0.6, 0.2)')
	ax13.set_title('probe 3 (1.0, 0.2)')

	ax11.set_ylabel(r'$v_x$')
	ax21.set_ylabel(r'$v_y$')

	fig.supxlabel('target time horizon (seconds)')



	# cosmetics
	xlim = ax11.get_xlim()
	ax11.set_xlim([xlim[0], 10])
	ax21.set_xlim([xlim[0], 10])

	ylim = ax11.get_ylim()
	ax11.set_ylim([ylim[0], 1.25])

	ylim = ax21.get_ylim()
	ax21.set_ylim([ylim[0], 0.4])

	rect = Rectangle((0, 0), width=t_train, height=1.25, hatch='/', color='grey', alpha=0.2, label='training region')
	ax11.add_patch(rect)
	rect = Rectangle((0, 0), width=t_train, height=1.25, hatch='/', color='grey', alpha=0.2, label='training region')
	ax12.add_patch(rect)
	rect = Rectangle((0, 0), width=t_train, height=1.25, hatch='/', color='grey', alpha=0.2, label='training region')
	ax13.add_patch(rect)


	rect = Rectangle((0, -0.4), width=t_train, height=0.8, hatch='/', color='grey', alpha=0.2, label='training region')
	ax21.add_patch(rect)
	rect = Rectangle((0, -0.4), width=t_train, height=0.8, hatch='/', color='grey', alpha=0.2, label='training region')
	ax22.add_patch(rect)
	rect = Rectangle((0, -0.4), width=t_train, height=0.8, hatch='/', color='grey', alpha=0.2, label='training region')
	ax23.add_patch(rect)
	
	x_pos_all 	= np.array([4, 5, 6, 7, 8, 9, 10])
	labels 		= np.array([4, 5, 6, 7, 8, 9, 10])
	ax11.set_xticks(x_pos_all)
	ax11.set_xticklabels(labels)
	ax21.set_xticks(x_pos_all)
	ax21.set_xticklabels(labels)

	tight_layout()

	savefig('figures/NS_example_probes.png', pad_inches=3)
	close()