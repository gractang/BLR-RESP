import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

def initialize_plot(ax, x_label, obj_name):
    font = 'Times New Roman'
    fs = 16
    ax.set_xlabel(x_label, fontname=font, fontsize=fs)
    ax.tick_params(direction='in')
    ax.set_xlim(-10000, 10000)
    plt.yticks([0], fontname=font, fontsize=fs)
    plt.xticks(fontname=font, fontsize=fs)
#    plt.xticks(np.arange(-10000, 10000+1, 5000))
    # ax.get_yaxis().set_visible(False)
    ax.text(.15, .8, obj_name, horizontalalignment='center', transform=ax.transAxes, fontsize=18, fontname=font)
    ax.axhline(0, color ='black', lw=1)
    ax.axvline(0, color ='black', lw=1)
    
    
rcParams.update({'figure.autolayout': True})    
obj_name = '1100+17'
obj = obj_name[0:4]
save_path = obj + '_fit_Grace.eps'
BLR_RESP_x = np.loadtxt('velocities.txt')
unobsc = np.loadtxt(obj + '_unobsc_prof.txt')
obsc = np.loadtxt(obj + '_obsc_prof.txt')
rel_v, obsvd = np.loadtxt(obj + '_obsvd_prof.txt', unpack = True)

ax = plt.axes()
initialize_plot(ax, 'Relative Velocity (km/s)', obj_name)
ax.plot(rel_v, obsvd, 'black')
ax.plot(BLR_RESP_x, unobsc, 'blue')
ax.plot(BLR_RESP_x, obsc, 'red')
plt.savefig(save_path, format='eps')