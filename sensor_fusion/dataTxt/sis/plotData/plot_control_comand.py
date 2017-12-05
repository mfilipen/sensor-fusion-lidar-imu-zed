#!/usr/bin/env python

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import rospkg


def f_plot(*args, **kwargs):
    xlist = []
    ylist = []
    for i, arg in enumerate(args):
        if (i % 2 == 0):
            xlist.append(arg)
        else:
            ylist.append(arg)

    colors = kwargs.pop('colors', 'k')
    linewidth = kwargs.pop('linewidth', 1.)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    i = 0
    for x, y, color in zip(xlist, ylist, colors):
        i += 1
        ax.plot(x, y, color=color, linewidth=linewidth, label=str(i))

    ax.grid(True)
    ax.legend()

# get an instance of RosPack with the default search paths
rospack = rospkg.RosPack()

# get the file path for sensor_fusion
rospack.get_path('sensor_fusion')

path=rospack.get_path('sensor_fusion')+'/dataTxt/sis/'

wz_comand= np.loadtxt(path+'wz_comand_interpolation_test.txt', delimiter=' ', dtype=np.float)
vx_comand= np.loadtxt(path+'vx_comand_interpolation_test.txt', delimiter=' ', dtype=np.float)

wz_comand_tra= np.loadtxt(path+'wz_comand_interpolation_training.txt', delimiter=' ', dtype=np.float)
vx_comand_tra= np.loadtxt(path+'vx_comand_interpolation_training.txt', delimiter=' ', dtype=np.float)
colors = ['red', 'blue', 'green']


t_wz = wz_comand[:,0]
wz = wz_comand[:,1]

t_vx = vx_comand[:,0]
vx = vx_comand[:,1]

t_wz_tra = wz_comand_tra[:,0]
wz_tra = wz_comand_tra[:,1]

t_vx_tra = vx_comand_tra[:,0]
vx_tra = vx_comand_tra[:,1]


f_plot(t_wz, wz, colors=colors, linewidth=2.)
f_plot(t_vx, vx, colors=colors, linewidth=2.)
f_plot(t_wz_tra, wz_tra, colors=colors, linewidth=2.)
f_plot(t_vx_tra, vx_tra, colors=colors, linewidth=2.)

plt.show()

