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
    list = kwargs.pop('list', 'k')
    linewidth = kwargs.pop('linewidth', 1.)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    i = 0
    for x, y, color in zip(xlist, ylist, colors):

        ax.plot(x, y, color=color, linewidth=linewidth, label=list[i])
        i += 1
    ax.grid(True)
    ax.legend()

# get an instance of RosPack with the default search paths
rospack = rospkg.RosPack()

# get the file path for sensor_fusion
rospack.get_path('sensor_fusion')

path=rospack.get_path('sensor_fusion')+'/dataTxt/laps/'

wz_comand= np.loadtxt(path+'wz_comand.txt', delimiter=' ', dtype=np.float)
vx_comand= np.loadtxt(path+'vx_comand.txt', delimiter=' ', dtype=np.float)
colors = ['red', 'blue', 'green']


t_wz = wz_comand[:,0]
wz = wz_comand[:,1]

t_vx = vx_comand[:,0]
vx = vx_comand[:,1]

iwz = np.zeros(wz.shape,np.float)
for i in range(len(wz)-1):
    i+=1
    iwz[i]=(wz[i-1]+wz[i])*(t_wz[i]-t_wz[i-1])/2

wz_yaw = np.zeros(iwz.shape,np.float)
for i in range(len(iwz)):
    for j in range(i):
        wz_yaw[i]+=iwz[j]

name_vx = ['vx(t)']
name_wz = ['wz(t)']

f_plot(t_wz, wz , colors=colors, list=name_wz, linewidth=2.)
f_plot(t_wz, wz_yaw , colors=colors, linewidth=2.)

for j in range(5):
    for i in range(len(wz_yaw)):
        if (wz_yaw[i]>3.14159265):
            wz_yaw[i]-=2*3.14159265




f_plot(t_wz, wz_yaw, colors=colors, list=name_wz, linewidth=2.)

f_plot(t_vx, vx, colors=colors, list=name_vx, linewidth=2.)

plt.show()

