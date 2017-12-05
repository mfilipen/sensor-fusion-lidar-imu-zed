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

def printToFile(f,t,data1,data2):
    for i in range(len(data1)):
        f.write("{:.9f} {:.9f} {:.9f}\n".format(t[i],data1[i],data2[i]))

# get an instance of RosPack with the default search paths
rospack = rospkg.RosPack()

# get the file path for sensor_fusion
rospack.get_path('sensor_fusion')

path=rospack.get_path('sensor_fusion')+'/dataTxt/sis/'

data_pose = np.loadtxt(path+"pose_position_test.txt", delimiter=' ', dtype=np.float)
data_pose_tra = np.loadtxt(path+"pose_position_training.txt", delimiter=' ', dtype=np.float)


colors = ['red', 'blue', 'green']

t_pose = data_pose[:,0]
x_pose = data_pose[:,1]
y_pose = data_pose[:,2]

t_pose_tra = data_pose_tra[:,0]
x_pose_tra = data_pose_tra[:,1]
y_pose_tra = data_pose_tra[:,2]

x_pose-=x_pose[0]
y_pose-=y_pose[0]


f= open(path+'pose_position_at_zero_test.txt', 'w')
printToFile(f,t_pose,x_pose,y_pose)

f_plot(t_pose, x_pose, colors=colors, linewidth=2.)
f_plot(t_pose, y_pose,colors=colors, linewidth=2.)
f_plot(x_pose, y_pose, colors=colors, linewidth=2.)

f_plot(t_pose_tra, x_pose_tra, colors=colors, linewidth=2.)
f_plot(t_pose_tra, y_pose_tra,colors=colors, linewidth=2.)
f_plot(x_pose_tra, y_pose_tra, colors=colors, linewidth=2.)

plt.show()

