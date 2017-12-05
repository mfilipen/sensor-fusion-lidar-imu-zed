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


def printToFile(f,t,data):
    for i in range(len(data)):
        f.write("{:.9f} {:.9f}\n".format(t[i],data[i]))

# get an instance of RosPack with the default search paths
rospack = rospkg.RosPack()

# get the file path for sensor_fusion
rospack.get_path('sensor_fusion')

path=rospack.get_path('sensor_fusion')+'/dataTxt/sis/'

data_lidar_odom = np.loadtxt(path+"pose_orientation_test.txt", delimiter=' ', dtype=np.float)
data_lidar_odom_tra = np.loadtxt(path+"pose_orientation_training.txt", delimiter=' ', dtype=np.float)

colors = ['red', 'blue', 'green', 'yellow','purple', 'brown']

t_pose_orientation = data_lidar_odom[:,0]
yaw_pose_orientation = data_lidar_odom[:,1]

t_pose_orientation_tra = data_lidar_odom_tra[:,0]
yaw_pose_orientation_tra = data_lidar_odom_tra[:,1]


f_plot(t_pose_orientation,yaw_pose_orientation,
       colors=colors, linewidth=2.)

f_plot(t_pose_orientation_tra,yaw_pose_orientation_tra,
       colors=colors, linewidth=2.)
plt.show()