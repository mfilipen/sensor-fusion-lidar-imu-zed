#!/usr/bin/env python

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import rospkg
from math import *

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

def denormalize(data):
    repeat=True

    while (repeat):
        repeat=False
        for i in range(len(data) - 1):
            if (data[i + 1] - data[i] > 2):
                repeat = True
                for j in range(len(data)):
                    if j > i:
                        data[j] -= 2 * 3.14159265

    repeat = True
    while (repeat):
        repeat = False
        for i in range(len(data) - 1):
            if (data[i + 1] - data[i] < -2):
                repeat = True
                for j in range(len(data)):
                    if j > i:
                        data[j] += 2 * 3.14159265

# get an instance of RosPack with the default search paths
rospack = rospkg.RosPack()

# get the file path for sensor_fusion
rospack.get_path('sensor_fusion')

path=rospack.get_path('sensor_fusion')+'/dataTxt/laps/'

def printToFile2(f,t,x,y, yaw):
    for i in range(len(t)):
        f.write("{:.9f} {:.9f} {:.9f} {:.9f}\n".format(t[i],x[i],y[i], yaw[i]))


data_pose = np.loadtxt(path+"ZED_pose_position.txt", delimiter=' ', dtype=np.float)
data_orie = np.loadtxt(path + "ZED_pose_orientation.txt", delimiter=' ', dtype=np.float)
colors = ['red', 'blue', 'green']

t_pose = data_pose[:,0]
x_pose = data_pose[:,1]
y_pose = data_pose[:,2]
yaw_pose = data_orie[:, 1]


for i in range(len(t_pose)):
    x_pose[i] += -0.295*cos(yaw_pose[i])
    y_pose[i] += -0.295*sin(yaw_pose[i])

x_pose+=0.295



f_plot(t_pose, x_pose, colors=colors, linewidth=2.)
f_plot(t_pose, y_pose,colors=colors, linewidth=2.)
f_plot(x_pose, y_pose, colors=colors, linewidth=2.)

path+="prepared/"
f = open(path+'zed_pose.txt', 'w')
denormalize(yaw_pose)
printToFile2(f,t_pose,x_pose, y_pose, yaw_pose)

plt.show()

