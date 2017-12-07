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

def printToFile2(f,t,x):
    for i in range(len(t)):
        f.write("{:.9f} {:.9f}\n".format(t[i],x[i]))


def denormalize(data):
    repeat = True

    while (repeat):
        repeat = False
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

data_acc = np.loadtxt(path+"gyro.txt", delimiter=' ', dtype=np.float)
colors = ['red', 'blue', 'green']

t_acc = data_acc[:,0]
x_acc = data_acc[:,1]
y_acc = data_acc[:,2]
z_acc = data_acc[:,3]


iz = np.zeros(z_acc.shape,np.float)
for i in range(len(z_acc)-1):
    i+=1
    iz[i]=(z_acc[i-1]+z_acc[i])*(t_acc[i]-t_acc[i-1])/2

z = np.zeros(iz.shape,np.float)
for i in range(len(iz)):
    for j in range(i):
        z[i]+=iz[j]


f_plot(t_acc, z_acc , colors=colors, linewidth=2.)
f_plot(t_acc, z , colors=colors, linewidth=2.)

path+="prepared/"
f = open(path+'gyro_yaw.txt', 'w')
denormalize(z)
printToFile2(f,t_acc,z)

for j in range(3):
    for i in range(len(z)):
        if (z[i]>3.14159265):
            z[i]-=2*3.14159265

f_plot(t_acc, z , colors=colors, linewidth=2.)

plt.show()

