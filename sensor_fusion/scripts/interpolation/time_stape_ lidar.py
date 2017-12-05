#!/usr/bin/env python

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import rospkg
from scipy.interpolate import interp1d


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

def prepareLatice(x, xnew):
    start = 0
    end = 0
    for i in range(len(xnew) - 1):
        if xnew[i] <= x[0]:
            start += 1

        if x[len(x)-1] <= xnew[i]:
            end += 1
    print(end.__str__() + ' ' + start.__str__() +"/n")

    return_x= xnew[(start + 1):(len(xnew) - end - 1)]
    return return_x


# get an instance of RosPack with the default search paths
rospack = rospkg.RosPack()

# get the file path for sensor_fusion
rospack.get_path('sensor_fusion')

path = rospack.get_path('sensor_fusion') + '/dataTxt/laps/'

data_pose = np.loadtxt(path + "pose_position.txt", delimiter=' ', dtype=np.float)
wz_comand = np.loadtxt(path + 'wz_comand.txt', delimiter=' ', dtype=np.float)
vx_comand = np.loadtxt(path + 'vx_comand.txt', delimiter=' ', dtype=np.float)

colors = ['red', 'blue', 'green']

t_pose = data_pose[:, 0]
n_t_pose = range(len(t_pose))

t_wz = wz_comand[:, 0]
wz = wz_comand[:, 1]

t_vx = vx_comand[:, 0]
vx = vx_comand[:, 1]

# f_plot(n_t_pose, t_pose, colors=colors, linewidth=2.)


x = t_wz
y = wz
f = interp1d(x, y)

xnew = t_pose

plt.plot(x, y, '-', xnew, f(xnew), 'o')
plt.legend(['wz(t)', 'wz(t)-linear interpolation'], loc='best')
plt.show()

f1 = open(path+'wz_comand_interpolation.txt', 'w')
printToFile(f1,xnew,f(xnew))

x = t_vx
y = vx
f = interp1d(x, y)

xnew = t_pose

plt.plot(x, y, '-', xnew, f(xnew), 'o')
plt.legend(['vx(t)', 'vx(t)-linear interpolation'], loc='best')
plt.show()

f2 = open(path+'vx_comand_interpolation.txt', 'w')
printToFile(f2,xnew,f(xnew))
