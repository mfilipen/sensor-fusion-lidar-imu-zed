#!/usr/bin/env python

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import rospkg
from math import *
from scipy.interpolate import interp1d

import rospy
from tf.transformations import euler_from_quaternion

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

def printToFile(f,t,x,y,roll, pitch, yaw):
    for i in range(len(t)):
        f.write("{:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f}\n".format(t[i],x[i],y[i],roll[i], pitch[i], yaw[i]))

def printToFile2(f,t,x,y, yaw):
    for i in range(len(t)):
        f.write("{:.9f} {:.9f} {:.9f} {:.9f}\n".format(t[i],x[i],y[i], yaw[i]))

def orbDataSpliting(x_pose_orb, y_pose_orb ,yaw_orb):
    k=0;
    for i in range(len(x_pose_orb)):
        if x_pose_orb[i] == 0.000000000:
            k+=1
    print(k)

    segment = np.zeros(shape=(k), dtype=np.integer)

    k = 0
    for i in range(len(x_pose_orb)):
        if x_pose_orb[i] == 0.000000000:
            segment[k]=i
            k+=1

    return segment

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

path = rospack.get_path('sensor_fusion') + '/dataTxt/laps/'

data_pose_orb = np.loadtxt(path + "orb_slam_pose.txt", delimiter=' ', dtype=np.float)
data_pose_lidar = np.loadtxt(path + "pose_position.txt", delimiter=' ', dtype=np.float)
data_lidar_odom = np.loadtxt(path + "pose_orientation.txt", delimiter=' ', dtype=np.float)
colors = ['red', 'blue', 'green', 'yellow','purple', 'brown']

t_pose_lidar = data_pose_lidar[:, 0]
x_pose_lidar = data_pose_lidar[:, 1]
y_pose_lidar = data_pose_lidar[:, 2]
yaw_pose_lidar = data_lidar_odom[:, 1]

t_pose_orb = data_pose_orb[:, 0]
x_pose_orb = data_pose_orb[:, 3]
y_pose_orb = data_pose_orb[:, 1]
z_pose_orb = data_pose_orb[:, 2]
quaternion = data_pose_orb[:, 4:8]

roll_orb  =  np.zeros(t_pose_orb .shape, np.float)
pitch_orb  = np.zeros(t_pose_orb .shape, np.float)
yaw_orb  = np.zeros(t_pose_orb .shape, np.float)

for i in range(len(yaw_orb)):
    (roll_orb [i], pitch_orb [i], yaw_orb [i]) = euler_from_quaternion(quaternion[i])

f_ORB_SLAM = open(path+'ORB_SLAM_x_y_yaw.txt', 'w')

printToFile(f_ORB_SLAM,t_pose_orb , x_pose_orb , y_pose_orb,roll_orb , pitch_orb , yaw_orb )



'''
f_plot(t_pose_orb, x_pose_orb, t_pose_orb, y_pose_orb, t_pose_orb, z_pose_orb, colors=colors, linewidth=2.)

f_plot(t_pose, roll, colors=colors, linewidth=2.)
f_plot(t_pose, pitch, colors=colors, linewidth=2.)
f_plot(t_pose, yaw, colors=colors, linewidth=2.)

f_plot(x_pose, y_pose, colors=colors, linewidth=2.)
plt.show()

f_plot(t_pose_orb, roll_orb, colors=colors, linewidth=2.)
f_plot(t_pose_orb, pitch_orb, colors=colors, linewidth=2.)
f_plot(t_pose_orb, yaw_orb, colors=colors, linewidth=2.)


f_plot(t_pose_orb, x_pose_orb, colors=colors, linewidth=2.)
f_plot(t_pose_orb, y_pose_orb, colors=colors, linewidth=2.)
f_plot(x_pose_orb, y_pose_orb, colors=colors, linewidth=2.)
f_plot(t_pose_orb, pitch_orb, colors=colors, linewidth=2.)
'''
segments=orbDataSpliting(x_pose_orb, y_pose_orb ,yaw_orb)

f_x_t = interp1d(t_pose_lidar, x_pose_lidar)
f_y_t = interp1d(t_pose_lidar, y_pose_lidar)
f_yaw_t = interp1d(t_pose_lidar, yaw_pose_lidar)

pi_half=1.57079632679

y_pose_orb=-y_pose_orb

for i in range(len(segments)-1):
    x_shift=f_x_t(t_pose_orb[segments[i]])
    y_shift=f_y_t(t_pose_orb[segments[i]])
    yaw_shift=f_yaw_t(t_pose_orb[segments[i]])


    yaw = yaw_shift*(-1)
    for j in range(segments[i],segments[i+1]):
        pitch_orb[j] = yaw_shift - pitch_orb[j]

        x = x_shift + 0.295*cos(pitch_orb[j]) + cos(yaw) * x_pose_orb[j] + sin(yaw) * y_pose_orb[j]
        y = y_shift + 0.295*sin(pitch_orb[j]) + sin(yaw)*(-1)* x_pose_orb[j] + cos(yaw) * y_pose_orb[j]

        x_pose_orb[j]=x
        y_pose_orb[j]=y

        '''
        x_pose_orb[j] += x_shift
        y_pose_orb[j] += y_shift
        '''

x_pose_orb-=0.295
f_plot(t_pose_orb, x_pose_orb,t_pose_lidar, x_pose_lidar,t_pose_orb, y_pose_orb,t_pose_lidar, y_pose_lidar, colors=colors, linewidth=2.)

'''
f_plot(t_pose_orb, pitch_orb, t_pose_lidar, yaw_pose_lidar, colors=colors, linewidth=2.)
f_plot(x_pose_orb, y_pose_orb, x_pose_lidar,y_pose_lidar, colors=colors, linewidth=2.)
for j in range(3):
    for i in range(len(pitch_orb)):
        if (pitch_orb[i]>3.14159265):
            pitch_orb[i]-=2*3.14159265

f_plot(t_pose_orb, x_pose_orb, colors=colors, linewidth=2.)
f_plot(t_pose_orb, y_pose_orb, colors=colors, linewidth=2.)
f_plot(x_pose_orb, y_pose_orb, colors=colors, linewidth=2.)
f_plot(t_pose_orb, pitch_orb, colors=colors, linewidth=2.)
'''

denormalize(pitch_orb)
f_plot(t_pose_orb, pitch_orb, colors=colors, linewidth=2.)

path+="prepared/"
f = open(path+'orb_pose.txt', 'w')
printToFile2(f,t_pose_orb,x_pose_orb, y_pose_orb, pitch_orb)


plt.show()
