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

def printToFile(f,t,data):
    for i in range(len(data)):
        f.write("{:.9f} {:.9f}\n".format(t[i],data[i]))

# get an instance of RosPack with the default search paths
rospack = rospkg.RosPack()

# get the file path for sensor_fusion
rospack.get_path('sensor_fusion')

path=rospack.get_path('sensor_fusion')+'/dataTxt/laps/'

data_imu = np.loadtxt(path+"magnetometer.txt", delimiter=' ', dtype=np.float)
data_lidar_odom = np.loadtxt(path+"pose_orientation.txt", delimiter=' ', dtype=np.float)
data_ZED_odom = np.loadtxt(path+"ZED_pose_orientation.txt", delimiter=' ', dtype=np.float)
data_acc = np.loadtxt(path+"gyro.txt", delimiter=' ', dtype=np.float)
colors = ['red', 'blue', 'green', 'yellow','purple', 'brown']

wz_comand= np.loadtxt(path+'wz_comand.txt', delimiter=' ', dtype=np.float)
vx_comand= np.loadtxt(path+'vx_comand.txt', delimiter=' ', dtype=np.float)

t_wz = wz_comand[:,0]
wz = wz_comand[:,1]

iwz = np.zeros(wz.shape,np.float)
for i in range(len(wz)-1):
    i+=1
    iwz[i]=(wz[i-1]+wz[i])*(t_wz[i]-t_wz[i-1])/2

wz_yaw = np.zeros(iwz.shape,np.float)
for i in range(len(iwz)):
    for j in range(i):
        wz_yaw[i]+=iwz[j]
'''
for j in range(3):
    for i in range(len(wz_yaw)):
        if (wz_yaw[i]>3.14159265):
            wz_yaw[i]-=2*3.14159265
'''

t_acc = data_acc[:,0]
z_acc = data_acc[:,3]


iz = np.zeros(z_acc.shape,np.float)
for i in range(len(z_acc)-1):
    i+=1
    iz[i]=(z_acc[i-1]+z_acc[i])*(t_acc[i]-t_acc[i-1])/2

z = np.zeros(iz.shape,np.float)
for i in range(len(iz)):
    for j in range(i):
        z[i]+=iz[j]
'''
for j in range(3):
    for i in range(len(z)):
        if (z[i]>3.14159265):
            z[i]-=2*3.14159265
'''
t_magnetometer = data_imu[:,0]
yaw_magnetometer = data_imu[:,1]

yaw_magnetometer+=2.552544031

'''
for i in range(len(yaw_magnetometer)):
    if (yaw_magnetometer[i] > 3.14159265):
        yaw_magnetometer[i] -= 2 * 3.14159265
'''

t_pose_orientation = data_lidar_odom[:,0]
yaw_pose_orientation = data_lidar_odom[:,1]

t_ZED_orientation = data_ZED_odom[:,0]
yaw_ZED_orientation = data_ZED_odom[:,1]

f_plot(t_magnetometer,yaw_magnetometer,
       colors=colors, linewidth=2.)


denormalize(yaw_magnetometer)
denormalize(yaw_pose_orientation)
denormalize(yaw_ZED_orientation)

f_yaw_magnetometer = open(path+'denormalize_yaw_magnetometer.txt', 'w')
f_yaw_pose_orientation = open(path+'denormalize_yaw_pose_orientation.txt', 'w')
f_yaw_ZED_orientation = open(path+'denormalize_yaw_ZED_orientation.txt', 'w')

printToFile(f_yaw_magnetometer,t_magnetometer,yaw_magnetometer)
printToFile(f_yaw_pose_orientation,t_pose_orientation,yaw_pose_orientation)
printToFile(f_yaw_ZED_orientation,t_ZED_orientation, yaw_ZED_orientation)

path+="prepared/"
f = open(path+'mag_yaw.txt', 'w')
printToFile(f,t_magnetometer,yaw_magnetometer)

'''
f_plot(t_pose_orientation,yaw_pose_orientation,
       colors=colors, linewidth=2.)

f_plot(t_magnetometer, t_magnetometer,
       t_pose_orientation,yaw_pose_orientation,
       t_ZED_orientation, yaw_ZED_orientation,
       t_acc, z,
       t_wz, wz_yaw,
       colors=colors, linewidth=2.)




'''
f_plot(t_magnetometer,yaw_magnetometer,
       colors=colors, linewidth=2.)
plt.show()