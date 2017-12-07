import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import rospkg
from scipy.interpolate import interp1d
from numpy import random, mean, var, std
from math import *

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

path = rospack.get_path('sensor_fusion') + '/dataTxt/laps/prepared/'

data_lidar = np.loadtxt(path + "lidar_pose.txt", delimiter=' ', dtype=np.float)
data_orb = np.loadtxt(path + "orb_pose.txt", delimiter=' ', dtype=np.float)
data_zed = np.loadtxt(path + "zed_pose.txt", delimiter=' ', dtype=np.float)
data_gyro = np.loadtxt(path + "gyro_yaw.txt", delimiter=' ', dtype=np.float)
data_mag = np.loadtxt(path + "mag_yaw.txt", delimiter=' ', dtype=np.float)

k=0
for i in range(len(data_lidar[:, 0])):
    if data_lidar[i, 0]<=1512232502.018991232:
        k+=1

e=0
for i in range(len(data_lidar[:, 0])):
    if data_lidar[i, 0]>=1512232712.210900068:
        e+=1

e = (len(data_lidar[:, 0]) - e - 20)
k+=200
t_lidar=data_lidar[k:e, 0]
x_lidar=data_lidar[k:e, 1]
y_lidar=data_lidar[k:e, 2]
yaw_lidar=data_lidar[k:e, 3]

k=0
for i in range(len(data_orb[:, 0])):
    if data_orb[i, 0]<=1512232502.018991232:
        k+=1

e=0
for i in range(len(data_orb[:, 0])):
    if data_orb[i, 0]>=1512232712.210900068:
        e+=1

e = (len(data_orb[:, 0]) - e - 1)

t_orb=data_orb[k:e, 0]
x_orb=data_orb[k:e, 1]
y_orb=data_orb[k:e, 2]
yaw_orb=data_orb[k:e, 3]

k=0
for i in range(len(data_zed[:, 0])):
    if data_zed[i, 0]<=1512232502.018991232:
        k+=1

e=0
for i in range(len(data_zed[:, 0])):
    if data_zed[i, 0]>=1512232712.210900068:
        e+=1

e = (len(data_zed[:, 0]) - e - 1)

t_zed=data_zed[k:e, 0]
x_zed=data_zed[k:e, 1]
y_zed=data_zed[k:e, 2]
yaw_zed=data_zed[k:e, 3]


t_gyro=data_gyro[:, 0]
yaw_gyro=data_gyro[:, 1]

t_mag=data_mag[:, 0]
yaw_mag=data_mag[:, 1]


#x data
plt.plot(t_lidar, x_lidar, t_orb, x_orb,  t_zed, x_zed)
plt.legend(['x(t) - lidar',
            'x(t) - orb',
            'x(t) - zed'], loc='best')
#plt.show()

#y data
plt.plot(t_lidar, y_lidar, t_orb, y_orb,  t_zed, y_zed)
plt.legend(['y(t) - lidar', 'y(t) - orb', 'y(t) - zed'], loc='best')
#plt.show()

#yaw data
plt.plot(t_lidar, yaw_lidar, t_orb, yaw_orb,  t_zed, yaw_zed, t_gyro,yaw_gyro,t_mag,yaw_mag)
plt.legend(['yaw(t) - lidar', 'yaw(t) - orb', 'yaw(t) - zed', 'yaw(t) - gyro', 'yaw(t) - mag'], loc='best')
plt.show()

t = t_lidar

forb = interp1d(t_orb, x_orb)
fzed = interp1d(t_zed, x_zed)


#x data
plt.plot(t_lidar, forb(t_lidar)-x_lidar,  t_lidar, fzed(t_lidar)-x_lidar )
x1=forb(t_lidar)-x_lidar
x2=fzed(t_lidar)-x_lidar
plt.legend(["x(t) error - orb. E="+str(mean(x1))+" D="+str(var(x1))+" std="+str(std(x1))+".",
            "x(t) error - zed. E="+str(mean(x2))+" D="+str(var(x2))+" std="+str(std(x2))+"."],
           loc='best')
plt.show()

forb = interp1d(t_orb, y_orb)
fzed = interp1d(t_zed, y_zed)

#y data
plt.plot(t_lidar, forb(t_lidar)-y_lidar,  t_lidar, fzed(t_lidar)-y_lidar )
x1=forb(t_lidar)-y_lidar
x2=fzed(t_lidar)-y_lidar
plt.legend(["y(t) error - orb. E="+str(mean(x1))+" D="+str(var(x1))+" std="+str(std(x1))+".",
            "y(t) error - zed. E="+str(mean(x2))+" D="+str(var(x2))+" std="+str(std(x2))+"."],
            loc='best')
plt.show()

forb = interp1d(t_orb, yaw_orb)
fzed = interp1d(t_zed, yaw_zed)
fgyro = interp1d(t_gyro, yaw_gyro)
fmag = interp1d(t_mag, yaw_mag)

#yaw
plt.plot(t_lidar, forb(t_lidar)-yaw_lidar,
         t_lidar, fzed(t_lidar)-yaw_lidar,
         t_lidar, fgyro(t_lidar) - yaw_lidar,
         t_lidar, fmag(t_lidar) - yaw_lidar)
x1=forb(t_lidar)-yaw_lidar
x2=fzed(t_lidar)-yaw_lidar
x3=fgyro(t_lidar) - yaw_lidar
x4=fmag(t_lidar) - yaw_lidar

plt.legend(["yaw(t) error - orb. E="+str(mean(x1))+" D="+str(var(x1))+" std="+str(std(x1))+".",
            "yaw(t) error - zed. E="+str(mean(x2))+" D="+str(var(x2))+" std="+str(std(x2))+".",
            "yaw(t) error - gyro. E="+str(mean(x3))+" D="+str(var(x3))+" std="+str(std(x3))+".",
            "yaw(t) error - mag. E="+str(mean(x4))+" D="+str(var(x4))+" std="+str(std(x4))+"."], loc='best')

plt.show()