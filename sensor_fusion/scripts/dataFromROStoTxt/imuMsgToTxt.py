#!/usr/bin/env python

import rospy
import math
import rospkg

from sensor_msgs.msg import Imu
from tf.transformations import euler_from_quaternion

calculate_average= False

if(calculate_average):
    averageAccelerationX = 0
    averageAccelerationY = 0
    N=0

rad2degrees = 180.0/math.pi
yaw_offset = 0 #used to align animation upon key press

# get an instance of RosPack with the default search paths
rospack = rospkg.RosPack()

# get the file path for sensor_fusion
rospack.get_path('sensor_fusion')

path=rospack.get_path('sensor_fusion')+'/dataTxt/laps/'

magnetometer = open(path+'magnetometer.txt', 'w')
magnetometer_cavariance = open(path+'magnetometer_covariance.txt', 'w')

gyro = open(path+'gyro.txt', 'w')
gyro_cavariance = open(path+'gyro_covariance.txt', 'w')

accelerometer = open(path+'accelerometer.txt', 'w')
accelerometer_cavariance = open(path+'accelerometer_covariance.txt', 'w')

if(calculate_average):
    averageAccelerationStable = open(path + 'averageAccelerationStable.txt', 'w')

def processIMU_message(imuMsg):
    global yaw_offset
    global f
    global N
    global averageAccelerationX
    global averageAccelerationY

    roll=0
    pitch=0
    yaw=0

    quaternion = (
      imuMsg.orientation.x,
      imuMsg.orientation.y,
      imuMsg.orientation.z,
      imuMsg.orientation.w)

    (sec, nsec) = (imuMsg.header.stamp.secs, imuMsg.header.stamp.nsecs)
    time = 1. / 1000000000 * nsec + sec

    (roll,pitch,yaw) = euler_from_quaternion(quaternion)

    magnetometer.write("{:.9f} {:.9f} {:.9f} {:.9f}\n".format(time, yaw, roll, pitch))
    magnetometer_cavariance.write("{}\n".format(imuMsg.orientation_covariance))

    if (calculate_average):
        averageAccelerationX+=imuMsg.linear_acceleration.x
        averageAccelerationY+=imuMsg.linear_acceleration.y
        N+=1

    #calibration constants
    accelerometer.write("{:.9f} {:.9f} {:.9f} {:.9f}\n".format(time, imuMsg.linear_acceleration.x - 0.149527500,
                                                               imuMsg.linear_acceleration.y + 0.053518950,
                                                               imuMsg.linear_acceleration.z
                                                               ))

    accelerometer_cavariance.write("{}\n".format(imuMsg.angular_velocity_covariance))

    gyro.write("{:.9f} {:.9f} {:.9f} {:.9f}\n".format(time, imuMsg.angular_velocity.x,imuMsg.angular_velocity.y, imuMsg.angular_velocity.z))
    gyro_cavariance.write("{}\n".format(imuMsg.linear_acceleration_covariance))


rospy.init_node("imuMsgToTxt")
sub = rospy.Subscriber('imu', Imu, processIMU_message)
rospy.spin()
if(calculate_average):
    averageAccelerationStable.write("{:.9f} {:.9f}\n".format(averageAccelerationX/N,averageAccelerationY/N))
