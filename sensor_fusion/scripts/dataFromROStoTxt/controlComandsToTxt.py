#!/usr/bin/env python

import rospy
import math
import rospkg

from geometry_msgs.msg import Twist

# get an instance of RosPack with the default search paths
rospack = rospkg.RosPack()

# get the file path for sensor_fusion
rospack.get_path('sensor_fusion')

path=rospack.get_path('sensor_fusion')+'/dataTxt/laps/'

wz_comand = open(path+'wz_comand.txt', 'w')
vx_comand = open(path+'vx_comand.txt', 'w')

wz=0
vx=0

def processTwist_message(imuMsg):
    global wz
    global vx
    now = rospy.get_rostime()

    rospy.loginfo("Current time %i %i", now.secs, now.nsecs)

    time = 1. / 1000000000 * now.nsecs + now.secs

    wz_comand.write("{:.9f} {:.9f}\n".format(time-0.000001, wz))
    vx_comand.write("{:.9f} {:.9f}\n".format(time-0.000001, vx))

    wz=imuMsg.angular.z
    vx=imuMsg.linear.x

    wz_comand.write("{:.9f} {:.9f}\n".format(time, imuMsg.angular.z))
    vx_comand.write("{:.9f} {:.9f}\n".format(time, imuMsg.linear.x))

rospy.init_node("imuMsgToTxt")
sub = rospy.Subscriber('cmd_vel', Twist, processTwist_message)
rospy.spin()
