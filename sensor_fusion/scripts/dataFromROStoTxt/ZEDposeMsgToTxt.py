#!/usr/bin/env python

import rospy
import math
import rospkg


from sensor_msgs.msg import Imu
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import PoseWithCovariance
from nav_msgs.msg import Odometry

rad2degrees = 180.0/math.pi
yaw_offset = 0 #used to align animation upon key press


# get an instance of RosPack with the default search paths
rospack = rospkg.RosPack()

# get the file path for sensor_fusion
rospack.get_path('sensor_fusion')

path=rospack.get_path('sensor_fusion')+'/dataTxt/laps/'

pose_position = open(path+'ZED_pose_position.txt', 'w')
pose_orientation = open(path+'ZED_pose_orientation.txt', 'w')


def processOdometry_message(dataMsg):
    global yaw_offset
    global f

    roll=0
    pitch=0
    yaw=0

    quaternion = (
        dataMsg.pose.pose.orientation.x,
        dataMsg.pose.pose.orientation.y,
        dataMsg.pose.pose.orientation.z,
        dataMsg.pose.pose.orientation.w)

    (roll, pitch, yaw) = euler_from_quaternion(quaternion)

    (sec,nsec)=(dataMsg.header.stamp.secs,dataMsg.header.stamp.nsecs)

    time = 1./1000000000 * nsec + sec

    (roll,pitch,yaw) = euler_from_quaternion(quaternion)

    pose_position.write("{:.9f} {:.9f} {:.9f} {:.9f}\n".format(time, dataMsg.pose.pose.position.x, dataMsg.pose.pose.position.y, dataMsg.pose.pose.position.z))
    pose_orientation.write("{:.9f} {:.9f} {:.9f} {:.9f}\n".format(time, yaw, roll, pitch))

rospy.init_node("ZEDposeupdateMsgToTxt")
sub = rospy.Subscriber('camera/odomZED', Odometry, processOdometry_message)
rospy.spin()
