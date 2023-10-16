#! /usr/bin/env python
"""
    Python 版本的 HelloVScode，执行在控制台输出 HelloVScode
    实现:
    1.导包
    2.初始化 ROS 节点
    3.日志输出 HelloWorld

rosrun poseestimation poseestimation.py 
"""

import rospy # 1.导包
import torch
import sys
sys.path.insert(1,"/homeL/zheng/ros_python/handtele/src/ikfk/scripts")

from fun_bio_ik import bio_ik_physical_hand

if __name__ == "__main__":

    # topic.sub
    while not rospy.is_shutdown():
        #   the publish cmd is moved to sub_call_back function
        # bio_ik_physical_hand.publish_regularized_angle()

        if bio_ik_physical_hand.cnt_tx >  bio_ik_physical_hand.endcnt or bio_ik_physical_hand.esc_cnt > 5 :
            rospy.signal_shutdown(' esc ')
            break