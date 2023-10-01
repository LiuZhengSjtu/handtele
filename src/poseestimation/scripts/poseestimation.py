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
sys.path.insert(1,"/homeL/zheng/ros_python/handtele/src/poseestimation/scripts")
import TriHornNet.eval as eval
from func import poseesti, topic

if __name__ == "__main__":

    # topic.sub
    while not rospy.is_shutdown():
        
        poseesti.run()
        topic.publish_state()
        topic.publish_angle()

        if topic.cnt_tx_angle >  poseesti.endcnt:
            break
