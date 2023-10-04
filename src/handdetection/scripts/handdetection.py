#! /usr/bin/env python
"""
    Python 版本的 HelloVScode，执行在控制台输出 HelloVScode
    实现:
    1.导包
    2.初始化 ROS 节点
    3.日志输出 HelloWorld


"""

import rospy # 1.导包

from yolov7_tiny_pytorch import predict
import sys
sys.path.insert(1,"/homeL/zheng/ros_python/handtele/src/handdetection/scripts")
from func import handdete,topic



if __name__=="__main__":
    while not rospy.is_shutdown():
        topic.publish()
        handdete.run(topic.rx_main_cmd)

        if topic.cnt_tx > handdete.endcnt or topic.esc_cnt > 10:
            break
    


