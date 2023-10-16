#! /usr/bin/env python
"""
    Python 版本的 HelloVScode，执行在控制台输出 HelloVScode
    实现:
    1.导包
    2.初始化 ROS 节点
    3.日志输出 HelloWorld

roslaunch mainloop mainlaunch.launch
"""

import rospy # 1.导包
from std_msgs.msg import Int32
import sys
sys.path.insert(1,"/homeL/zheng/ros_python/handtele/src/mainloop/scripts")
# print(sys.path)
from func import kybd, topic, mloop


#------------------------------------------------------------------------------------------------------#
if __name__ == "__main__":

    while not rospy.is_shutdown():
   
        topic.publish(state = kybd.updatastate())

        if topic.cnt > mloop.endcnt or kybd.esc_pressed > 15:
            break
        
      





