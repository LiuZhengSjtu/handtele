# from pynput.keyboard import Key, Listener

from pynput import keyboard 
import rospy # 1.导包
from std_msgs.msg import UInt32
import numpy as np
import time



class keyboard_detect():
    def __init__(self) -> None:
        self.state = 0
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.detect()
        self.time_restart = 0

    def on_press(self,key):
        if key == keyboard.Key.esc:
            # Stop listener, stop and wait
            self.state = 1
        elif key == keyboard.Key.pause:
            #   restart (don't need to initial again)
            self.state = 2
            self.time_restart = time.time()

    def detect(self):
        self.listener.start()

    def updatastate(self):
        if self.state == 2 and time.time() - self.time_restart > 1:
            self.state = 0

        # print(self.state)
        return self.state


class theTopic():
    def __init__(self,name="mainloop",rate_tx=1,rate_rx=1) -> None:
        self.pkgname = name
        rospy.init_node(self.pkgname)  # 2.初始化 ROS 节点
        rospy.loginfo("---- "+self.pkgname+" ---- Start the mainloop, state control ----")  #3.日志输出 HelloWorld
        rospy.loginfo("---- "+self.pkgname+" ---- 1. send cmd to func_1 - deal with hand area detection ----")
        rospy.loginfo("---- "+self.pkgname+" ---- 2. send cmd to func_2 - deal with hand position estimation ----")
        rospy.loginfo("---- "+self.pkgname+" ---- 3. send cmd to func_3 - data saving ----")
        rospy.loginfo("---- "+self.pkgname+" ---- 4. send cmd to func_4 - position reschedule ----")
        rospy.loginfo("---- "+self.pkgname+" ---- 5. send cmd to func_5 - communicate with SHADOW ----")

        rate_tx =  rospy.get_param('~tx_rate')

        self.pub = rospy.Publisher(self.pkgname+"_cmd", UInt32,queue_size=10)
        self.tx = UInt32()
        self.rate = rospy.Rate(rate_tx)
        self.cnt = np.uint16(0)    #   low 16 bits
        self.cmd = np.uint16(0)    #   high 16 bits

    def publish(self, state = 0 ):
        self.cmd = state

        self.tx.data = (self.cnt & 0xffff) + ((self.cmd & 0xffff)<<16)
        self.pub.publish(self.tx)
        # rospy.loginfo("----1 %s topic tx: %d ----",self.pkgname,self.tx.data)
        self.cnt += 1

        self.rate.sleep()

class loop():
    def __init__(self):
        self.cnt = 0
        self.endcnt = rospy.get_param( '~endcnt')
        




topic = theTopic()

kybd = keyboard_detect()

mloop = loop()