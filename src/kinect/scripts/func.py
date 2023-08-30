#! /usr/bin/env python

import rospy # 1.导包
from std_msgs.msg import UInt32
from mainloop.msg import hand
import numpy as np
import time
import os
import socket
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
from PIL import Image as PILImage
import struct


class tcp():
    def __init__(self) -> None:
        self.srv = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        self.srv.bind(('134.100.13.131',666))
        self.srv.listen(1)
        time.sleep(2)
        self.con, addr = self.srv.accept()
        print("address: ",addr)
    def rx(self):
        self.data = self.srv.recv(1024)
        if  len(self.data) :
            print(self.data)
        else:
            print('no data')


class theTopic():
    def __init__(self,pub= True,sub=True, name="kinectv2",rate_tx=1,rate_rx=1) -> None:
        self.pkgname = name
        rospy.init_node(self.pkgname)  # 2.初始化 ROS 节点
        rospy.loginfo("---- "+self.pkgname+" ---- Start the kinect----")  #3.日志输出 HelloWorld
        rospy.loginfo("---- "+self.pkgname+" ---- 1. receive cmd from mainloop  ----")
        rospy.loginfo("---- "+self.pkgname+" ---- 2. send results to hand detection ----")
        rospy.loginfo("---- "+self.pkgname+" ---- 3. read depth image ----")
        rospy.loginfo("---- "+self.pkgname+" ---- 4. crop ----")

        self.pub = pub
        self.sub = sub

        self.bridge = CvBridge()

        self.rx_main = UInt32()
        self.rx_kinect = Image()
        if pub:
            # self.pub = rospy.Publisher(self.pkgname +"_cmd", Int32,queue_size=10)
            # self.tx = Int32()
            self.pub = rospy.Publisher(self.pkgname +"_cmd", hand,queue_size=10)
            self.tx = hand()
            self.rate_tx = rospy.Rate(rate_tx)
            self.cnt_tx = 0    #   low 16 bits
            self.cmd = 0    #   high 16 bits

        if sub:
            rospy.sleep(1)
            self.sub_main = rospy.Subscriber(name="mainloop"+"_cmd",data_class=UInt32,callback=self.sub_callback_main, queue_size=2)
            # self.sub_kinect = rospy.Subscriber(name="kinect2/sd/image_depth_rect",data_class=Image,callback=self.sub_callback_kinect,queue_size=2)
            self.cnt_rx = 0
            self.cnt_rx_kinect = 0
            self.imgarr = np.zeros((424,512),dtype=np.uint16)

            

    def publish(self):
        if self.pub:
            self.tx.name = "data_from_kinect"
            # self.tx.cmd = (self.cnt_tx & 0xffff) + ((self.cmd & 0xffff)<<16)
            self.tx.cmd = self.rx_main.data
            # self.tx.data = np.arange(10,dtype=np.uint16)
            self.tx.data = np.array([1,2,3,4,5,6,4,5,6,7,8,9],dtype=np.uint16)
            self.pub.publish(self.tx)

            rospy.loginfo("----2 %s topic tx: %s,  %d ----",self.pkgname, self.tx.name ,self.rx_main.data)
            self.cnt_tx += 1
            self.rate_tx.sleep()
        else:
            rospy.loginfo("no publisher for %s",self.pkgname)

    def sub_callback_main(self,msg):
        if self.sub:
            self.rx_main.data = msg.data
            rospy.loginfo("----2 %s rx main topic:      %d ----",self.pkgname, self.rx_main.data)
            self.cnt_rx += 1


    def sub_callback_kinect(self,msg):
        if self.sub:
            self.rx_kinect = msg
            rospy.loginfo("----2 %s rx kinect topic:      %d --**********************--",self.pkgname, self.rx_kinect.width )
            
            self.cnt_rx_kinect += 1
            # if (self.cnt_rx_kinect > 100):

            #     length = int(len(self.rx_kinect.data)/2)
            #     decarr = np.ones(int(length))
            #     for i in range(length):
            #         self.imgarr[i // self.rx_kinect.width][i % self.rx_kinect.width] = int.from_bytes(self.rx_kinect.data[i*2:i*2+2],'little')
            #     # depth_img3 = np.array([self.imgarr, self.imgarr, self.imgarr],dtype=np.float32)
            #     depth_img255 = ( ( self.imgarr - np.min(self.imgarr) ) / (2000 -np.min(self.imgarr) ) * 255).astype('uint8')
            #     # img = PILImage.fromarray(self.imgarr)
            #     cv2.imshow("num=3",depth_img255)
            #     cv2.waitKey(10)



class Kinect():
    def __init__(self) -> None:
        self.cnt = 0

    def run(self):
        self.cnt += 1



topic = theTopic()
kinectv2 = Kinect()
# depthdata = tcp()