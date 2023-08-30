#! /usr/bin/env python

import rospy # 1.导包
from std_msgs.msg import UInt32
import torch
import TriHornNet.eval as eval
from mainloop.msg import hand
import numpy as np
import cv2


class Color():
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)
    YELLOW = (0, 211, 255)
    PURPLE = (255, 0, 255)
    CYAN = (255, 255, 0)
    BROWN = (30, 105, 205)

class theTopic():
    def __init__(self,pub= True,sub=True,   name="poseestimation",rate_tx=1,rate_rx=1) -> None:
        self.pkgname = name
        rospy.init_node(self.pkgname)  # 2.初始化 ROS 节点
        rospy.loginfo("---- " + self.pkgname + " ---- Start the hande detection----")  #3.日志输出 HelloWorld
        rospy.loginfo("---- " + self.pkgname + " ---- 1. receive cmd from mainloop  ----")
        rospy.loginfo("---- " + self.pkgname + " ---- 2. send results to mainloop ----")
        rospy.loginfo("---- " + self.pkgname + " ---- 3. import depth image ----")
        rospy.loginfo("---- " + self.pkgname + " ---- 4. estimation ----")

        self.tx_rate = rospy.get_param('~tx_rate')
        self.pub = pub
        self.sub = sub
        self.rx = hand() # UInt32()

        if self.pub:
            self.pub_state = rospy.Publisher(self.pkgname +"_state", UInt32,queue_size=2)
            self.tx_state = UInt32(0)
            self.rate_tx_state = rospy.Rate(rate_tx)
            self.cnt_tx_state = 0   

            self.pub_angle = rospy.Publisher(self.pkgname +"_angle", UInt32,queue_size=2)
            self.tx_angle = UInt32()
            self.rate_tx_angle = rospy.Rate(rate_tx)
            self.cnt_tx_angle = 0    

        if self.sub:
            rospy.sleep(1)
            self.sub = rospy.Subscriber(name="handdetection_cmd",data_class=hand,callback=self.sub_callback, queue_size=2)
            self.cnt_rx = 0

            self.handarea128 = np.zeros(shape=(128,128),dtype=np.uint16)
            self.datalength = 0
            self.handarea128_update = False


    def publish_state(self):
        self.tx_state.data = 0
        self.pub_state.publish(self.tx_state)
        if self.cnt_tx_angle % 10 == 0:
            rospy.loginfo("----4 %s topic tx angle: %d ----",self.pkgname,self.rx.cmd)
        self.cnt_tx_state += 1

    def publish_angle(self):
        if self.pub:
            self.tx_angle.data = (self.cnt_tx_angle & 0xffff) + (( 0 & 0xffff)<<16)
            # self.pub.publish(self.tx)
            self.pub_angle.publish(self.tx_angle)
            if self.cnt_tx_angle % 10 == 0:
                rospy.loginfo("----4 %s topic tx angle: %d ----",self.pkgname,self.rx.cmd)
            self.cnt_tx_angle += 1
            self.rate_tx_angle.sleep()


    def sub_callback(self,msg):
        if self.sub:
            #   self.rx = hand()
            self.rx  = msg
            # self.datalength = len(msg.data)
            if self.rx.annotation[2] > 0:
                handarea128_uint16 = np.array(msg.data).reshape(128,128)
                # handarea128_uint16.astype(np.float_)
                self.handarea128 = ( handarea128_uint16 -msg.annotation[2] ) / msg.annotation[5]

                # print('*'*100)
                # print('type: ',type(self.handarea128), '.  shape: ',self.handarea128.shape)
                # print(self.handarea128[50])
                self.handarea128_update = True
                if self.cnt_rx % 10 == 0:
                    rospy.loginfo("----4 %s rx cmd: %d,        ----",self.pkgname, msg.cmd)


                showimg = False
                if showimg:
                    depth_img255 = ( ( self.handarea128  +1 ) / 2 * 255).astype('uint8')
                    cv2.imshow('deptin image of the hand in pose estimation ',depth_img255)
                    cv2.waitKey(10)

                    # if self.cnt_rx % 20 == 0:
                    #     np.savetxt("/homeL/zheng/ros_python/tempsave/poseesti/" + str(self.cnt_rx).rjust(5,'0') + '.txt', depth_img255,fmt='%d')
                    #     cv2.imwrite("/homeL/zheng/ros_python/tempsave/poseesti/" + str(self.cnt_rx).rjust(5,'0') + '.png',depth_img255)


            else:
                print('4: rx cmd',msg.cmd,' and empty data')

            self.cnt_rx += 1








class poseEstimation():
    def __init__(self) -> None:
        
        self.source = rospy.get_param('~source')
        self.endcnt = rospy.get_param('~endcnt')
        self.path = rospy.get_param('~path')
        self.pred=eval.pred()
        self.cnt = 0
        self.init = False

        self.color =[Color().BLUE, Color().BROWN, Color().YELLOW, Color().GREEN, Color().PURPLE,Color().RED, Color().CYAN]
        
    def run(self):
        if self.init == False:
            if self.source == 0:
                self.pred.run_dataset()
            elif self.source == 1:
                self.pred.run_realtime_init()

                #       *******--------     change to proper inputs please   -------------******************
                print('init poseestimate--> run')
                self.inputs = torch.rand((1,1,128,128)).to("cuda:0") * 2 - 1
                pred_out = self.pred.run_realtime(self.inputs)
                # rospy.loginfo(pred_out)
                # print('init   type : ',type(self.inputs),' .  shape: ',self.inputs.shape,'    in init, inputs[0,0,0,:]: ',self.inputs[0,0,0])
            self.init = True
            # print("estimate inti = false")
        else:
            #   read sub-image from KinectV2, and then normalize it to a tensor((1,1,128,128))
            if topic.handarea128_update:
                subimg0 = torch.from_numpy(topic.handarea128)
                # self.subimg = subimg0[None,None,:,:].to("cuda:0")
                self.inputs[0,0,:,:] = subimg0

                self.subimg = self.inputs
                # print('-'*100)
                # print('type: ',type(self.subimg),'   shape:  ',   self.subimg.shape)
                # print(self.subimg[0,0,50,:])


                #   predict
                pred_out = self.pred.run_realtime(self.inputs)

                # rospy.loginfo('---- estimation resuls:   '+str(self.cnt)+' -*-*-*-*-*-* '+str(pred_out.size()))

                #   restore pred_out to general result, (u,v,depth(mm))

                #   send result to ...

                self.cnt += 1
                self.handarea128_updata = False


                showimg = True
                if showimg:
                    # print('pred_out size',pred_out.shape,'pred_out: ',pred_out)

                    preds  =  pred_out[0,:,:]
                    # print('preds: ',preds)

                    depth_img255 = ( ( topic.handarea128  +1 ) / 2 * 255).astype('uint8')

                    imgrgb = cv2.cvtColor(depth_img255, cv2.COLOR_BGR2RGB)
                    # for i in range(preds.shape[0]):
                    #     cv2.circle(imgrgb, center=(int(preds[i, 0]), int(preds[i, 1])), radius=2, color=(0, 255, 0), thickness=1)     #   green,  prediction
                    imgrgb = self.skeleton(imgrgb=imgrgb,preds=preds)
                    cv2.imshow('deptin image of the hand in pose estimation ',imgrgb)
                    cv2.waitKey(10)

                    if True:
                        cv2.imwrite("/homeL/zheng/ros_python/tempsave/poseesti0819/" + str(self.cnt).rjust(5,'0') + '.png',imgrgb)

    def skeleton(self,imgrgb,preds):
#       wrist(1) + index(4,mcp, pip, dip, tip) + middle(4) + ring(4) +little(4) + thumber(4)
        length = preds.shape[0]
        
        for j in range(length):
            i = j + 3                   #   3 - 23
            size = np.mod(i,4)+1        #   1-4 --> mcp, pip, dip, tip
            color = i // 4              #   0 - 5   
            #   circle for joint
            cv2.circle(imgrgb, center=(int(preds[j, 0]), int(preds[j, 1])), radius=size, color=self.color[color], thickness=1) 
            #   line for skeleton
            if size < 4 and color > 0:
                cv2.line(img=imgrgb,pt1=(int(preds[j, 0]), int(preds[j, 1])),pt2=(int(preds[j+1, 0]), int(preds[j+1, 1])),color=self.color[color])

            
        return imgrgb

        



topic = theTopic(rate_tx=10)
poseesti = poseEstimation()

