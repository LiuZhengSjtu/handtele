#! /usr/bin/env python

import rospy # 1.导包
from std_msgs.msg import UInt32
import torch
import TriHornNet.eval as eval
from mainloop.msg import hand
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import time
import os


class Color():
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)
    YELLOW = (0, 211, 255)
    PURPLE = (255, 0, 255)
    CYAN = (255, 255, 0)
    BROWN = (30, 105, 205)

class ShareData():
    def __init__(self):
        self.trackenable = -1
        self.gesturests = -1
        # self.handcenter = np.array([0,0,0])
        self.handcentercode = 0          # for UInt32.data,   0~15 bits--> distance, 16~23 bits--> y pixel (of 128), 24~31 bits--> x pixel
        self.com = [0,0,0]
        self.com_f = [0,0,0]

sharedata = ShareData()

class theTopic():
    def __init__(self,pub= True,sub=True,   name="poseestimation",rate_tx=10,rate_rx=1) -> None:
        self.pkgname = name
        rospy.init_node(self.pkgname)  # 2.初始化 ROS 节点
        rospy.loginfo("---- " + self.pkgname + " ---- Start the hande detection----")  #3.日志输出 HelloWorld
        rospy.loginfo("---- " + self.pkgname + " ---- 1. receive cmd from mainloop  ----")
        rospy.loginfo("---- " + self.pkgname + " ---- 2. send results to mainloop ----")
        rospy.loginfo("---- " + self.pkgname + " ---- 3. import depth image ----")
        rospy.loginfo("---- " + self.pkgname + " ---- 4. estimation ----")

        #   from image coor to tripod coor, rotate along Z,X and Y axes respectively, for following angles.
        self.theta_z = rospy.get_param('~theta_z')
        self.theta_x = rospy.get_param('~theta_x')
        self.theta_y = rospy.get_param('~theta_y')

        self.Rot = R.from_euler("ZXY",[-self.theta_z, -self.theta_x, -self.theta_y],degrees = True)
        # print(self.Rot.as_dcm())
        # print('12121')
        #   use self.Rot.apply([px,py,pz])

        self.tx_rate = rospy.get_param('~tx_rate')
        self.pub = pub
        self.sub = sub
        self.rx = hand() # UInt32()
        self.rx.annotation = [0,0,250,128,128,100]

        if self.pub:
            #   to hand detection pkg
            self.pub_state = rospy.Publisher(self.pkgname +"_state", UInt32,queue_size=2)
            self.tx_state = UInt32(0)
            self.rate_tx_state = rospy.Rate(rate_tx)
            self.cnt_tx_state = 0   

            #   to bio_ik
            self.pub_angle = rospy.Publisher(self.pkgname +"_cmd", hand,queue_size=2)
            self.tx_angle = hand()
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
        #   feedback the state of the estimation to the hand detection pkg, including the wrist point location (com)
        global sharedata
        self.tx_state.data = sharedata.handcentercode
        self.pub_state.publish(self.tx_state)
        if self.cnt_tx_state % 10 == 0:
            rospy.loginfo("----4 %s topic tx angle: %d ----gesture_state: %d, trackenable: %d .",self.pkgname,self.tx_state.data ,sharedata.gesturests, sharedata.trackenable)
            rospy.loginfo("the center code: %d ",sharedata.handcentercode)
        self.cnt_tx_state += 1

    def publish_angle(self):
        #    contain the key points location in wrist frame, to the bio_ik pkg. unit: 0.01 mm
        if self.pub:
            self.tx_angle.data = (self.cnt_tx_angle & 0xffff) + (( 0 & 0xffff)<<16)
            # self.pub.publish(self.tx)
            self.pub_angle.publish(self.tx_angle)
            if self.cnt_tx_angle % 10 == 0:
                rospy.loginfo("----4 %s topic tx angle: %d ----",self.pkgname,self.rx.cmd)
            self.cnt_tx_angle += 1
            self.rate_tx_angle.sleep()


    def sub_callback(self,msg):
        global sharedata
        if self.sub:
            #   self.rx = hand()
            self.rx  = msg
            sharedata.com = self.rx.annotation[0:3]

            # self.datalength = len(msg.data)
            if self.rx.annotation[2] > 300:
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

        self.halfimg = 64
        self.fw = 512 / 70 * 57.3
        self.fh = 424 / 60 * 57.3
        self.preds_3d = np.zeros((21,3))        #   in camera frame, x y z order

        self.gesturests = 0             #   0: initial, 1: up+disperse, first, 2: up+pinch 3: up+disperse, second 4: up+pinch
        self.gesturejudge = [0,0,0,0]     #   [pinch, disperse, palm up, Timer]
        self.thuber_dis = np.array([0,0,0,0])
        self.ctrlenable = -1            #   control robotic hand
        self.trackenable = -1           #   feedback handcenter to hand detection
        
        tm = time.localtime()
        timestr = str(tm.tm_year)+'-'+str(tm.tm_mon).rjust(2,'0') +'-'+ str(tm.tm_mday).rjust(2,'0') +'-'+ str(tm.tm_hour).rjust(2,'0') +'-'+ str(tm.tm_min).rjust(2,'0') +'-'+ str(tm.tm_sec).rjust(2,'0') 
        self.savepath = '/homeL/zheng/Downloads/temp/' + timestr + '/' 
        os.mkdir(self.savepath)
        
        self.filehandle = open(self.savepath + timestr + '.txt','w')

        self.filehandle.write('matrix: ' + '\n')
        self.filehandle.write(str(topic.Rot.as_dcm())+'\n')

        #   the order re-map from the predictions to the physical model frame, wrist frame.
        # self.index_remap = np.array([])
        
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

                preds  =  pred_out[0,:,:]
                print('preds pre process: ',preds)
                self.img2grd(preds0=preds)

                print('-*'*50)
                print('preds_3d: ',self.preds_3d)
                print('M: ',topic.Rot.as_dcm())
                print('preds_ground: ',self.preds_ground)
                print('handcenter: ',self.handcenter)

                self.gesturestatemachine()

                self.dataupdate()

                # rospy.loginfo(pred_out)
                # print('init   type : ',type(self.inputs),' .  shape: ',self.inputs.shape,'    in init, inputs[0,0,0,:]: ',self.inputs[0,0,0])

            self.init = True
            # print("estimate inti = false")
        else:
            #   read sub-image from KinectV2, and then normalize it to a tensor((1,1,128,128))
            if topic.handarea128_update:
                topic.handarea128_update = False
                subimg0 = torch.from_numpy(topic.handarea128)
                # self.subimg = subimg0[None,None,:,:].to("cuda:0")
                self.inputs[0,0,:,:] = subimg0

                self.subimg = self.inputs
                # print('-'*100)
                # print('type: ',type(self.subimg),'   shape:  ',   self.subimg.shape)
                # print(self.subimg[0,0,50,:])


                #   predict
                pred_out = self.pred.run_realtime(self.inputs)
                preds  =  pred_out[0,:,:]
                self.preds_original = preds

                self.img2grd(preds0=preds)

                self.gesturestatemachine()

                self.dataupdate()

                print('-*'*50)
                # print('preds: ',preds)
                print('preds_ground: ',self.preds_ground)
                print('handcenter: ',self.handcenter)



                # rospy.loginfo('---- estimation resuls:   '+str(self.cnt)+' -*-*-*-*-*-* '+str(pred_out.size()))

                #   restore pred_out to general result, (u,v,depth(mm))

                #   send result to ...

                self.cnt += 1
                self.handarea128_updata = False




                showimg = True
                if showimg:
                    # print('pred_out size',pred_out.shape,'pred_out: ',pred_out)

                    
                    # print('preds: ',preds)

                    depth_img255 = ( ( topic.handarea128  +1 ) / 2 * 255).astype('uint8')

                    imgrgb = cv2.cvtColor(depth_img255, cv2.COLOR_BGR2RGB)
                    # for i in range(preds.shape[0]):
                    #     cv2.circle(imgrgb, center=(int(preds[i, 0]), int(preds[i, 1])), radius=2, color=(0, 255, 0), thickness=1)     #   green,  prediction
                    imgrgb = self.skeleton(imgrgb=imgrgb,preds=preds)
                    cv2.imshow('depth image of the hand in pose estimation ',imgrgb)
                    cv2.waitKey(10)

                    if False:
                        cv2.imwrite(self.savepath + str(self.cnt).rjust(5,'0') + '.png',imgrgb)
                        self.filehandle.write(str(self.cnt).rjust(5,'0') + '\n')

                        # self.filehandle.write('original preds: ' + '\n')
                        # self.filehandle.write(str(self.preds_original) + '\n')

                        self.filehandle.write('annotation: ' + '\n')
                        self.filehandle.write(str(topic.rx.annotation) + '\n')

                        # self.filehandle.write('3D preds: ' + '\n')
                        # self.filehandle.write(str(self.preds_3d) + '\n')

                        self.filehandle.write('self.gesturejudge: ' + '\n')
                        self.filehandle.write(str(self.gesturejudge) + '\n')

                        self.filehandle.write('ground preds: ' + '\n')
                        self.filehandle.write(str(self.preds_ground) + '\n')
                        
                        self.filehandle.write('*'*30 + '\n'*2)



    def skeleton(self,imgrgb,preds):

        #   draw key points by circles.
#       wrist(1) + index(4,mcp, pip, dip, tip) + middle(4) + ring(4) +little(4) + thumber(4)
        length = preds.shape[0]
        
        for j in range(length):
            i = j + 3                   #   3 - 23
            size = np.mod(i,4)+1        #   1-4 --> mcp, pip, dip, tip
            color = i // 4              #   0 - 5   
            #   circle for joint
            cv2.circle(imgrgb, center=(int(preds[j, 0]), int(preds[j, 1])), radius=size, color=self.color[color], thickness= 1) 
            #   line for skeleton
            if size < 4 and color > 0:
                cv2.line(img=imgrgb,pt1=(int(preds[j, 0]), int(preds[j, 1])),pt2=(int(preds[j+1, 0]), int(preds[j+1, 1])),color=self.color[color])

            #   not track (0,0,254) red, track (254,0,0) blue
            cv2.putText(imgrgb,str(self.gesturests),(100,20),cv2.FONT_HERSHEY_SIMPLEX,0.75,color=( 127 + 127 * self.trackenable, 0, 127 - 127 * self.trackenable),thickness=2)

            
        return imgrgb

    def img2grd(self,preds0):
        preds = preds0.cpu().detach().numpy()
        #   preds is the estimated result from reshaped image array. Further inverse reshape on X and Y axes is neccesary.
        self.preds_3d[:,2] = preds[:,2] * topic.rx.annotation[5] + topic.rx.annotation[2]
        x_pix = ( (preds[:,0] - self.halfimg) / self.halfimg * topic.rx.annotation[4] ) #   in cropped hand image, not neccessary equal to reshaped 128*128
        y_pix = ( (preds[:,1] - self.halfimg) / self.halfimg * topic.rx.annotation[3] ) #   origianl image pixel scale 
        self.preds_3d[:,0] = (topic.rx.annotation[1] - 256 + ( (preds[:,0] - self.halfimg) / self.halfimg * topic.rx.annotation[4] )) / self.fw * self.preds_3d[:,2]      #topic.rx.annotation[2]
        self.preds_3d[:,1] = (topic.rx.annotation[0] - 208 + ( (preds[:,1] - self.halfimg) / self.halfimg * topic.rx.annotation[3] )) / self.fh * self.preds_3d[:,2]      #topic.rx.annotation[2]

        #   from camera frame to camera-based horizon frame
        self.preds_ground = topic.Rot.apply(self.preds_3d)

        #   center of four fingers (x,y,z) pix, pix, mm. x and y are in 128*128 image, so x and y are [0~128]
        self.handcenter = ( np.mean(preds[[1,5,9,13],0]) ,  np.mean(preds[[1,5,9,13],1])  ,  np.mean(self.preds_3d[[1,5,9,13],2]) )


    def gesturestatemachine(self):

        #   whether the distance between thumber and each finger is less than 30 mm, or larger than 60 mm.
        self.thuber_dis[0] = np.linalg.norm(self.preds_ground[20,:] - self.preds_ground[4,:])   #   distance between thumber and index finger
        self.thuber_dis[1] = np.linalg.norm(self.preds_ground[20,:] - self.preds_ground[8,:])   #   distance between thumber and middle finger
        self.thuber_dis[2] = np.linalg.norm(self.preds_ground[20,:] - self.preds_ground[12,:])   #   distance between thumber and ring finger
        self.thuber_dis[3] = np.linalg.norm(self.preds_ground[20,:] - self.preds_ground[16,:])   #   distance between thumber and litle finger
        if np.max(self.thuber_dis) < 45:
            self.gesturejudge[0] = 1
        else:
            self.gesturejudge[0] = 0

        if np.min(self.thuber_dis) > 60:
            self.gesturejudge[1] = 1
        else:
            self.gesturejudge[1] = 0

        #   whether palm upward, upward is the -y direction
        # v1 = self.handcenter - self.preds_ground[0,:]   
        # v2 = self.preds_ground[1,:] - self.preds_ground[13,:]
        # palm_v = np.cross(v2,  v1)
        # if palm_v[1] > 0 and self.preds_ground[0,1] > self.preds_ground[20,1]:
        if self.preds_ground[0,1] > self.preds_ground[20,1]:
            self.gesturejudge[2] = 1
        else:
            self.gesturejudge[2] = 0


        #   according to gesture, judge the enable flag.
        if self.gesturests == 0:
            if self.gesturejudge[0] == 0 and self.gesturejudge[1] == 1 and self.gesturejudge[2] == 1:
                self.gesturests = 1
        elif self.gesturests == 1:
            if self.gesturejudge[0] == 1 and self.gesturejudge[1] == 0 and self.gesturejudge[2] == 1:
                self.gesturests = 2
                self.gesturejudge[3] = time.time()
        elif self.gesturests == 2:
            if self.gesturejudge[0] == 0 and self.gesturejudge[1] == 1 and self.gesturejudge[2] == 1:
                self.gesturests = 3
            if time.time() - self.gesturejudge[3] > 1.5:
                self.gesturests = 0
        elif self.gesturests == 3:
            if time.time() - self.gesturejudge[3] < 1.5:
                if self.gesturejudge[0] == 1 and self.gesturejudge[1] == 0 and self.gesturejudge[2] == 1:
                    self.gesturests = 4
                    self.trackenable = -self.trackenable
            else:
                self.gesturests = 0
        elif self.gesturests == 4:
            if time.time() - self.gesturejudge[3] > 3.5:
                self.gesturests = 0
                self.ctrlenable = -self.ctrlenable

        if abs(sharedata.com[0] - sharedata.com_f[0])> 64 or abs(sharedata.com[1] - sharedata.com_f[1]) > 64:
            #   when a hand detection position shift seriously, back to initial state
            self.gesturests = 0
        sharedata.com_f = sharedata.com
        

    def dataupdate(self):
        #   update data to handdetection module and robotic arm
        global sharedata

        #   data update for the feedback to hand detection
        sharedata.trackenable = self.trackenable
        sharedata.gesturests = self.gesturests
        if self.trackenable == 1:
            sharedata.handcentercode = (int(self.handcenter[0])<<24) + (int(self.handcenter[1])<<16) + int(self.handcenter[2])
        else:
            sharedata.handcentercode = 0

        #   data update for the commands to regulator module for robotic hand



    def physics(self):
        #   turn the preds to wrist frame.
        #   x axis: in palm plane, vertical to y axis.
        #   y axis: in palm plane, in middle of the index and ring fingers.
        #   z axis: vertical to palm
        preds_3d = self.preds_ground - self.preds_ground[0,:]

        #   determine the y axis, the vector from wrist to MCPs of 4 fingers
        v_if = self.preds_3d[1,:]
        v_if = v_if / np.linalg.norm(v_if)

        v_mf = self.preds_3d[5,:] 
        v_mf = v_mf / np.linalg.norm(v_mf)

        v_rf = self.preds_3d[9,:] 
        v_rf = v_rf / np.linalg.norm(v_rf)

        v_lf = self.preds_3d[13,:] 
        v_lf = v_lf / np.linalg.norm(v_lf)


        v_y =  v_if + v_mf + v_rf + v_lf
        v_y = v_y / np.linalg.norm(v_y)

        #   determine the z axis, based on the cross product
        v_z_if = np.cross(v_if, v_y) 
        v_z_if = v_z_if / np.linalg.norm(v_z_if)

        v_z_mf = np.cross(v_mf, v_y)
        v_z_mf = v_z_mf / np.linalg.norm(v_z_mf)
        if np.dot(v_z_if, v_z_mf) < 0:
            v_z_mf = - v_z_mf

        v_z_rf = np.cross(v_y, v_rf)
        v_z_rf = v_z_rf / np.linalg.norm(v_z_rf)
        if np.dot(v_z_if, v_z_rf) < 0:
            v_z_rf = - v_z_rf

        v_z_lf = np.cross(v_y, v_lf)
        v_z_lf = v_z_lf / np.linalg.norm(v_z_lf)
        if np.dot(v_z_if, v_z_lf) < 0:
            v_z_lf = - v_z_lf

        v_z = v_z_if + v_z_mf + v_z_rf + v_z_lf
        v_z = v_z  / np.linalg.norm(v_z)

        #   determine the x axis
        v_x = np.cross(v_y, v_z)

        #   above x y z are the axes of the wrist frame expressed  in ground frame 
        #   so, from the ground frame to the wrist frame, the transform matrix is:
        tf_grd2rst = np.array([v_x, v_y, v_z])

        #   20 points in wrist frame. size: 20*3
        self.preds_wrist = np.dot(tf_grd2rst,preds_3d[1:,:].T).T
        self.preds_wrist_1 = np.reshape(self.preds_wrist,(60,1))





topic = theTopic(rate_tx=10)
poseesti = poseEstimation()

