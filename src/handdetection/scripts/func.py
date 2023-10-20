#! /usr/bin/env python

import rospy # 1.导包
from std_msgs.msg import UInt32, UInt16, UInt64
from yolov7_tiny_pytorch import predict
from mainloop.msg import hand
import numpy as np
from sensor_msgs.msg import Image as msgimg
import cv2
import threading
from PIL import Image
import struct
import time
from scipy import ndimage
import os


len_pred = 0      #   num of detected hands
# detectres = []  #   a array, each raw means the (top, left, bottom, right) index
conf_sort = []


Info_detect2estimate = hand()


class theTopic(): 
    def __init__(self,pub= True,sub=True, name="handdetection",rate_tx=10,rate_rx=1) -> None:
        global Info_detect2estimate
        ##   annotation: 0: row num, 1: column num, 2: distance mm, 3: height of pixel of half cube size, 4: width of pixel to half cube size, 5: half cube size
        Info_detect2estimate.annotation = list(np.array([1,2,3,4,5,6], dtype= UInt16))
        Info_detect2estimate.data = np.ones((128,128),dtype=UInt16).flatten().tolist()

        self.pkgname = name
        rospy.init_node(self.pkgname)  # 初始化 ROS 节点
        rospy.loginfo("---- "+self.pkgname+" ---- Start the hande detection----")  

        self.esc_press = False
        self.esc_cnt = 0

        self.subimg_w = 64

        self.handarea = np.zeros(shape=(self.subimg_w*2,self.subimg_w*2),dtype=np.uint16)
        self.handarea128 = hand()
        self.handarea128.name = 'image_128_128'

        rate_tx = rospy.get_param('~tx_rate')

        if pub:
            self.pub_pose = rospy.Publisher(self.pkgname +"_cmd", hand, queue_size=1)
            self.rate_tx = rospy.Rate(rate_tx)
            self.cnt_tx = 0    #   low 16 bits
            self.cmd = 0    #   high 16 bits

        if sub:
            rospy.sleep(1)
            self.rx_main = UInt64()
            self.rx_main_cnt = np.uint32(0)
            self.rx_main_cmd = np.uint32(0)
            self.rx_kinect = msgimg()
            self.rx_poseesti = UInt32()
            #   receive cmd info from mainloop pkg
            self.sub_main = rospy.Subscriber(name="mainloop"+"_cmd",data_class=UInt64,callback=self.sub_main_callback, queue_size=2)
            #   receive image from kinect
            self.sub_kinect = rospy.Subscriber(name="kinect2/sd/image_depth_rect",data_class=msgimg,callback=self.sub_kinect_callback, queue_size=2)
            #   receive feedback from estimation pkg
            self.sub_poseesti_fbk = rospy.Subscriber(name="poseestimation_state",data_class=UInt32,callback=self.sub_poseesti_fbk_callback,queue_size=2)
            #   receive count
            self.cnt_rx_main = 0
            self.cnt_rx_kinect = 0
            self.cnt_rx_poseesti_fbk = 0

            self.sub_kinect_newflag = False
            self.sub_poseesti_fbk_doneflag = False

            self.sub_kinect_h = 424
            self.sub_kinect_w = 512
            self.sub_kinect_crop = 416
            
            self.imgarr = np.zeros((self.sub_kinect_h,self.sub_kinect_w ),dtype=np.uint16)
            self.imgarrcrop = np.zeros((self.sub_kinect_crop,self.sub_kinect_crop ),dtype=np.uint16)
            #   crop area index, top,bottom, left, right
            self.cropidx = np.array( [(self.sub_kinect_h-self.sub_kinect_crop)*0.5, (self.sub_kinect_h+self.sub_kinect_crop)*0.5, 
                                      (self.sub_kinect_w-self.sub_kinect_crop)*0.5,  (self.sub_kinect_w + self.sub_kinect_crop)*0.5],dtype=np.uint16)

            # self.add_thread = threading.Thread(target=self.thread_job)
            # self.add_thread.start()
            self.saveflg = rospy.get_param('~saveflg')
            if self.saveflg:
                tm = time.localtime()
                timestr = str(tm.tm_year)+'-'+str(tm.tm_mon).rjust(2,'0') +'-'+ str(tm.tm_mday).rjust(2,'0') +'-'+ str(tm.tm_hour).rjust(2,'0') +'-'+ str(tm.tm_min).rjust(2,'0') +'-'+ str(tm.tm_sec).rjust(2,'0') 
                self.savepath = '/homeL/zheng/Downloads/temp/1depth/' + timestr + '/' 
                os.mkdir(self.savepath)

    def thread_job():
        rospy.spin()

    def publish(self):
        global Info_detect2estimate
        #   whether the handdetection is done
        # print('annotation: ',Info_detect2estimate.annotation, '.  type: ',type(Info_detect2estimate.annotation))
        if Info_detect2estimate.annotation[2] > 0 :
            self.handarea128.annotation = Info_detect2estimate.annotation
            self.handarea128.cmd = self.rx_main_cmd
            self.handarea128.cnt = self.rx_main_cnt
            self.handarea128.data = Info_detect2estimate.data
            # self.handarea128.data = np.ones((128,128),dtype=UInt16).flatten().tolist()
            # Info_detect2estimate.annotation = np.array([1,2,3,4,5,6],dtype=np.uint16).flatten().tolist
            self.pub_pose.publish(self.handarea128)
            
            self.cnt_tx += 1
            # if self.cnt_tx % 40== 0:
            # rospy.loginfo("----2 %s topic tx, rxcnt: %d ----",self.pkgname, self.rx_main_cnt)

            Info_detect2estimate.annotation[2] = 0
            self.rate_tx.sleep()

    def sub_main_callback(self,msg):
        self.rx_main = msg
        self.rx_main_cnt = self.rx_main.data & 0xffffffff
        self.rx_main_cmd = self.rx_main.data >> 32
        if self.rx_main_cmd == 1:
            self.esc_press = True
        if self.esc_press == True:
            self.esc_cnt += 1

        # if self.cnt_rx_main %40 == 0:
        # rospy.loginfo("----2 %s rx main topic: data: %d, cnt: %d ----",self.pkgname , self.rx_main.data ,  self.cnt_rx_main)
        
        self.cnt_rx_main += 1

        # print('in hand detection, cnt and rx_main_cmd: ', self.rx_main_cnt, self.rx_main_cmd)

    def sub_kinect_callback(self,msg):
        global len_pred, conf_sort, Info_detect2estimate

        self.sub_kinect_newflag = True
        self.rx_kinect = msg

        self.cnt_rx_kinect += 1
        
        #   turn raw depth data (in sensor.Image format) to array
        length = int(len(self.rx_kinect.data)/2)
        img_deserial = np.frombuffer(self.rx_kinect.data, np.uint8)
        img_array = img_deserial.reshape(length,2)
        self.imgarr = np.dot(img_array , np.array([[1],[256]])).reshape(self.rx_kinect.height,self.rx_kinect.width)

        #   416 * 416, ignore the over-far pixel, over 2000 mm
        self.imgarrcrop = self.imgarr[self.cropidx[0]:self.cropidx[1], self.cropidx[2]:self.cropidx[3]]
        self.imgarrcrop = np.clip(self.imgarrcrop,0,2000)

        showimage = True
        if showimage :
            depth_img255 = ( ( self.imgarrcrop - 0 ) / (2001 -0 ) * 255).astype('uint8')

            if Info_detect2estimate.annotation[0] > 0 :
                cv2.rectangle(depth_img255,(Info_detect2estimate.annotation[1]-Info_detect2estimate.annotation[4],Info_detect2estimate.annotation[0]-Info_detect2estimate.annotation[3]),
                              (Info_detect2estimate.annotation[1]+Info_detect2estimate.annotation[4],Info_detect2estimate.annotation[0]+Info_detect2estimate.annotation[3]),(0, 0, 0),thickness=3)
            cv2.imshow("image depth, hand detection received from kinect",depth_img255)

            if self.saveflg:
                if self.rx_main_cnt % 10 == 0:
                    graypath = self.savepath + str(self.rx_main_cnt).rjust(5,'0') + '.png'
                    cv2.imwrite(graypath,depth_img255)
                    # np.savetxt("/homeL/zheng/ros_python/tempsave/" + str(self.cnt_rx_kinect).rjust(5,'0') + '.txt', depth_img255,fmt='%d')


            cv2.waitKey(10)

    def sub_poseesti_fbk_callback(self,msg):
        
        self.cnt_rx_poseesti_fbk += 1
        # self.rx_poseesti = msg.data
        self.rx_poseesti = msg
        

        #   reflect whether the hand estimation is done, if done, no hand detection is needed.
        if self.rx_poseesti.data > 0 :
            self.distance = self.rx_poseesti.data & 0xffff
            self.pix_x = (self.rx_poseesti.data & 0xff000000)>>24
            self.pix_y = (self.rx_poseesti.data & 0xff0000) >> 16

            self.sub_poseesti_fbk_doneflag = True
        else:
            self.sub_poseesti_fbk_doneflag = False


            




class handDetection():
    def __init__(self) -> None:
        self.cnt = 0
        self.source = rospy.get_param('~source')
        self.endcnt = rospy.get_param('~endcnt')
        self.pred=predict.pred()
        self.len_pred = 0

        self.img_w = 208
        self.subimg_w = 64
        self.time0 = time.time()

        self.fw = 512 / 70 * 57.3
        self.fh = 424 / 60 * 57.3

        self.com = np.array((0,0,0), np.uint16)

        self.main_cmd_f = 2

    def run(self,main_cmd = 0):
        if self.main_cmd_f == 2 and main_cmd == 0:
            self.start_cmd = time.time()
        self.main_cmd_f = main_cmd
        if True:
            #   if the estimation is not done, or in the main loop, the 'pause' is just pressed
            if topic.sub_poseesti_fbk_doneflag == False or (time.time() - self.start_cmd < 2) :
                global len_pred, conf_sort

                self.cnt += 1
                

                #   predict the dataset
                if self.source == 0:
                    self.pred.run_dataset()
                elif self.source == 1:
                    #   predict real-time depth images
                    if topic.sub_kinect_newflag and True:

                        res0 = self.pred.run_realtime(image=topic.imgarrcrop)
                        #  [ top, left, bottom, right ][ confidence ]



                        #   if no hand found
                        if  res0 == None :
                            self.len_pred = 0
                            detectres = None
                            
                        else:
                            #   the position array of the found hands.  confidence array.
                            resarr = np.array(res0[0],dtype=np.float32)
                            confres = np.array(res0[1],dtype=np.float32)

                            detectres0 = np.clip( resarr[:, :4], 0.0 , topic.sub_kinect_crop - 1)
                            detectres = detectres0.astype(int)

                            #   the number of found hands
                            if len(resarr.shape) == 1:
                                self.len_pred = 1
                                conf_sort = 0
                            else:
                                self.len_pred = resarr.shape[0]
                                self.detectrefine(detectres,confres)
                            #   save the conf_sort and len_pred in global variables
                                conf_sort = self.conf_sort
                        len_pred = self.len_pred

                        topic.sub_kinect_newflag = False
            else:
                rate = 0.75                     #   proportional control of visual servo control
                delt_x = int((topic.pix_x - 64) / 64 * self.cube_pix_num_w_half * rate)
                delt_y = int((topic.pix_y - 64) / 64 * self.cube_pix_num_h_half * rate)
                # delt_z = topic.distance
                self.com[0] = self.com[0] + delt_y
                self.com[1] = self.com[1] + delt_x
                self.com[2] = topic.distance

            self.cube_img(cube_size=200)


    def detectrefine(self,pos,conf):
        #   according to the farest distance of each rectangle to image center, 
        #   according to prediction box area
        # sort candidat

        self.dist = np.zeros_like(conf)
        self.refine_conf = np.zeros_like(conf)
        self.area = np.zeros_like(conf)
        for i in range(self.len_pred):

            # self.dist[i] = np.linalg.norm([pos[i,0] + pos[i,2] - topic.sub_kinect_crop , pos[i,1] + pos[i,3] - topic.sub_kinect_crop]) / 2

#           distance of each rectangle to the image center
            d = np.zeros(shape=4 ,dtype= np.float64)
            d_cnt = 0
            for hh in [pos[i,0],pos[i,2]]:
                for ww in [pos[i,1],pos[i,3]]:
                    d[d_cnt] = np.linalg.norm([hh - self.img_w , ww - self.img_w])
                    d_cnt += 1
            self.dist[i] = np.max(d)

#           area of each rectangle
            self.area[i] = (pos[i,2] - pos[i,0]) * (pos[i,3] - pos[i,1])
#           new confidence
            self.refine_conf[i] =  conf[i] * ((self.img_w / (self.img_w +   1*self.dist[i]))**2)  / np.sqrt(self.area[i])
        self.conf_sort = np.argsort(self.refine_conf)[::-1]       #   sort (0~3) means the privilege of a hand, 0 refer to the most possible hand

        #   top bottom left right #   row, col
        # self.center = np.array(np.around([ 0.5* ( pos[self.conf_sort[0],0] + pos[self.conf_sort[0],2])  ,  0.5 * (pos[self.conf_sort[0],1] + pos[self.conf_sort[0],3] ) ]),dtype=np.uint16 )
        # self.center = np.clip(self.center, self.subimg_w, self.img_w * 2 - self.subimg_w)

        cubesize = 200
        self.com = self.getcom(pos,cubesize=cubesize)

        # self.cube_img(cube_size=cubesize)


    def getcom(self,pos,cubesize=200):
        '''
        input: the detected results (x y position)
                the input is a array with multi results, according to privilege self.conf_sort, choose the most likely area
                then, cauculate the center of the mass, com, in the 416 image coordinate
        '''
        handarr = topic.imgarrcrop[ pos[self.conf_sort[0],0] : pos[self.conf_sort[0],2],  pos[self.conf_sort[0],1] : pos[self.conf_sort[0],3] ]


        #   ---- save the detected rectangle from yolo. ----
        # handarr255 = (handarr / 2000 * 255).astype('uint8')
        # if topic.cnt_rx_kinect % 20 == 0:
        #     np.savetxt("/homeL/zheng/ros_python/tempsave/handdetect/" + str(topic.cnt_rx_kinect).rjust(5,'0') + '.txt', handarr255,fmt='%d')
        #     cv2.imwrite("/homeL/zheng/ros_python/tempsave/handdetect/" + str(topic.cnt_rx_kinect).rjust(5,'0') + '.png',handarr255)


        handarr[handarr < 300] = 0
        handarr[handarr > 1999] = 0

        self.handarr = handarr
        handarr_com = ndimage.measurements.center_of_mass(handarr > 0) 
        handarr_num = np.count_nonzero(handarr)

        #   ----    solve the distance of COM ----
        if False:
            distance = handarr.sum()/handarr_num
        else:
        #   another idea of solving the com distance
            distance = 0
            bin_n = 220
            hist_accum = 0
            pic_hist,bins  = np.histogram(handarr,bins=bin_n,range=(300,1999))
            hist_valid_sum = pic_hist.sum()
            for hist_i in range(6,bin_n):
                hist_accum += pic_hist[hist_i]
                if hist_accum > 0.015 * hist_valid_sum :
                    distance = bins[hist_i-6] + cubesize * 0.5
                    break


        if handarr_num == 0:
            return np.array((0,0,0), np.uint16)
        else:
            return np.array((handarr_com[0] + pos[self.conf_sort[0],0] ,handarr_com[1] + pos[self.conf_sort[0],1] , distance),np.uint16)
                            #   row             columon  in the 416 image, com


    def cube_img(self,cube_size = 200.0,cubepixel=(128,128)):
        '''
        based on the com, select a cube with specified size and project the corresponding depth info to a 128*128 sub-img
        with value from -1 to 1, background is 1
        '''

        global Info_detect2estimate

        if self.com[2] > 0 :

            bkg = int(self.com[2] + cube_size / 2 )
            
            len_w = int(cube_size / self.com[2] * self.fw /2)  #   pixel nums *** of half cube size
            len_h = int(cube_size / self.com[2] * self.fh /2)
            len_d = cube_size / 2

            self.cube_pix_num_w_half = len_w
            self.cube_pix_num_h_half = len_h
            self.cube_d_half = len_d


            # print('-'*30, 'com: h,w ',self.com,'len w: ',len_w, ' len h', len_h, 'shape of imgarrcrop: ',topic.imgarrcrop.shape)
            cube_ori = topic.imgarrcrop[max(0,self.com[0]-len_h):min(self.img_w *2 ,self.com[0]+len_h), max(0,self.com[1]-len_w):min(self.img_w * 2,self.com[1] + len_w) ]
            # print('create cube_ori with size:  ',cube_ori.shape)
            #   in case of the cube is out of view field, pad it for lacked pixel
            cube_ori = np.pad(cube_ori,((abs(self.com[0]-len_h)-max(0,self.com[0]-len_h), abs(self.com[0] + len_h) - min(self.img_w *2 ,self.com[0]+len_h) ),
                                        ( abs(self.com[1]-len_w) - max(0,self.com[1]-len_w)  , abs(self.com[1] + len_w) - min(self.img_w * 2, self.com[1] + len_w))))
            # print('after pad, cube_ori with size:  ',cube_ori.shape)
            # print(cube_ori)
            #   set the pixel that out of the cube to background, the bkg is the distance out of work length. Here, set to the furthest distance of the cube as bkg
            msk1 = np.bitwise_or(cube_ori < (self.com[2] - len_d), cube_ori > (self.com[2] + len_d))
            # msk2 = np.bitwise_and(cube_ori > (self.com[2] + len_d), cube_ori != 0)
            
            cube_ori[msk1] = bkg
            # print('size of cube_ori: ',cube_ori.shape,'cube_ori should be in range of self.com[2]-len_d ',self.com[2]-len_d, 'to self.com[2]+len_d: ',self.com[2]+len_d, 'and cube ori is: ',cube_ori)
            img_128 = np.ones(cubepixel,np.float32) * bkg
            if len_w > len_h:
                #   if the width is larger than the height, the hand image, cube size (200,200,200)
                w_resize = cubepixel[1]
                h_resize = int(cubepixel[1] / len_w * len_h)
                h_ept0 = int((cubepixel[0] - h_resize) / 2)
                h_ept1 =  h_resize + h_ept0

                # print(w_resize,h_resize,cube_ori.shape)
                cubeimg_resize = cv2.resize(cube_ori, dsize=(w_resize,h_resize),interpolation=cv2.INTER_NEAREST)
                img_128[h_ept0:h_ept1,:] = cubeimg_resize
            else:
                h_resize = cubepixel[0]
                w_resize = int(cubepixel[0]/len_h * len_w)
                w_ept0 = int((cubepixel[1] - w_resize)/2)
                w_ept1 = w_resize + w_ept0
                cubeimg_resize = cv2.resize(cube_ori,(w_resize,h_resize),interpolation=cv2.INTER_NEAREST)
                img_128[:,w_ept0:w_ept1] = cubeimg_resize

            self.cube_nonnorm = img_128
            self.cube_norm = (img_128 - self.com[2]) / len_d


            cube_nonnorm = self.cube_nonnorm.astype(np.uint16)
            Info_detect2estimate.data = cube_nonnorm.flatten().tolist()
            tmp = np.array([self.com[0], self.com[1], self.com[2], len_h, len_w, int(len_d) ],dtype=UInt16)
            Info_detect2estimate.annotation = list(tmp)
            #   annotation: 0: row num, 1: column num, 2: distance mm, 3: height of pixel of half cube size, 4: width of pixel to half cube size, 5: half cube size
            # print(Info_detect2estimate.annotation)
            # print(self.cube_norm.shape,'\n',self.cube_norm)

        else:
            # print( 'no hand found ')
            pass


topic = theTopic()
handdete = handDetection()

