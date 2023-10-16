#!/usr/bin/env python
from __future__ import print_function
import rospy

import bio_ik_msgs
import bio_ik_msgs.msg
import bio_ik_msgs.srv
from moveit_msgs.msg import RobotState
import moveit_msgs.srv
from moveit_msgs.srv import GetPositionFK
from moveit_msgs.srv import GetPositionFKRequest
from moveit_msgs.srv import GetPositionFKResponse
from sensor_msgs.msg import JointState
import trajectory_msgs.msg
import numpy as np
import geometry_msgs.msg
from std_msgs.msg import Float32MultiArray, Int16MultiArray
from mainloop.msg import hand
import os
import time
import moveit_commander
from scipy.spatial.transform import Rotation as R
np.set_printoptions(formatter={'float': '{:.5f}'.format})

# rospy.init_node("bio_ik_service_example")


#   actually, this py file acts as a client of the bio_ik.
#   

#------------------------------------------------------------------

# print('---- current joint values are: ',joint_value)
# # #------------------------------------------------------------------

class theBioikModule(): 
    def __init__(self,pub= True,sub=True, name="bio_ik_hand",rate_tx=10,rate_rx=1) -> None:
        rospy.loginfo('---- init the bio_ik module ----')
        self.pkgname = name
        rospy.init_node(self.pkgname)  # 2.初始化 ROS 节点
        rospy.loginfo("---- "+self.pkgname+" ---- Start the bio_ik ----")  #3.日志输出 HelloWorld

        self.endcnt = rospy.get_param('~endcnt')


        self.esc_cnt = 0
        self.esc_press = False
        self.trackflg = 0
        self.trackcnt = 0

        #   -------------------------------------------------------------------------------------------------------------------
        # print('test two bio_iks and two robot models ... ')
        # rbt = moveit_commander.RobotCommander()
        # # print('robot and name: ',rbt._robot, rbt._name)
        # print('group names: ', rbt.get_group_names())


        #   ----------------------------------------------- bio_ik for  physical model of hand -------------------------------------
        #   init the bio_ik
        rospy.loginfo('---- init the for  physical model  ----')

        rospy.wait_for_service("/bio_ik/get_bio_ik")
        self.get_bio_ik_phy = rospy.ServiceProxy("/bio_ik/get_bio_ik", bio_ik_msgs.srv.GetIK)

        self.req_ik_phy = bio_ik_msgs.msg.IKRequest()
        self.req_ik_phy.robot_description = 'physical_hand'
        self.req_ik_phy.group_name = "hand"

        self.req_ik_phy.timeout.secs = 0
        self.req_ik_phy.timeout.nsecs = 20 * 1000 * 1000

        self.req_ik_phy.approximate = True

        self.req_ik_phy.avoid_collisions = False

        phy_j_states = rospy.wait_for_message('/joint_states',JointState,3)
        self.req_ik_phy.robot_state.joint_state = phy_j_states
        print('physical hand joint states . name: ', phy_j_states.name)

        #   20 points for goal setup
        self.frame_name = ['if_h', 'if_1_b', 'if_2_b', 'if_3_b',
                           'mf_h', 'mf_1_b', 'mf_2_b', 'mf_3_b',
                           'rf_h', 'rf_1_b', 'rf_2_b', 'rf_3_b',
                           'lf_h', 'lf_1_b', 'lf_2_b', 'lf_3_b',
                           'th_y', 'th_1_b', 'th_2_b', 'th_3_b']
        self.goal_len = len(self.frame_name)
        self.regularized_goal = np.zeros((self.goal_len,3),dtype=np.float32)
        
        # self.frame_location = np.zeros((self.goal_len,3),dtype=np.float32)   #   real location
        self.goal_location  = np.zeros((self.goal_len,3),dtype=np.float32)   #   goal location

        #   all joint name, 47
        self.joint_name = ['j_palm_h', 'j_if_w', 'j_if_h', 'j_if_p', 'j_if_1_a', 'j_if_1_b', 'j_if_2_a', 'j_if_2_b', 'j_if_3_a', 'j_if_3_b',    #   0 ~ 9
                           'j_mf_w', 'j_mf_h', 'j_mf_p', 'j_mf_1_a', 'j_mf_1_b', 'j_mf_2_a', 'j_mf_2_b', 'j_mf_3_a', 'j_mf_3_b',                #   10 ~ 18
                           'j_rf_w', 'j_rf_h', 'j_rf_p', 'j_rf_1_a', 'j_rf_1_b', 'j_rf_2_a', 'j_rf_2_b', 'j_rf_3_a', 'j_rf_3_b',                #   19 ~ 27
                           'j_lf_w', 'j_lf_h', 'j_lf_p', 'j_lf_1_a', 'j_lf_1_b', 'j_lf_2_a', 'j_lf_2_b', 'j_lf_3_a', 'j_lf_3_b',                #   28 ~ 36
                           'j_th_x', 'j_th_z', 'j_th_y', 'j_th_p',   'j_th_1_a', 'j_th_1_b', 'j_th_2_a', 'j_th_2_b', 'j_th_3_a', 'j_th_3_b'     #   37 ~ 46
                           ]
        self.joint_value = np.ones(len(self.joint_name),dtype=float)         #   the regularized joint values of the hand
        self.joint_len = len(self.joint_value )

        #   map the names from the joint_name --> joint_cons_name, index
        self.joint_name_map = np.array([0, 1, 2, 5, 7, 9,
                                        10, 11, 14, 16, 18,
                                        19, 20, 23, 25, 27,
                                        28, 29, 32, 34, 36,
                                        37, 38, 39, 42, 44, 46])

        #   constraint joint name, the translation joint
        self.joint_cons_name = ['j_palm_h', 'j_if_w', 'j_if_h','j_if_1_b','j_if_2_b','j_if_3_b',        #   0 ~ 5
                                'j_mf_w', 'j_mf_h', 'j_mf_1_b','j_mf_2_b','j_mf_3_b',                   #   6 ~ 10
                                'j_rf_w', 'j_rf_h', 'j_rf_1_b','j_rf_2_b','j_rf_3_b',                   #   11 ~ 15
                                'j_lf_w', 'j_lf_h', 'j_lf_1_b','j_lf_2_b','j_lf_3_b',                   #   16 ~ 20
                                'j_th_x', 'j_th_z', 'j_th_y',  'j_th_1_b','j_th_2_b','j_th_3_b'         #   21 ~ 26
                                ]
        
        self.joint_cons_value = np.zeros(len(self.joint_cons_name),dtype=float)
        self.joint_cons_value_new = np.zeros(len(self.joint_cons_value),dtype=float)
        self.joint_cons_cnt = 0
        self.joint_cons_len = len(self.joint_cons_value )


        #   init the goal list
        for i in range(self.goal_len):
                self.req_ik_phy.position_goals.append(bio_ik_msgs.msg.PositionGoal())
                self.req_ik_phy.position_goals[-1].link_name = self.frame_name[i]
                self.req_ik_phy.position_goals[-1].position.x = 0.0
                self.req_ik_phy.position_goals[-1].position.y = 0.0
                self.req_ik_phy.position_goals[-1].position.z = 0.0

        # print('the robot_models of the bio_ik: \n')
        # print(len(self.get_bio_ik_phy.robot_models),'\n')
        # print(self.get_bio_ik_phy.robot_models)


        #   --------------------------------------------- forward kinematics ----------------------------------------------------
        rospy.loginfo('---- init the for  forward kine  ----')
        self.fk_srv = rospy.ServiceProxy('/compute_fk', GetPositionFK)
        self.fk_srv.wait_for_service()

        self.req_fk = GetPositionFKRequest()
        self.req_fk.header.frame_id = 'base_link'
        self.req_fk.fk_link_names = self.frame_name

        # self.j_states_phy = rospy.Subscriber('/phy_hand/joint_states', JointState, self.update_j_sts, queue_size=1)
        # self.r_sts_phy = None
        # phy_j_states = None
        # phy_j_states = rospy.wait_for_message('/phy_hand/joint_states',JointState, 2 )
        # # print('phy fk, j_states_phy: \n',self.j_states_phy)
        # if self.j_states_phy == None:
        #     rospy.logerr('**** NO human hand physical model joint_state received in 10 seconds ****')
        # else:
        self.req_fk.robot_state.joint_state = phy_j_states
        # print('physical hand joint states . position: ',phy_j_states.position)

        # print('the self.j_sts has position type and len as: ',type(self.j_sts.position),len(self.j_sts.position))






       #   ----------------------------------------------- bio_ik for  shadow robot + ur -------------------------------------
       
        #   init the bio_ik
        
        if False:

            rospy.loginfo('---- init the for  shadow hand  ----')
            rospy.wait_for_service("/bio_ik2/get_bio_ik2")
            self.get_bio_ik_sha = rospy.ServiceProxy("/bio_ik2/get_bio_ik2", bio_ik_msgs.srv.GetIK)

            self.req_ik_sha = bio_ik_msgs.msg.IKRequest()
            self.req_ik_sha.robot_description = 'ursrh'
            self.req_ik_sha.group_name = "srur"

            self.req_ik_sha.timeout.secs = 0
            self.req_ik_sha.timeout.nsecs = 20 * 1000 * 1000

            self.req_ik_sha.approximate = True
            self.req_ik_sha.avoid_collisions = False



            # rospy.loginfo('---- init the wait for message ursrh/robot_states ----')
            # # self.r_sts_sha = None
            # self.r_sts_sha = rospy.wait_for_message('/ursrh/move_group/status',RobotState,10)
            # self.req_ik_sha.robot_state = self.r_sts_sha
            # rospy.loginfo('---- init the wait for message ursrh/robot_states done ----')
            #   5 tip points for goal setup
            self.frame_name_sha = ['rh_fftip', 'rh_mftip', 'rh_rftip', 'rh_lftip', 'rh_thtip' ]
            self.goal_len_sha = len(self.frame_name_sha)
            self.goal_position_sha = np.zeros((self.goal_len_sha,3),dtype=np.float32)

            #   all joint name, 47
            self.joint_name_sha = ['ra_elbow_joint', 'ra_shoulder_lift_joint', 'ra_shoulder_pan_joint', 'ra_wrist_1_joint', 'ra_wrist_2_joint', 'ra_wrist_3_joint',     #   0 ~ 5 ur
                            'rh_WRJ1', 'rh_WRJ2',                                        #   6 ~ 7
                            'rh_FFJ1', 'rh_FFJ2', 'rh_FFJ3', 'rh_FFJ4',                  #   8 ~ 11
                            'rh_MFJ1', 'rh_MFJ2', 'rh_MFJ3', 'rh_MFJ4',                  #   12 ~ 15
                            'rh_RFJ1', 'rh_RFJ2', 'rh_RFJ3', 'rh_RFJ4',                  #   16 ~ 19
                            'rh_LFJ1', 'rh_LFJ2', 'rh_LFJ3', 'rh_LFJ4',  'rh_LFJ5',      #   20 ~ 24
                            'rh_THJ1', 'rh_THJ2', 'rh_THJ3', 'rh_THJ4'                   #   25 ~ 28
                            ]
            self.joint_value_sha = np.zeros(len(self.joint_name_sha),dtype=float)         #   the regularized joint values of the hand
            self.joint_len_sha = len(self.joint_value_sha )

            #   init the goal list
            for i in range(self.goal_len_sha):
                    self.req_ik_sha.position_goals.append(bio_ik_msgs.msg.PositionGoal())
                    self.req_ik_sha.position_goals[-1].link_name = self.frame_name_sha[i]
                    self.req_ik_sha.position_goals[-1].position.x = 0.0
                    self.req_ik_sha.position_goals[-1].position.y = 0.0
                    self.req_ik_sha.position_goals[-1].position.z = 0.0


        self.tfRot = R.from_euler("X", -90 ,degrees = True)

        #   ----------------------------------------------- data save ------------------------------------------------------------
        rospy.loginfo('---- init the for  data save  ----')
        self.saveflg = rospy.get_param('~saveflg')
        if self.saveflg:
            tm = time.localtime()
            timestr = str(tm.tm_year)+'-'+str(tm.tm_mon).rjust(2,'0') +'-'+ str(tm.tm_mday).rjust(2,'0') +'-'+ str(tm.tm_hour).rjust(2,'0') +'-'+ str(tm.tm_min).rjust(2,'0') +'-'+ str(tm.tm_sec).rjust(2,'0') 
            self.savepath = '/homeL/zheng/Downloads/temp/2bioik/' + timestr + '/' 
            os.mkdir(self.savepath)
            
            self.filehandle = open(self.savepath + timestr + '.txt','w')
            self.filehandle.write('---- save the data ----'+'\n')
            # self.filehandle.write(str(topic.Rot.as_dcm())+'\n')


        #   ----------------------------------------------- topic ------------------------------------------------------------
        if pub:
            self.pub_phy_hand_points = rospy.Publisher(self.pkgname +"_res", data_class = Int16MultiArray, queue_size=1)
            self.rate_tx = rospy.Rate(rate_tx)
            self.cnt_tx = 0    #   low 16 bits
            self.cmd = 0    #   high 16 bits
            self.tx = Int16MultiArray()

        if sub:
            rospy.sleep(1)
            self.rx = hand()
            self.sub_key_point = rospy.Subscriber(name="poseestimation_cmd",data_class = hand, callback=self.sub_callback, queue_size=2)
            self.cnt_rx = 0
            self.mainloop_cnt = 0

        #****************************************************************************************************************************
        


    def publish_regularized_angle(self,saveres=False):

        info = np.array([self.mainloop_cmd, self.mainloop_cnt, self.trackcnt])

        #   include the cmd, cnt and track into the target position data. overall length:3 + 63
        regu_goal_world_amplify = (self.regu_goal_world * 10000).astype(np.int16)
        txdata = np.vstack((info, regu_goal_world_amplify )).flatten().tolist()
        self.tx.data = txdata
        self.pub_phy_hand_points.publish(self.tx)
        
        # print('----5 published data is: ',self.tx)
        rospy.loginfo("----5 %s pub,   mainloop_cnt: %d     ----",self.pkgname,  self.mainloop_cnt)

        self.cnt_tx += 1

        # j_vals = np.array(self.tx.data)
        # if self.mainloop_cnt % 10 == 0:
        #     rospy.loginfo("---- 5 bio_ik %s topic tx with cmd: %d",self.pkgname, self.rx.cmd)
            # print('---- 5 bio_ik joint values: %d ----',j_vals)

        if self.saveflg :
            if self.mainloop_cnt % 10 == 0:
                self.filehandle.write('cnt: '+str(self.mainloop_cnt)+'\n')
                self.filehandle.write('origian posiiton goal: '+str(self.goal_location)+'\n')
                self.filehandle.write('regularized position: '+str(self.regularized_goal)+'\n')
                self.filehandle.write('*'*30 + '\n'*2)

    def sub_callback(self,msg):

        #   data from poseestimaiton, original unit: 0.01 mm, turn the unit to m
        #   after regularizing, the new 20 pints are sent to next module

        self.joint_cons_cnt += 1
        self.rx = msg
        self.mainloop_cnt = self.rx.cnt
        self.mainloop_cmd = self.rx.cmd
        self.trackflg = self.rx.annotation[0]
        rospy.loginfo('---- 5 in the fun bio ik, mainloop cnt %d, cmd %d, and trackflg %d ----',self.mainloop_cnt, self.mainloop_cmd, self.trackflg)
        # print('in fun bio ik, mainloop cnt and cmd and trackflg: ', )
        if self.mainloop_cmd == 1:
            self.esc_press = True
        if self.esc_press == True:
            self.esc_cnt += 1
        
        if self.trackflg > 0:
            self.trackcnt += 1
        else:
            self.trackcnt = 0

        rx_data0 = np.array(self.rx.data, dtype=np.float32)
        rx_data = np.reshape(rx_data0,(24,3))

        
        self.key_points = rx_data[0:20,:] * 0.00001     #   turn to unit 1m
        self.tf_grd2rst = rx_data[20:23,:] * 0.001
        self.wrist_p0 = rx_data[23,:] * 0.1 * 0.001     #   wrist position in camera centerred ground frame. unit: 1 m
        self.wrist_p = np.expand_dims(self.wrist_p0, axis= 0)

        # if self.mainloop_cnt % 10== 0:
        #     rospy.loginfo('----5 bio_ik_phy_hand receive the key points, and in total received: %d, at rx_cmd: %d', self.joint_cons_cnt, self.rx.cmd)
            # print('----5 bio_ik regula, 20 goal positions in wrist frame, unit: m : ',self.key_points)
        #   from key_points --> goal_location, choose 20 points and keep in proper order, in wrist frame, unit: meter
        self.goal_location = self.key_points
        # print('goal_locations: \n ', self.goal_location[0:4,:])


        #   for bio_ik
        self.config_goal_phy()
        print('---- 5 fun bio ik, before bio ik: ', time.time())
        self.call_bio_ik_phy()
        print('---- 5 fun bio ik, after bio ik and before fk: ', time.time())
        self.config_cons_phy()

        self.get_fk_phy()
        print('---- 5 fun bio ik, after fk: ', time.time())

        self.tf_rst2world()

        # self.config_goal_sha()
        # self.call_bio_ik_sha()

        self.cnt_rx += 1
        # if self.cnt_rx %40 == 0:
        #     rospy.loginfo("----5 %s rx 20 points data: %d, cnt: %d ----",self.pkgname , np.array(self.rx.data) ,  self.cnt_rx)

        # self.tx.data = self.joint_value.flatten().tolist()
        self.publish_regularized_angle(saveres = True)
        
    def config_goal_phy(self):
        for i in range(self.goal_len):
            # self.req_ik_phy.position_goals[i].link_name = self.frame_name[i]
            self.req_ik_phy.position_goals[i].position.x = self.goal_location[i,0]
            self.req_ik_phy.position_goals[i].position.y = self.goal_location[i,1]
            self.req_ik_phy.position_goals[i].position.z = self.goal_location[i,2]

    def call_bio_ik_phy(self):
        # after setting up the self.req_ik_phy, the response can be gotten

        # rospy.loginfo('---- before ik execute, one line command')
        resp_phy = self.get_bio_ik_phy(self.req_ik_phy).ik_response
        # rospy.loginfo('---- after ik execute, one line command')
        self.joint_position_tuple = resp_phy.solution.joint_state.position           #   tuple, should be sent to moveit for forward kinematics calculation
        self.req_ik_phy.robot_state.joint_state.position = self.joint_position_tuple    #   update the start optimization point for bio_ik
        self.joint_value = np.asarray(self.joint_position_tuple,dtype=np.float32)
        # rospy.loginfo('----5 physical hand regularized joints: %s ----', str(self.joint_position_tuple[0:5]))
    

    def config_cons_phy(self):
        #   based on the result from bio_ik, record joint_cons_value_new,
        #   then, based on it, update the joint_cons_value.
        alpha = (self.joint_cons_cnt - 1 ) / self.joint_cons_cnt 
        if self.joint_cons_cnt > 99:
            alpha = 0.99
        for j in range(self.joint_cons_len):
            # print('len of joint values: ', len(self.joint_value))
            self.joint_cons_value_new[j] = self.joint_value[self.joint_name_map[j]]
            self.joint_cons_value = self.joint_cons_value * alpha + (1 - alpha) * self.joint_cons_value_new


        #   all constraint links should have specific lengths.
        if self.joint_cons_cnt == 1:
            self.req_ik_phy.joint_variable_goals = []
            for i in range(self.joint_cons_len):
                self.req_ik_phy.joint_variable_goals.append(bio_ik_msgs.msg.JointVariableGoal())
                self.req_ik_phy.joint_variable_goals[-1].variable_name = self.joint_cons_name[i]
                self.req_ik_phy.joint_variable_goals[-1].variable_position = self.joint_cons_value[i]
                self.req_ik_phy.joint_variable_goals[-1].weight = 1
                self.req_ik_phy.joint_variable_goals[-1].secondary = False

            # print('at initial, length of joint_variable_goals: ',len(self.req_ik_phy.joint_variable_goals))
            # print('and it is: ',self.req_ik_phy.joint_variable_goals)
        else:
            for i in range(self.joint_cons_len):
                self.req_ik_phy.joint_variable_goals[i].variable_position = self.joint_cons_value[i]

    def get_fk_phy(self):
        #   forward kinematic calculation
        self.req_fk.robot_state.joint_state.position    = self.joint_position_tuple
        # print('req_fk.robot_state: ', self.req_fk.robot_state )
        resp_fk = self.fk_srv.call(self.req_fk)
        # print('the fk respose pose :','-- \n', type(resp_fk.pose_stamped), type(resp_fk.pose_stamped[2]),'-- \n', type(resp_fk.pose_stamped[2].pose))
        for i, posestamped in enumerate(resp_fk.pose_stamped) :
            self.regularized_goal[i,0] = posestamped.pose.position.x
            self.regularized_goal[i,1] = posestamped.pose.position.y
            self.regularized_goal[i,2] = posestamped.pose.position.z
        # print('in wrist frame, physical hand self.regularized_goal: \n',self.regularized_goal[0:4,:])

        


    def tf_rst2world(self):
        '''
        first step: turn the results from wrist frame to ground frame, which is located at camera.
        second step: turn above results from the ground frame to the world frame, which is defined in the ur10e+shadow urdf.
        '''
        # print('shape of regularized_goal, and tf_grd2rst, and wrist_p', self.regularized_goal.shape, self.tf_grd2rst.shape, self.wrist_p.shape)
        self.regu_goal_grd = np.dot(self.regularized_goal, self.tf_grd2rst )   + self.wrist_p   #   right product, from wrist to ground

        regu_goal_ground = np.vstack((self.wrist_p,self.regu_goal_grd)  )                         #   21 point position, camera center ground frame, unit: m
        # print('physical hand, regu goal ground : \n', self.regu_goal_grd[0:5,:])
        # print('include wrist position: ', regu_goal_ground.shape , regu_goal_ground)

        #   from the camera center ground frame to ur+shadow world frame, unit: m
        self.regu_goal_world = self.tfRot.apply(regu_goal_ground) + np.array([0, 0, 0]) # np.array([0, -1.5, 1.5])
        # rospy.loginfo('----5 bio ik fun, after convert to world %s in meter',self.regu_goal_world)
        




    def call_bio_ik_sha(self):
        #   the index number depends on the self.regularized_goal array, which is determined in above function--(get_fk())
        #   left: if_3_b,   mf_3_b, rf_3_b, lf_3_b, th_3_b
        #   right:  ff_tip, mf_tip, rf_tip, lf_tip, th_tip

        #   in ground frame
        
        self.goal_position_sha[0] = self.regu_goal_grd[3,0]
        self.goal_position_sha[1] = self.regu_goal_grd[7,0]
        self.goal_position_sha[2] = self.regu_goal_grd[11,0]
        self.goal_position_sha[3] = self.regu_goal_grd[15,0]
        self.goal_position_sha[4] = self.regu_goal_grd[19,0]

        resp_sha = self.get_bio_ik_sha(self.req_ik_sha).ik_response
        # rospy.loginfo('---- after ik execute, one line command')
        self.joint_position_sha_tuple = resp_sha.solution.joint_state.position           #   tuple, should be sent to moveit for forward kinematics calculation
        self.joint_value_sha = np.asarray(self.joint_position_sha_tuple,dtype=np.float32)
        print('shadow hand joint: ',self.joint_value_sha)

    def config_goal_sha(self):
        for i in range(self.goal_len_sha):
            # self.req_ik_phy.position_goals[i].link_name = self.frame_name[i]
            self.req_ik_phy.position_goals[i].position.x = self.goal_location[i,0]
            self.req_ik_phy.position_goals[i].position.y = self.goal_location[i,1]
            self.req_ik_phy.position_goals[i].position.z = self.goal_location[i,2]

bio_ik_physical_hand = theBioikModule()




