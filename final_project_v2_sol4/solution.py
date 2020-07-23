# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC San Diego.
# Created by Yuzhe Qin, Fanbo Xiang

from final_env import FinalEnv, SolutionBase
import numpy as np
from sapien.core import Pose
from transforms3d.euler import euler2quat, quat2euler
from transforms3d.quaternions import quat2axangle, qmult, qinverse
from collections import defaultdict

class Solution(SolutionBase):
    """
    Implement the init function and act functions to control the robot
    Your task is to transport all cubes into the blue bin
    You may only use the following functions

    FinalEnv class:
    - get_agents
    - get_metadata

    Robot class:
    - get_observation
    - configure_controllers
    - set_action
    - get_metadata
    - get_compute_functions

    Camera class:
    - get_metadata
    - get_observation

    All other functions will be marked private at test time, calling those
    functions will result in a runtime error.

    How your solution is tested:
    1. new testing environment is initialized
    2. the init function gets called
    3. the timestep  is set to 0
    4. every 5 time steps, the act function is called
    5. when act function returns False or 200 seconds have passed, go to 1
    """

    def __init__(self):
        self.phase = 0
        self.counter = 0
        self.box_ids = []
        self.selected_x = None
        self.selected_z = None
        self.pose_left = None
        self.pose_right = None
        self.box_num = 0
        self.bin_id = 0
        self.top_view = None
        self.front_view = None
        self.spade_id = 0
        self.spade_width = 0
        self.spade_length = 0



    def init(self, env: FinalEnv):
        """called before the first step, this function should also reset the state of
        your solution class to prepare for the next run

        """
        meta = env.get_metadata()
        self.box_ids = meta['box_ids']
        self.bin_id = meta['bin_id']
        
        robot_left, robot_right, c1, c2, c3, c4 = env.get_agents()
        self.spade_id = robot_right.get_metadata()['link_ids'][-1]

        self.ps = [1000, 800, 600, 600, 200, 200, 100]
        self.ds = [1000, 800, 600, 600, 200, 200, 100]
        robot_left.configure_controllers(self.ps, self.ds)
        robot_right.configure_controllers(self.ps, self.ds)

        self.box_num = self.check_box(c2)
        self.top_view, self.front_view = self.get_bin(c1, c2, c3, c4)


    def act(self, env: FinalEnv, current_timestep: int):
        """called at each (actionable) time step to set robot actions. return False to
        indicate that the agent decides to move on to the next environment.
        Returning False early could mean a lower success rate (correctly placed
        boxes / total boxes), but it can also save you some time, so given a
        fixed total time budget, you may be able to place more boxes.

        """
        # robot_left, robot_right, camera_front, camera_left, camera_right, camera_top = env.get_agents()
        
        robot_left, robot_right, c1, c2, c3, c4 = env.get_agents()

        pf_left = f = robot_left.get_compute_functions()['passive_force'](True, True, False)
        pf_right = f = robot_right.get_compute_functions()['passive_force'](True, True, False)

        if self.phase == 0:
            self.counter += 1
            t1 = [2, 1, 0, -1.5, -1, 1, -2.7]
            t2 = [-2, 1, 0, -1.5, 1, 1, -2]
            robot_left.set_action(t1, [0] * 7, pf_left)
            robot_right.set_action(t2, [0] * 7, pf_right)

            if np.allclose(robot_left.get_observation()[0], t1, 0.05, 0.05) and np.allclose(
                    robot_right.get_observation()[0], t2, 0.05, 0.05):
                self.phase = 1
                self.counter = 0
                self.selected_x = None
                self.spade_width, self.spade_length = self.get_spade(c4)
                print(self.spade_width, self.spade_length)

            if (self.counter > 8000/5):
                self.counter = 0
                return False

        if self.phase == 1:
            # print(11111)
            self.counter += 1

            if (self.counter == 1):
                self.selected_x, self.selected_z = self.pick_box(c4)
                print(self.selected_x,self.selected_z)
                if self.selected_x is None:
                    self.counter = 0
                    self.phase = 0
                    return False
            # change 0.67 to 0.69
            if self.counter < 2000 / 5:
                target_pose_left = Pose([-0.25, 0.35, 0.55], euler2quat(np.pi, -np.pi / 3, -np.pi/2 ))
                self.diff_drive(robot_left, 9, target_pose_left)

                target_pose_right = Pose([-0.25, -0.35, 0.55], euler2quat(np.pi, -np.pi / 3, np.pi/2))
                self.diff_drive(robot_right, 9, target_pose_right)
 
            elif self.counter == 2000 / 5:
                # self.phase = 2
                # self.counter = 0
                pose = robot_left.get_observation()[2][9]
                p, q = pose.p, pose.q
                p[0] = 1
                self.pose_left = Pose(p, q)

                pose = robot_right.get_observation()[2][9]
                p, q = pose.p, pose.q
                p[0] = 1
                self.pose_right = Pose(p, q)
            
            elif self.counter < 2800 / 5:
                self.diff_drive(robot_left, 9, self.pose_left)
                self.diff_drive(robot_right, 9, self.pose_right)

            elif self.counter < 4800 / 5:
                target_pose_left = Pose([-0.25, 0.1, 0.55], euler2quat(np.pi, -np.pi / 3, -np.pi/2 ))
                self.diff_drive(robot_left, 9, target_pose_left)

                target_pose_right = Pose([-0.25, 0.1, 0.55], euler2quat(np.pi, -np.pi / 3, np.pi/2))
                self.diff_drive(robot_right, 9, target_pose_right)

            elif self.counter == 4800 / 5:
                pose = robot_left.get_observation()[2][9]
                p, q = pose.p, pose.q
                p[0] = 1
                self.pose_left = Pose(p, q)

                pose = robot_right.get_observation()[2][9]
                p, q = pose.p, pose.q
                p[0] = 1
                self.pose_right = Pose(p, q)
            
            elif self.counter < 5400 / 5:
                self.diff_drive(robot_left, 9, self.pose_left)
                self.diff_drive(robot_right, 9, self.pose_right)
            
            else:
                self.phase = 3
                self.counter = 0

        
        # if self.phase == 2:
        #     # print(22222)
        #     self.counter += 1
        #     self.diff_drive(robot_left, 9, self.pose_left)
        #     self.diff_drive(robot_right, 9, self.pose_right)
        #     if self.counter == 800 / 5:
        #         self.phase = 3
        #         self.counter = 0
               
       
        if self.phase == 3:
            self.counter += 1
            t1 = [2, 1, 0, -1.5, -1, 1, -2]
            t2 = [-2, 1, 0, -1.5, 1, 1, -2]

            robot_left.set_action(t1, [0] * 7, pf_left)
            robot_right.set_action(t2, [0] * 7, pf_right)

            if np.allclose(robot_left.get_observation()[0], t1, 0.05, 0.05) and np.allclose(
                    robot_right.get_observation()[0], t2, 0.05, 0.05):
                self.phase = 4
                self.counter = 0
                self.selected_x = None
                self.spade_width, self.spade_length = self.get_spade(c4)
                print(self.spade_width, self.spade_length)

            if (self.counter > 8000/5):
                self.counter = 0
                return False

        if self.phase == 4:
            # print(11111)
            self.counter += 1

            if (self.counter == 1):
                self.selected_x, self.selected_z = self.pick_box(c4)
                print(self.selected_x,self.selected_z)
                if self.selected_x is None:
                    self.counter = 0
                    self.phase = 0
                    return False
            # change 0.67 to 0.69
            
            target_pose_left = Pose([self.selected_x, 0.6, self.selected_z + 0.65], euler2quat(np.pi, -np.pi / 3, -np.pi / 2))
            self.diff_drive(robot_left, 9, target_pose_left)

            target_pose_right = Pose([self.selected_x, -0.6, self.selected_z + 0.55], euler2quat(np.pi, -np.pi / 3, np.pi / 2))
            self.diff_drive(robot_right, 9, target_pose_right)
 
            if self.counter == 1500 / 5:
                self.phase = 5
                self.counter = 0
                pose = robot_left.get_observation()[2][9]
                p, q = pose.p, pose.q
                p[1] = self.spade_length / 2
                self.pose_left = Pose(p, q)

                pose = robot_right.get_observation()[2][9]
                p, q = pose.p, pose.q
                p[1] = - self.spade_length / 2
                self.pose_right = Pose(p, q)

        if self.phase == 5:
            # print(22222)
            self.counter += 1
            self.diff_drive(robot_left, 9, self.pose_left)
            self.diff_drive(robot_right, 9, self.pose_right)
            if self.counter == 2500 / 5:
                self.phase = 6
    
                pose = robot_right.get_observation()[2][9]
                p, q = pose.p, pose.q
                self.pose_right = Pose(p, q)
                self.counter = 0
               
    
        if self.phase == 6:
            # print(33333)
            pose = robot_left.get_observation()[2][9]
            p, q = pose.p, pose.q
            p[1] += 0.1
            p[2] += 0.1
            q = euler2quat(np.pi, -np.pi , -np.pi /2)
            self.pose_left = Pose(p, q)
            self.counter += 1
            self.diff_drive(robot_left, 9, self.pose_left)

            if self.counter == 200 / 5:
                self.phase = 7
                self.counter = 0
          
        if self.phase == 7:
            self.counter += 1
        
            if (self.counter < 100 / 5):
                pose = robot_right.get_observation()[2][9]
                p, q = pose.p, pose.q
                # p[1] += 0.1
                # p[2] += 0.1
                q = euler2quat(0, 0, np.pi / 2)
                # q = euler2quat(np.pi, -np.pi , np.pi /2)
                self.diff_drive(robot_right, 9, Pose(p, q))
     
            elif (self.counter < 6800 / 5):
                p = [self.top_view[0], self.top_view[1] + self.spade_length /3, 2.0*self.front_view[2]]
                q = euler2quat(0, -np.pi / 1.5, 0)
                self.diff_drive(robot_right, 9, Pose(p, q))
            
            elif (self.counter < 7800 / 5):
                pose = robot_right.get_observation()[2][9]
                p = pose.p
                q = euler2quat(0, -np.pi / 1.5, 0)
                self.diff_drive(robot_right, 9, Pose(p, q))

            elif (self.counter >= 7800/5):
                pose = robot_right.get_observation()[2][9]
                p,q = pose.p, pose.q
                p[1] -= 1
                q = euler2quat(0, -np.pi / 1.5, 0)
                self.diff_drive(robot_right, 9, Pose(p, q))
            
            if (self.counter < 1500 / 5):
                pose = robot_left.get_observation()[2][9]
                p, q = pose.p, pose.q
                p[1] += 1
                self.diff_drive(robot_left, 9, Pose(p, q))
            elif (self.counter < 3000 / 5):
                pose = robot_left.get_observation()[2][9]
                p, q = pose.p, pose.q
                p[1] -= 1
                self.diff_drive(robot_left, 9, Pose(p, q))
            elif (self.counter < 9500 / 5):
                p = [self.top_view[0], self.top_view[1]+ self.spade_length /3, 2*self.front_view[2]]
                q = euler2quat(0, -np.pi / 1.5, 0)
                self.diff_drive(robot_left, 9, Pose(p, q))

            elif (self.counter < 11000 / 5):   
                pose = robot_left.get_observation()[2][9]
                p = pose.p
                q = euler2quat(0, -np.pi / 1.5, 0)
                self.diff_drive(robot_left, 9, Pose(p, q))

            if (self.counter >= 11000/5):
                self.phase = 0
                self.counter = 0
                return False

    def diff_drive(self, robot, index, target_pose, target_joint = None):
        """
        this diff drive is very hacky
        it tries to transport the target pose to match an end pose
        by computing the pose difference between current pose and target pose
        then it estimates a cartesian velocity for the end effector to follow.
        It uses differential IK to compute the required joint velocity, and set
        the joint velocity as current step target velocity.
        This technique makes the trajectory very unstable but it still works some times.
        """
        pf = robot.get_compute_functions()['passive_force'](True, True, False)
        max_v = 0.1
        max_w = np.pi
        qpos, qvel, poses = robot.get_observation()
        current_pose: Pose = poses[index]
        delta_p = target_pose.p - current_pose.p
        delta_q = qmult(target_pose.q, qinverse(current_pose.q))

        axis, theta = quat2axangle(delta_q)
        if (theta > np.pi):
            theta -= np.pi * 2

        t1 = np.linalg.norm(delta_p) / max_v
        t2 = theta / max_w
        t = max(np.abs(t1), np.abs(t2), 0.001)
        thres = 0.1
        if t < thres:
            k = (np.exp(thres) - 1) / thres
            t = np.log(k * t + 1)
        v = delta_p / t
        w = theta / t * axis
        target_qvel = robot.get_compute_functions()['cartesian_diff_ik'](np.concatenate((v, w)), 9)
        robot.set_action(qpos, target_qvel, pf)


    def get_global_position_from_camera(self, camera, depth, x, y):
        """
        This function is provided only to show how to convert camera observation to world space coordinates.
        It can be removed if not needed.

        camera: an camera agent
        depth: the depth obsrevation
        x, y: the horizontal, vertical index for a pixel, you would access the images by image[y, x]
        """
        cm = camera.get_metadata()
        proj, model = cm['projection_matrix'], cm['model_matrix']
        w, h = cm['width'], cm['height']

        # get 0 to 1 coordinate for (x, y) coordinates
        xf, yf = (x + 0.5) / w, 1 - (y + 0.5) / h

        # get 0 to 1 depth value at (x,y)
        zf = depth[int(y), int(x)]

        # get the -1 to 1 (x,y,z) coordinate
        ndc = np.array([xf, yf, zf, 1]) * 2 - 1

        # transform from image space to view space
        v = np.linalg.inv(proj) @ ndc
        v /= v[3]

        # transform from view space to world space
        v = model @ v

        return v
    
    def pick_box(self, c):
        color, depth, segmentation = c.get_observation()
        
        # np.random.shuffle(self.box_ids)
        position = defaultdict(list)
        size = defaultdict(float)
        gmax_x = -1
        gmin_x = 10000
        gmax_y = -1
        gmin_y = 10000
        for i in self.box_ids:
            m = np.where(segmentation == i)
            if len(m[0]):
                min_x = 10000
                max_x = -1
                min_y = 10000
                max_y = -1
                for y, x in zip(m[0], m[1]):
                    min_x = min(min_x, x)
                    max_x = max(max_x, x)
                    min_y = min(min_y, y)
                    max_y = max(max_y, y)
                x, y = round((min_x + max_x) / 2), round((min_y + max_y) / 2)
                position[i] = self.get_global_position_from_camera(c, depth, x, y)
                diff = self.get_global_position_from_camera(c, depth, max_x, max_y) - self.get_global_position_from_camera(c, depth, min_x, min_y)
                size[i] = abs(diff[0]) / 2
                gmax_x = max(gmax_x , position[i][0] + size[i])
                gmin_x = min(gmin_x , position[i][0] - size[i])
                gmax_y = max(gmax_y , position[i][1] + size[i])
                gmin_y = min(gmin_y , position[i][1] - size[i])
    
        # print(size)
        if len(position) == 0:
            return False
        num = int((gmax_x - gmin_x) / self.spade_width * 2) 
        select_x = 0
        max_c = 0
        # print(gmax_x,gmin_x)
        low = gmin_x
        select_size = 0
        while (low < gmax_x):
            high = low + self.spade_width
            # print(high, low)
            count = 0
            min_size = 100
            for p in position:
                if low + size[p] < position[p][0] and position[p][0] < high - size[p]:
                    count += 1
                    min_size = min(min_size, size[p])
            # print(count)
            if count > max_c:
                select_x = (high + low) / 2
                max_c = count
                select_size = min_size
            low += 0.01
        print(max_c)
        
        return select_x, 2*select_size


        
    
    def check_box(self, c):
        color, depth, segmentation = c.get_observation()
         
        count = 0
        box_ids = self.box_ids
        for i in box_ids:
            m = np.where(segmentation == i)
            if len(m[0]):
                count+=1
        return count

    def get_bin(self, c1, c2, c3, c4):
        color1, depth1, segmentation1 = c1.get_observation()
        color4, depth4, segmentation4 = c4.get_observation()

        m = np.where(segmentation4 == self.bin_id)
        min_x = 10000
        max_x = -1
        min_y = 10000
        max_y = -1
        for y, x in zip(m[0], m[1]):
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)
        x, y = round((min_x + max_x) / 2), round((min_y + max_y) / 2)
        top_view = self.get_global_position_from_camera(c4, depth4, x, y)
        # print("top",top_view)

        m = np.where(segmentation1 == self.bin_id)
        min_x = 10000
        max_x = -1
        min_z = 10000
        max_z = -1
        for z, x in zip(m[0], m[1]):
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_z = min(min_z, z)
            max_z = max(max_z, z)
        x, z = round((min_x + max_x) / 2), round((min_z + max_z) / 2)
        front_view =  self.get_global_position_from_camera(c1, depth1, x, z)
        # print("front",front_view)
        return top_view, front_view
        
    def get_spade(self, c):
        color, depth, segmentation = c.get_observation()
        m = np.where(segmentation == self.spade_id)
        
        if len(m[0]):
            min_x = 10000
            max_x = -1
            min_y = 10000
            max_y = -1
            for y, x in zip(m[0], m[1]):
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)
                        
            view1 = self.get_global_position_from_camera(c, depth, max_x, max_y)
            view2 = self.get_global_position_from_camera(c, depth, min_x, min_y)
            diff = abs(view1 - view2)
            return diff[0], diff[1] 
