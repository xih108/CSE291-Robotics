# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC San Diego.
# Created by Yuzhe Qin, Fanbo Xiang

from .stacking_env import StackingEnv
import sapien.core as sapien
import numpy as np
from typing import List, Tuple, Sequence


class HW2Env(StackingEnv):
    def __init__(self, timestep: float):
        """Class for homework2

        Args:
            timestep: timestep to balance the precision and the speed of physical simulation
        """
        StackingEnv.__init__(self, timestep)

        self.near, self.far = 0.1, 100
        self.camera_link = self.scene.create_actor_builder().build(is_kinematic=True)
        self.gl_pose = sapien.Pose([0, 0, 0], [0.5, -0.5, 0.5, -0.5])
        self.camera_link.set_pose(sapien.Pose([1.2, 0.0, 0.8], [0, -0.258819, 0, 0.9659258]))
        self.camera = self.scene.add_mounted_camera('fixed_camera', self.camera_link,
                                                    sapien.Pose(), 1920, 1080,
                                                    np.deg2rad(50), np.deg2rad(50), self.near, self.far)

        self.arm_joints = [joint for joint in self.robot.get_joints() if
                           joint.get_dof() > 0 and not joint.get_name().startswith("panda_finger")]
        self.set_joint_group_property(self.arm_joints, 1000, 400)
        assert len(self.arm_joints) == self.robot.dof - 2
        self.set_joint_group_property(self.gripper_joints, 200, 60)

        self.step()
        self.robot.set_drive_target(self.robot.get_qpos())

    def cam2base_gt(self) -> np.ndarray:
        """Get ground truth transformation of camera to base transformation

        Returns:
            Ground truth transformation from camera to robot base

        """
        camera_pose = self.camera_link.get_pose() * self.gl_pose
        base_pose = self.robot.get_root_pose()
        return self.pose2mat(base_pose.inv() * camera_pose)

    def get_current_ee_pose(self) -> np.ndarray:
        """Get current end effector pose for calibration calculation

        Returns:
            Transformation from end effector to robot base

        """
        return self.pose2mat(self.end_effector.get_pose())

    def get_object_point_cloud(self, seg_id: int) -> np.ndarray:
        """Fetch the object point cloud given segmentation id

        For example, you can use this function to directly get the point cloud of a colored box and use it for further
        calculation.

        Args:
            seg_id: segmentation id, you can get it by e.g. box.get_id()

        Returns:
            (3, n) dimension point cloud in the camera frame with(x, y, z) order

        """
        self.scene.update_render()
        self.camera.take_picture()
        camera_matrix = self.camera.get_camera_matrix()[:3, :3]
        gl_depth = self.camera.get_depth()
        y, x = np.where(gl_depth < 1)
        z = self.near * self.far / (self.far + gl_depth * (self.near - self.far))

        point_cloud = (np.dot(np.linalg.inv(camera_matrix),
                              np.stack([x, y, np.ones_like(x)] * z[y, x], 0)))

        seg_mask = self.camera.get_segmentation()[y, x]
        selected_index = np.nonzero(seg_mask == seg_id)[0]
        return point_cloud[:, selected_index]

    @staticmethod
    def compute_ik(ee2base: np.ndarray) -> List[List[float]]:
        """Compute the inverse kinematics of franka panda robot.

        This function is provided to help do the inverse kinematics calculation.
        The output of this function is deterministic.
        It will return a list of solutions for the given cartesian pose.
        In practice, some solutions are not physically-plausible due to self collision.
        So in this homework, you may need to choose the free_joint_value and which solution to use by yourself.

        References:
            ikfast_pybind:
            ikfast_pybind is a python binding generation library for the analytic kinematics engine IKfast.
            Please visit here for more information: https://pypi.org/project/ikfast-pybind/

            ikfast:
            ikfast is a powerful inverse kinematics solver provided within
            Rosen Diankov’s OpenRAVE motion planning software.
            Diankov, R. (2010). Automated construction of robotic manipulation programs.

        Args:
            ee2base: transformation from end effector to base

        Returns:
            A list of possible IK solutions when the last joint value is set as free_joint_value

        """
        try:
            import ikfast_franka_panda as panda
        except ImportError:
            print("Please install ikfast_pybind before using this function")
            print("Install: pip3 install ikfast-pybind")
            raise ImportError

        link72ee = np.array([[0.7071068, -0.7071068, 0, 0], [0.7071068, 0.7071068, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        link7_pose = ee2base @ link72ee
        pos = link7_pose[:3, 3]
        rot = link7_pose[:3, :3]
        return panda.get_ik(pos, rot, [0.785])

    def close_gripper(self):
        for joint in self.gripper_joints:
            joint.set_drive_target(0.001)

    def open_gripper(self):
        for joint in self.gripper_joints:
            joint.set_drive_target(0.04)

    def clear_velocity_command(self):
        for i, joint in enumerate(self.arm_joints):
            joint.set_drive_velocity_target(0)

    def wait_n_steps(self, n: int):
        self.clear_velocity_command()
        for i in range(n):
            passive_force = self.robot.compute_passive_force()
            self.robot.set_qf(passive_force)
            self.step()
            self.render()
        self.robot.set_qf([0] * self.robot.dof)

    def internal_controller(self, qvel: np.ndarray) -> None:
        """Control the robot dynamically to execute the given twist for one time step

        This method will try to execute the joint velocity using the internal dynamics function in SAPIEN.

        Note that this function is only used for one time step, so you may need to call it multiple times in your code
        Also this controller is not perfect, it will still have some small movement even after you have finishing using
        it. Thus try to wait for some steps using self.wait_n_steps(n) like in the hw2.py after you call it multiple
        time to allow it to reach the target position

        Args:
            qvel: (7,) vector to represent the joint velocity

        """
        assert qvel.size == len(self.arm_joints)
        target_qpos = qvel * self.scene.get_timestep() + self.robot.get_drive_target()[:-2]
        for i, joint in enumerate(self.arm_joints):
            joint.set_drive_velocity_target(qvel[i])
            joint.set_drive_target(target_qpos[i])
        passive_force = self.robot.compute_passive_force()
        self.robot.set_qf(passive_force)
        # print("torque",passive_force)   
    
    def evaluate_first_two_box(self) -> bool:
        """Evaluate whether you stack the first two boxes successfully"""
        position, size = self.target
        rbox, gbox, _ = self.boxes
        contacts = self.scene.get_contacts()

        red_target_position = np.array([position[0], position[1], size])
        green_target_position = np.array([position[0], position[1], 3 * size])
        red_in_place = np.linalg.norm(rbox.get_pose().p - red_target_position) < 0.01
        green_in_place = np.linalg.norm(gbox.get_pose().p - green_target_position) < 0.01
        return green_in_place and red_in_place

    def evaluate_final_result(self) -> bool:
        """Evaluate whether you stack the all three boxes successfully"""
        position, size = self.target
        rbox, gbox, bbox = self.boxes
        static = (rbox.velocity @ rbox.velocity) < 1e-5 and \
                 (gbox.velocity @ gbox.velocity) < 1e-5 and \
                 (bbox.velocity @ bbox.velocity) < 1e-5
        if not static:
            return False

        first_two_box_success = self.evaluate_first_two_box()
        blue_target_position = np.array([position[0], position[1], 5 * size])
        blue_in_place = np.linalg.norm(bbox.get_pose().p - blue_target_position) < 0.01
        return blue_in_place and first_two_box_success

    ####################################################################################################################
    # ============================== You will need to implement all the functions below ================================
    ####################################################################################################################
    @staticmethod
    def pose2mat(pose: sapien.Pose) -> np.ndarray:
        """You need to implement this function

        You will need to implement this function first before any other functions.
        In this function, you need to convert a (position: pose.p, quaternion: pose.q) into a SE(3) matrix

        You can not directly use external library to transform quaternion into rotation matrix.
        Only numpy can be used here.
        Args:
            pose: sapien Pose object, where Pose.p and Pose.q are position and quaternion respectively

        Hint: the convention of quaternion

        Returns:
            (4, 4) transformation matrix represent the same pose

        """
        
        P = np.array([pose.p])
        w = pose.q[0]
        q = np.array(pose.q[1:])
        skew = np.array([[0, -q[2], q[1]],
                         [q[2], 0, -q[0]],
                         [-q[1], q[0], 0]])

        E = np.concatenate((np.array([-q]).T, w * np.identity(3) + skew), axis = 1)
        G = np.concatenate((np.array([-q]).T, w * np.identity(3) - skew), axis = 1)
        R = E @ G.T
      
        T = np.identity(4)    
        T[:3,:3] = R
        T[:3,3] = P.reshape(1,3)
        T[-1,:] = np.array([0,0,0,1])
        
        return T

    def pose2exp_coordinate(self, pose: np.ndarray) -> Tuple[np.ndarray, float]:
        """You may need to implement this function

        Compute the exponential coordinate corresponding to the given SE(3) matrix
        Note: unit twist is not a unit vector

        Args:
            pose: (4, 4) transformation matrix

        Returns:
            Unit twist: (6, ) vector represent the unit twist
            Theta: scalar represent the quantity of exponential coordinate
        """

        R = pose[:3, :3]
        p = pose[:3, 3]
       
        if np.trace(R) >= 3:
            omega = np.array([0,0,0])
            v = p / np.linalg.norm(p)
            Theta = np.linalg.norm(p)

        else:
            if np.trace(R) <= -1:
                Theta = np.arccos( -1 )
                omega = np.array([R[0][2],R[1][2],1 + R[2][2]]) / np.sqrt(2*(1 + R[2][2]))
                omega_bar = np.array([[0, -omega[2], omega[1]],
                            [omega[2], 0, -omega[0]],
                            [-omega[1], omega[0], 0]])

            else:
                Theta = np.arccos( (np.trace(R) - 1)/2 )
                omega_bar = (R - R.T) / (2*np.sin(Theta))
                omega = np.array( [omega_bar[2][1] , omega_bar[0][2], omega_bar[1][0]])

            omega = omega / np.linalg.norm(omega)
            
            #find parallel, perpendicular component
            t_para = (p @ omega) * omega
            t_perpen = p - t_para
            #find screw point
            b = np.tan(Theta / 2) * omega
            screw_q = (np.cross(b, p) - np.cross(b, np.cross(b,p)))/ (2 * b @ b)
            v = -np.cross(omega, screw_q) + t_para/Theta
        
        twist = np.zeros(6)
        twist[:3] = omega
        twist[3:] = v
        # print(twist, Theta)
        return (twist, Theta)


    def compute_joint_velocity_from_twist(self, twist: np.ndarray) -> np.ndarray:
        """You need to implement this function

        This function is a kinematic-level calculation which do not consider dynamics.
        Pay attention to the frame of twist, is it spatial twist or body twist

        Jacobian is provided for your, so no need to compute the velocity kinematics
        ee_jacobian is the geometric Jacobian on account of only the joint of robot arm, not gripper
        Jacobian in SAPIEN is defined as the derivative of spatial twist with respect to joint velocity

        Args:
            twist: (6,) vector to represent the twist

        Returns:
            (7, ) vector for the velocity of arm joints (not include gripper)

        """
        assert twist.size == 6
        # Jacobian define in SAPIEN use twist (v, \omega) which is different from the definition in the slides
        # So we perform the matrix block operation below
        dense_jacobian = self.robot.compute_spatial_twist_jacobian()  # (num_link * 6, dof())
        ee_jacobian = np.zeros([6, self.robot.dof - 2])  # (6, 7)
        print(dense_jacobian.shape)
        ee_jacobian[:3, :] = dense_jacobian[self.end_effector_index * 6 - 3:self.end_effector_index * 6, :7]
        ee_jacobian[3:6, :] = dense_jacobian[(self.end_effector_index - 1) * 6:self.end_effector_index * 6 - 3, :7]
        theta_v = np.linalg.pinv(ee_jacobian) @ twist
        print(self.end_effector_index)
        print(ee_jacobian.shape)
        print("vel，",theta_v)
        return theta_v

    def move_to_target_pose_with_internal_controller(self, target_ee_pose: np.ndarray, num_steps: int) -> None:
        """You need to implement this function

        Move the robot hand dynamically to a given target pose
        You may need to call self.internal_controller and your self.compute_joint_velocity_from_twist in this function

        To make command (e.g. internal controller) take effect and simulate all the physical effects, you need to step
        the simulation world for one step and render the new scene for visualization by something like:
            for i in range(num_step):
                # Do something
                self.internal_controller(target_joint_velocity)
                self.step()
                self.render()

        Args:
            target_ee_pose: (4, 4) transformation of robot hand in robot base frame (ee2base)
            num_steps: how much steps to reach to target pose, each step correspond to self.scene.get_timestep() seconds
                in physical simulation

        """
       
        executed_time = num_steps * self.scene.get_timestep()
        
        for i in range(num_steps):
            T_sh1 = self.get_current_ee_pose()
            T_h1h2 = np.linalg.inv(T_sh1) @ target_ee_pose
            # print(T_h1h2)
            unit_twist, theta = self.pose2exp_coordinate(T_h1h2)
            body_twist = unit_twist * (theta / executed_time)

            R = T_sh1[:3,:3]
            p = T_sh1[:3,3]
            Ad = np.zeros((6,6))
            Ad[:3,:3] = R
            Ad[3:,:3] = np.array([[0, -p[2], p[1]],
                                [p[2], 0, -p[0]],
                                [-p[1], p[0], 0]]) @ R
            Ad[3:,3:] = R
            spatial_twist = Ad @ body_twist
        
            target_joint_velocity = self.compute_joint_velocity_from_twist(spatial_twist)

            self.internal_controller(target_joint_velocity)
            self.step()
            self.render()   
            executed_time  -= self.scene.get_timestep()

    def pick_up_object_with_internal_controller(self, seg_id: int, height: float) -> None:
        """You need to implement this function

        Pick up a specific box to a target height using the given internal controller


        You can use the following function to get the segmented point cloud:
            point_cloud = self.get_object_point_cloud(seg_id)

        Args:
            seg_id: segmentation id, you can get it by e.g. box.get_id()
            height: target height of the box

        """

        point_cloud = self.get_object_point_cloud(seg_id)
        vmax = np.amax(point_cloud, axis=1) 
        vmin = np.amin(point_cloud, axis=1)  
        size_x = vmax[0] - vmin[0]
        center = np.ones(4)
        center[:3] =  (vmax + vmin)/2
        cam2base = self.cam2base_gt()
        center_base = cam2base @ center
        # print("base", center_base)
        center_base[2] += 0.105

        ee2base = self.get_current_ee_pose()
        gripper_pose = np.identity(4)
        gripper_pose[2][2] = -1
        gripper_pose[1][1] = -1
        gripper_pose[:3, 3] = center_base[:3]
        gripper_pose[2,3] = ee2base[2,3]

        self.move_to_target_pose_with_internal_controller(gripper_pose, 100)
        self.wait_n_steps(50)
        
        gripper_pose[2,3] = center_base[2]
        self.move_to_target_pose_with_internal_controller(gripper_pose, 100)
        self.wait_n_steps(50)
        self.close_gripper()
        self.wait_n_steps(100)
       
        # move up
        gripper_pose[2,3] += height
        self.move_to_target_pose_with_internal_controller(gripper_pose, 100)
        self.wait_n_steps(50)

    def place_object_with_internal_controller(self, seg_id: int, target_object_position: np.ndarray) -> None:
        """You need to implement this function

        Place a specific box to a target position
        This function do not consider rotation, so you can just assume a fixed rotation or any orientation you want

        After place the box, you also need to move the gripper to some safe place. Thus when you pick up next box, it
        will not jeopardise your previous result.

        Args:
            seg_id: segmentation id, you can get it by e.g. box.get_id()
            target_object_position: target position of the box

        """

        target_object_position[2] += 0.105
        ee2base = self.get_current_ee_pose()
        gripper_pose = np.identity(4)
        gripper_pose[2][2] = -1
        gripper_pose[1][1] = -1
        gripper_pose[:2, 3] = target_object_position[:2]
        gripper_pose[2,3] = ee2base[2,3]

        # move forward
        self.move_to_target_pose_with_internal_controller(gripper_pose, 100)
        self.wait_n_steps(50)

        # move down
        gripper_pose[2,3] = target_object_position[2]
        self.move_to_target_pose_with_internal_controller(gripper_pose, 100)
        self.wait_n_steps(50)
        self.open_gripper()
        self.wait_n_steps(100)
        
        # move up
        gripper_pose[2,3] += 0.2
       
        self.move_to_target_pose_with_internal_controller(gripper_pose, 100)
        self.wait_n_steps(50)
   

    def user_controller(self, qvel: np.ndarray, target_qpos: np.ndarray ) -> None:
        assert qvel.size == len(self.arm_joints)
        pids = []
        pid_parameters = [(1000, 500, 10), (1000, 500, 10), (900, 400, 9), (300, 200, 6), 
                        (200, 100, 5), (200, 100, 5),(200, 100, 5)]
        # pid_parameters = [(1000, 500, 10), (1000, 500, 10), (900, 400, 9), (300, 200, 6), 
                        # (200, 150, 5), (200, 150, 5),(200, 150, 5)]
                        # 0.09, 0.10/ 0.105
        # pid_parameters = [(900, 500, 10), (900, 500, 10), (800, 400, 9), (300, 200, 6), 
        #                 (200, 150, 5), (200, 150, 5),(200, 150, 5)]
                        # 0.09, 0.103


        for i, joint in enumerate(self.arm_joints):
            pids.append(PID(pid_parameters[i][0], pid_parameters[i][1], pid_parameters[i][2]))
        

        qf = self.robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True, external=False)
        pid_qf = pid_forward(pids, target_qpos, self.robot.get_qpos()[:-2], self.scene.get_timestep())
        qf[:-2] += pid_qf
        self.robot.set_qf(qf)


    def move_to_target_pose_with_user_controller(self, target_ee_pose: np.ndarray, num_steps: int) -> None:
        """You need to implement this function

        Similar to self.move_to_target_pose_with_internal_controller. However, this time you need to implement your own
        controller instead of the SAPIEN internal controller.

        You can use anything you want to perform dynamically execution of the robot, e.g. PID, compute torque control
        You can write additional class or function to help implement this function.
        You can also use the inverse kinematics to calculate target joint position.
        You do not need to follow the given timestep exactly if you do not know how to do that.

        However, you are not allow to magically set robot's joint position and velocity using set_qpos() and set_qvel()
        in this function. You need to control the robot by applying appropriate force on the robot like real-world.

        There are two function you may need to use (optional):
            gravity_compensation = self.robot.compute_passive_force(gravity=False, coriolis_and_centrifugal=True,
                                                                    external=False)
            coriolis_and_centrifugal_compensation = self.robot.compute_passive_force(gravity=False,
                                                                                    coriolis_and_centrifugal=True,
                                                                                    external=False)

        The first function calculate how much torque each joint need to apply in order to balance the gravity
        Similarly, the second function calculate how much torque to balance the coriolis and centrifugal force

        To controller your robot actuator dynamically (actuator is mounted on each joint), you can use
        self.robot.set_qf(joint_torque)
        Note that joint_torque is a (9, ) vector which also includes the joint torque of two gripper

        Args:
            target_ee_pose: (4, 4) transformation of robot hand in robot base frame (ee2base)
            num_steps: how much steps to reach to target pose, each step correspond to self.scene.get_timestep() seconds

        """
        timestep = self.scene.get_timestep()
        executed_time = num_steps * timestep
        target_qpos = self.robot.get_qpos()[:-2]

        for i in range(num_steps):

            T_sh1 = self.get_current_ee_pose()
            T_h1h2 = np.linalg.inv(T_sh1) @ target_ee_pose
            # print(T_h1h2) 
            unit_twist, theta = self.pose2exp_coordinate(T_h1h2)
            body_twist = unit_twist * (theta / executed_time)

            R = T_sh1[:3,:3]
            p = T_sh1[:3,3]
            Ad = np.zeros((6,6))
            Ad[:3,:3] = R
            Ad[3:,:3] = np.array([[0, -p[2], p[1]],
                                [p[2], 0, -p[0]],
                                [-p[1], p[0], 0]]) @ R
            Ad[3:,3:] = R
            spatial_twist = Ad @ body_twist 
        
            target_joint_velocity = self.compute_joint_velocity_from_twist(spatial_twist)
            target_qpos += target_joint_velocity * timestep

            self.user_controller(target_joint_velocity, target_qpos)
            self.step()
            self.render()   
            executed_time  -= self.scene.get_timestep()


    def pick_up_object_with_user_controller(self, seg_id: int, height: float) -> None:
        """You need to implement this function

        Pick up a specific box to a target height using your own controller

        To achieve a pick up action, you can either call your self.move_to_target_pose_with_user_controller function to
        move a sequence of pose designed manually, or using a motion planning algorithm like RRT to get a trajectory and
        execute the trajectory.

        Args:
            seg_id: segmentation id, you can get it by e.g. box.get_id()
            height: target height of the box

        """

        point_cloud = self.get_object_point_cloud(seg_id)
        vmax = np.amax(point_cloud, axis=1) 
        vmin = np.amin(point_cloud, axis=1)  
        size_x = vmax[0] - vmin[0]
        center = np.ones(4)
        center[:3] =  (vmax + vmin)/2
        cam2base = self.cam2base_gt()
        center_base = cam2base @ center
        # print("base", center_base)
        center_base[2] += 0.105

        ee2base = self.get_current_ee_pose()
        gripper_pose = np.identity(4)
        gripper_pose[2][2] = -1
        gripper_pose[1][1] = -1
        gripper_pose[:3, 3] = center_base[:3]
        gripper_pose[2,3] = ee2base[2,3]

        self.move_to_target_pose_with_user_controller(gripper_pose, 500)
        self.wait_n_steps(50)
    
        # # # move down
        gripper_pose[2,3] = center_base[2]
        self.move_to_target_pose_with_user_controller(gripper_pose, 500)
        self.wait_n_steps(50)
        self.close_gripper()
        self.wait_n_steps(200)
       
        # # # move up
        gripper_pose[2,3] = self.get_current_ee_pose()[2,3] + height
        self.move_to_target_pose_with_user_controller(gripper_pose, 500)
        self.wait_n_steps(50)

    def place_object_with_user_controller(self, seg_id: int, target_object_position: np.ndarray) -> None:
        """You need to implement this function

        Similar to the last function, place the box to the given position with your own controller
        If you have already implemented the pick_up_object_with_user_controller, this function is not hard for you

        Args:
            seg_id: segmentation id, you can get it by e.g. box.get_id()
            target_object_position: target position of the box

        """

        target_object_position[2] += 0.105
        ee2base = self.get_current_ee_pose()
        gripper_pose = np.identity(4)
        gripper_pose[2][2] = -1
        gripper_pose[1][1] = -1
        gripper_pose[:2, 3] = target_object_position[:2]
        gripper_pose[2,3] = ee2base[2,3]

        # move forward
        self.move_to_target_pose_with_user_controller(gripper_pose, 500)
        self.wait_n_steps(50)

        # move down
        gripper_pose[2,3] = target_object_position[2]
        self.move_to_target_pose_with_user_controller(gripper_pose, 500)
        self.wait_n_steps(50)
        self.open_gripper()
        self.wait_n_steps(200)
        
        # move up
        gripper_pose[2,3] += 0.2
       
        self.move_to_target_pose_with_user_controller(gripper_pose, 500)
        self.wait_n_steps(50)

class PID:
    def __init__(self, kp = 0.0, ki = 0.0, kd = 0.0):
        self.p = kp
        self.i = ki
        self.d = kd

        self._cp = 0
        self._ci = 0
        self._cd = 0

        self._last_error = 0
    
   
    def compute(self, current_error, dt):
        d_error = current_error - self._last_error

        self._cp = current_error
        self._ci += current_error * dt
        if abs(self._last_error) > 0.01:
            self._cd = d_error / dt

        self._last_error = current_error
        signal = (self.p * self._cp) + (self.i * self._ci) + (self.d * self._cd)
        return signal


def pid_forward(pids: list, target_pos: np.ndarray, current_pos: np.ndarray, dt: float) -> np.ndarray:
    qf = np.zeros(len(pids))
    errors = target_pos - current_pos
    for i in range(len(pids)):
        qf[i] = pids[i].compute(errors[i], dt)
    return qf