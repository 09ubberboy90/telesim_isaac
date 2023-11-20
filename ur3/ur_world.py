from collections import defaultdict
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core.objects.cuboid import (DynamicCuboid, FixedCuboid,
                                            VisualCuboid)
from omni.isaac.core.prims.xform_prim import XFormPrim
from omni.isaac.core.utils.extensions import (disable_extension,
                                              enable_extension)
from omni.isaac.cortex.motion_commander import PosePq
import pxr

disable_extension("omni.isaac.ros_bridge")
enable_extension("omni.isaac.ros2_bridge")
# enable_extension("semu.xr.openxr")

simulation_app.update()

from omni.isaac.debug_draw import _debug_draw
import omni

import numpy as np
import numpy.matlib as npm
import pyquaternion as pyq
from scipy.spatial.transform import Rotation
# Note that this is not the system level rclpy, but one compiled for omniverse
import rclpy
from geometry_msgs.msg import Pose, PoseArray
from std_msgs.msg import Bool
from sensor_msgs.msg import JointState, PointCloud2, PointField
from omni.isaac.core.utils.stage import add_reference_to_stage
from general_world import TeleopWorld
from ur3.ur_robotiq import CortexUR
from omni.isaac.core_nodes.scripts.utils import set_target_prims

def quat_multiply(quat1,quat2):
    w0, x0, y0, z0 = quat1
    w1, x1, y1, z1 = quat2
    # Computer the product of the two quaternions, term by term
    w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
    x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
    y = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
    z = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1
    
    return np.array([w, x, y, z])

def quat_inverse(quat):
    return [quat[0], -quat[1],-quat[2],-quat[3]]

def quat_mean(Q):
    ## Taken from https://github.com/christophhagen/averaging-quaternions/blob/master/averageQuaternions.py
    # Number of quaternions to average
    M = Q.shape[0]
    A = npm.zeros(shape=(4,4))

    for i in range(0,M):
        q = Q[i,:]
        # multiply q with its transposed version q' and add A
        A = np.outer(q,q) + A

    # scale
    A = (1.0/M)*A
    # compute eigenvalues and -vectors
    eigenValues, eigenVectors = np.linalg.eig(A)
    # Sort by largest eigenvalue
    eigenVectors = eigenVectors[:,eigenValues.argsort()[::-1]]
    # return the real part of the largest eigenvector (has only real part)
    return np.real(eigenVectors[:,0].A1)

class UR_World(TeleopWorld):
    def __init__(self):
        self.urdf_path = "/home/ubb/Documents/Baxter_isaac/ROS2/src/ur_robotiq/ur_robotiq_isaac/urdfs/ur5e_robotiq.urdf"
        self.rmp_path = "/home/ubb/Documents/Baxter_isaac/ROS2/src/ur_robotiq/ur_robotiq_isaac/config"
        self.obj_pose = None

        super().__init__(simulation_app)
        self.ros_sub = self.create_subscription(Pose, "right_hand/pose", self.move_right_cube_callback, 10)
        self.ros_sub2 = self.create_subscription(Pose, "left_hand/pose", self.move_left_cube_callback, 10)
        self.ros_sub = self.create_subscription(Pose, "cpp/right_hand/pose", self.move_right_cube_unreal_callback, 10)
        self.ros_sub2 = self.create_subscription(Pose, "cpp/left_hand/pose", self.move_left_cube_unreal_callback, 10)
        self.ros_sub3 = self.create_subscription(Pose, "offset", self.offset_set, 10)
        self.gripper_bool = False
        self.controller_sub = self.create_subscription(Bool, "controller_switch", self.controller_switch, 10)
        self.obj_sub = self.create_subscription(PoseArray, "/object_pose", self.move_obj_callback, 10)
        self.trigger_sub = self.create_subscription(Bool, "right_hand/trigger", self.right_trigger_callback, 10)
        self.trigger_sub2 = self.create_subscription(Bool, "left_hand/trigger", self.left_trigger_callback, 10)
        self.gripper_sub = self.create_subscription(JointState, "senseglove_motor", self.gripper_callback, 10)
        self.timer_bool = True
        self.cloud = None
        self.overhead_rot = [0.5,-.5,-.5,-.5]
        self.offset = [-0.08,0.08,0.95]
        self.offset_rot = [0.707,0.0,0.707,0]
        self.obj_pos_ls = np.zeros((3,3))
        self.obj_rot_ls = np.zeros((3,4))
        self.obj_rot_ls[:,0] = 1 ## Make sure the quaternion is valid
        self.trigger = defaultdict(bool)

    def controller_switch(self, data: Bool):
        self.gripper_bool = data.data
        self.robot.gripper.direct_control = data.data
        self.right_robot.gripper.direct_control = data.data

    def timer_cb(self):
        self.timer_bool = True

    def set_limits(self, value):
        return min(max(value, 0.0), 115) * np.pi/180.0

    def gripper_callback(self, data: JointState):

        self.robot.gripper.set_gripper([self.set_limits(data.position[0]),0.0, self.set_limits(data.position[1]),0.0])
        self.right_robot.gripper.set_gripper([self.set_limits(data.position[0]),0.0, self.set_limits(data.position[1]),0.0])

    def move_obj_callback(self, data: PoseArray):
        data = data.poses[0]
        q1 = pyq.Quaternion(x=data.orientation.x, y=data.orientation.y, z=data.orientation.z, w=data.orientation.w)
        x_rot = pyq.Quaternion(w=0.707, x=-0.707, y=0.0, z=0.0)  ## Handles X rotation
        y_rot = pyq.Quaternion(w=0.707, x=0.0, y=0.707, z=0.0)  ## Handles X rotation
        z_rot = pyq.Quaternion(w=0.707, x=0.0, y=0.0, z=-0.707)  ## Handles X rotation
        # q1 = y_rot *q1
        # q1 = x_rot * q1 
        # q1 = z_rot * q1 
        self.obj_pos_ls = np.roll(self.obj_pos_ls,-1,axis=0)
        self.obj_pos_ls[-1] = (data.position.x, -data.position.z, -data.position.y)
        pos_mean = np.mean(self.obj_pos_ls, axis=0)

        self.obj_rot_ls = np.roll(self.obj_rot_ls,-1,axis=0)
        self.obj_rot_ls[-1] = (q1.w, q1.x, q1.y, q1.z)
        rot_mean = quat_mean(self.obj_rot_ls)
        self.obj_pose = (
            (pos_mean[0] + self.offset[0], pos_mean[1] + self.offset[1], pos_mean[2] + self.offset[2]),
            (q1.w, q1.x, q1.y, q1.z),
        )

    def move_right_cube_callback(self, data: Pose):
        if data.position.x > 3000:
            self.tracking_enabled["right"] = False
        else:
            self.tracking_enabled["right"] = True

        q1 = pyq.Quaternion(x=data.orientation.x, y=data.orientation.y, z=data.orientation.z, w=data.orientation.w)
        mul_rot = pyq.Quaternion(w=0.0, x=0.0, y=0.707, z=0.707)  ## Handles axis correction ## Maybe needs to be removed and other adjusted
        x_rot = pyq.Quaternion(w=0.707, x=-0.707, y=-0.0, z=0.0)  ## Handles X rotation
        y_rot = pyq.Quaternion(w=0.0, x=0.0, y=-1.0, z=0.0)  ## Handles Y rotation

        offset_rot = pyq.Quaternion(w=0.707, x=0.0, y=0.0, z=0.707)  ## Handles Changing axis

        q1 = mul_rot * q1
        q1 *= x_rot
        q1 *= y_rot
        
        q1 = offset_rot * q1

        self.right_cube_pose = (
            (data.position.z, data.position.x, data.position.y),
            (q1.w, q1.x, q1.y, q1.z),
        )

    def move_right_cube_unreal_callback(self, data: Pose):
        q1 = pyq.Quaternion(x=data.orientation.x, y=data.orientation.y, z=data.orientation.z, w=data.orientation.w)
        self.right_cube_pose = (
            (data.position.x, data.position.y, data.position.z),
            (q1.w, q1.x, q1.y, q1.z),
        )

    def move_left_cube_unreal_callback(self, data: Pose):
        q1 = pyq.Quaternion(x=data.orientation.x, y=data.orientation.y, z=data.orientation.z, w=data.orientation.w)
        self.right_cube_pose = (
            (data.position.x, data.position.y, data.position.z),
            (q1.w, q1.x, q1.y, q1.z),
        )

    def offset_set(self, data: Pose):
        self.offset = [data.position.x, data.position.z, data.position.y]
        self.offset_rot = [data.orientation.w, data.orientation.x, data.orientation.y, data.orientation.z]

    def move_left_cube_callback(self, data: Pose):
        if data.position.x > 3000:
            self.tracking_enabled["left"] = False
        else:
            self.tracking_enabled["left"] = True

        q1 = pyq.Quaternion(x=data.orientation.x, y=data.orientation.y, z=data.orientation.z, w=data.orientation.w)
        mul_rot = pyq.Quaternion(w=0.0, x=0.0, y=0.707, z=0.707)  ## Handles axis correction ## Maybe needs to be removed and other adjusted
        x_rot = pyq.Quaternion(w=0.707, x=-0.707, y=-0.0, z=0.0)  ## Handles X rotation
        y_rot = pyq.Quaternion(w=0.707, x=0.0, y=-0.707, z=0.0)  ## Handles Y rotation

        offset_rot = pyq.Quaternion(w=0.707, x=0.0, y=0.0, z=0.707)  ## Handles Changing axis

        q1 = mul_rot * q1
        q1 *= x_rot
        q1 *= y_rot
        
        q1 = offset_rot * q1

        self.left_cube_pose = (
            (data.position.x, -data.position.z, data.position.y),
            (q1.w, q1.x, q1.y, q1.z),
        )


    def right_trigger_callback(self, data: Bool):
        self.trigger["right"] = data.data
    
    def left_trigger_callback(self, data: Bool):
        self.trigger["left"] = data.data

    def transform_pose(self, pose, rot, mat):
        tmp = pyq.Quaternion(w=rot[0], x=rot[1], y=rot[2], z=rot[3])
        tmp = tmp.transformation_matrix
        tmp[0,3] = pose[0]
        tmp[1,3] = pose[1]
        tmp[2,3] = pose[2]
        new_t = np.matmul(mat, tmp)
        new_pose = new_t[:3, 3]
        new_rot = Rotation.as_quat(Rotation.from_matrix(new_t[:3, :3]))
        return new_pose, [new_rot[3], *new_rot[:3]]

    def robot_run_simulation(self):
        self.robot.motion_policy.update_world()
        self.right_robot.motion_policy.update_world()
        if self.obj_pose is not None:
            self.obj.set_world_pose(*self.obj_pose)

        if self.global_tracking:  ## Make sure this is not enable when working with corte

            if (not self.gripper_bool) and self.tracking_enabled["left"]:
                if self.trigger["left"] and self.robot.gripper.is_open():
                    ## Passing None need to make sure the self.command = None is commented out in the cortex file
                    self.robot.gripper.close(speed= None)
                    # print("Closing gripper")
                elif not (self.robot.gripper.is_open() and self.trigger["left"]):
                    ## Passing None need to make sure the self.command = None is commented out in the cortex file
                    self.robot.gripper.open(speed= None)
                    # print("Opening gripper")

            if self.tracking_enabled["left"]:
                if self.left_cube_pose is not None:
                    self.visual_left_cube.set_world_pose(*self.left_cube_pose)
                pose, rot = self.visual_left_cube.get_world_pose()
                new_pose = pose - self.robot_pos[0]
                self.left_cube.set_world_pose(*self.transform_pose(new_pose, quat_multiply(rot, self.overhead_rot), self.robot_pos_mat))
                self.robot.arm.send_end_effector(target_pose=PosePq(*self.left_cube.get_world_pose()))


            if (not self.gripper_bool) and self.tracking_enabled["right"]:
                if self.trigger["right"] and self.right_robot.gripper.is_open():
                    ## Passing None need to make sure the self.command = None is commented out in the cortex file
                    self.right_robot.gripper.close(speed= None)
                    # print("Closing gripper")
                elif not (self.right_robot.gripper.is_open() and self.trigger["right"]):
                    ## Passing None need to make sure the self.command = None is commented out in the cortex file
                    self.right_robot.gripper.open(speed= None)
                    # print("Opening gripper")

            if self.tracking_enabled["right"]:
                if self.right_cube_pose is not None:
                    self.visual_right_cube.set_world_pose(*self.right_cube_pose)
                pose, rot = self.visual_right_cube.get_world_pose()
                ## TODO: Create homogeonous trandofrmation to handle base rotation changing axis order
                new_pose = pose - self.right_robot_pos[0]
                self.right_cube.set_world_pose(*self.transform_pose(new_pose, quat_multiply(rot, self.overhead_rot), self.right_robot_pos_mat))
                self.right_robot.arm.send_end_effector(target_pose=PosePq(*self.right_cube.get_world_pose()))
            


    def setup_robot_scene(self):
        self.robot_pos = (np.array([0.55, 0, 0]),np.array([0.707, 0.0, 0.0, 0.707]))
        # self.robot_pos = (np.array([0.55, 0, 0]),np.array([1.0, 0.0, 0.0, 0.0]))
        self.robot = self.ros_world.add_robot(CortexUR(name="ur", urdf_path=self.urdf_path, rmp_path=self.rmp_path,position=self.robot_pos[0], orientation=self.robot_pos[1]))
        self.robot.set_world_pose(*self.robot_pos)
        self.robot.set_default_state(*self.robot_pos)
        
        self.right_robot_pos = (np.array([-0.55, 0, 0]),np.array([0.707, 0.0, 0.0, -0.707]))
        # self.right_robot_pos = (np.array([-0.55, 0, 0]),np.array([1.0, 0.0, 0.0, 0.0]))
        self.right_robot = self.ros_world.add_robot(CortexUR(name="ur_2", urdf_path=self.urdf_path, rmp_path=self.rmp_path, position=self.right_robot_pos[0], orientation=self.right_robot_pos[1]))
        self.right_robot.set_world_pose(*self.right_robot_pos)
        self.right_robot.set_default_state(*self.right_robot_pos)
        self._right_robot = self.right_robot
        
        
        tmp = pyq.Quaternion(w=0.707, x=0.0, y=0.0, z=-0.707) 
        self.robot_pos_mat = tmp.transformation_matrix
        # self.robot_pos_mat[0,3] = -0.55
        
        tmp = pyq.Quaternion(w=0.707, x=0.0, y=0.0, z=0.707)
        self.right_robot_pos_mat = tmp.transformation_matrix
        # self.right_robot_pos_mat[0,3] = 0.55

        
        
        add_reference_to_stage(
            usd_path=self.assets_root_path + "/Isaac/Environments/Office/Props/SM_TableC.usd",
            prim_path=f"/World/Tables/table",
        )
        add_reference_to_stage(
            # usd_path=self.assets_root_path + "/../../../../Projects/YCBV/Cracker/textured_obj.usd",
            # usd_path=self.assets_root_path + "/../../../../Projects/YCBV/Banana/textured_obj.usd",
            usd_path=self.assets_root_path + "/../../../../Projects/YCBV/Wood/textured_obj.usd",
            prim_path=f"/World/obj",
        )
        self.obj = XFormPrim(
            prim_path=f"/World/obj",
            name="cracker",
        )  # w,x,y,z

        self.table = XFormPrim(
            prim_path=f"/World/Tables/table",
            name="table",
            position=np.array([0.0, 0.25, -0.76]),
            orientation=np.array([0.707, 0, 0, 0.707]),
            scale=[1.4, 1.3, 1.7]
        )  # w,x,y,z

        self.cortex_table = FixedCuboid(
            "/World/Tables/cortex_table",
            position=np.array([0.0, 0.25, -0.08]),
            orientation=np.array([1, 0, 0, 0]),
            color=np.array([0, 0.2196, 0.3961]),
            size=0.8,
            scale=[1.8625, 1, 0.1]
            

        )

        # Create a cuboid to visualize where the ee frame is according to the kinematics"
        self.right_cube = VisualCuboid(
            "/World/Control/right_cube",
            position=np.array([0.07, 0.3, 1.02]),
            orientation=np.array([0.5,-.5,-.5,-.5]),
            size=0.005,
            color=np.array([0, 0, 1]),
        )
        self.left_cube = VisualCuboid(
            "/World/Control/left_cube",
            position=np.array([0.07, 0.3, 1.02]),
            orientation=np.array([0.5,-.5,-.5,-.5]),
            size=0.005,
            color=np.array([0, 0, 1]),
        )

        # Create a cuboid to visualize where the ee frame is according to the kinematics"
        self.visual_right_cube = VisualCuboid(
            "/World/Control/visual_right_cube",
            position=np.array([0.07, 0.3, 1.02]) + self.right_robot_pos[0],
            size=0.05,
            color=np.array([0, 0, 1]),
        )
        self.visual_left_cube = VisualCuboid(
            "/World/Control/visual_left_cube",
            position=np.array([0.07, 0.3, 1.02]) + self.robot_pos[0],
            size=0.05,
            color=np.array([0, 0, 1]),
        )




if __name__ == "__main__":
    rclpy.init()
    subscriber = UR_World()
    subscriber.run_simulation()
