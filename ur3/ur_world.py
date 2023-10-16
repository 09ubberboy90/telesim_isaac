from collections import defaultdict
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core.objects.cuboid import (DynamicCuboid, FixedCuboid,
                                            VisualCuboid)
from omni.isaac.core.prims.xform_prim import XFormPrim
from omni.isaac.core.utils.extensions import (disable_extension,
                                              enable_extension)
from omni.isaac.cortex.motion_commander import PosePq


disable_extension("omni.isaac.ros_bridge")
enable_extension("omni.isaac.ros2_bridge")

simulation_app.update()

import numpy as np
import pyquaternion as pyq
# Note that this is not the system level rclpy, but one compiled for omniverse
import rclpy
from geometry_msgs.msg import Pose
from std_msgs.msg import Bool
from sensor_msgs.msg import JointState
from omni.isaac.core.utils.stage import add_reference_to_stage
from general_world import TeleopWorld
from ur3.ur_robotiq import CortexUR
from omni.isaac.core_nodes.scripts.utils import set_target_prims

class UR_World(TeleopWorld):
    def __init__(self):
        self.urdf_path = "/home/ubb/Documents/Baxter_isaac/ROS2/src/ur_robotiq/ur_robotiq_isaac/urdfs/ur5e_robotiq.urdf"
        self.rmp_path = "/home/ubb/Documents/Baxter_isaac/ROS2/src/ur_robotiq/ur_robotiq_isaac/config"

        super().__init__(simulation_app)
        self.ros_sub = self.create_subscription(Pose, "right_hand/pose", self.move_right_cube_callback, 10)
        self.ros_sub2 = self.create_subscription(Pose, "left_hand/pose", self.move_left_cube_callback, 10)
        self.gripper_bool = False
        self.controller_sub = self.create_subscription(Bool, "controller_switch", self.controller_switch, 10)
        self.trigger_sub = self.create_subscription(Bool, "right_hand/trigger", self.right_trigger_callback, 10)
        self.trigger_sub2 = self.create_subscription(Bool, "left_hand/trigger", self.left_trigger_callback, 10)
        self.gripper_sub = self.create_subscription(JointState, "senseglove_motor", self.gripper_callback, 10)

        self.trigger = defaultdict(bool)

    def controller_switch(self, data: Bool):
        self.gripper_bool = data.data
        self.robot.gripper.direct_control = data.data
        self.robot_2.gripper.direct_control = data.data

    def set_limits(self, value):
        return min(max(value, 0.0), 115) * np.pi/180.0

    def gripper_callback(self, data: JointState):

        self.robot.gripper.set_gripper([self.set_limits(data.position[0]),0.0, self.set_limits(data.position[1]),0.0])
        self.robot_2.gripper.set_gripper([self.set_limits(data.position[0]),0.0, self.set_limits(data.position[1]),0.0])

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
            (-data.position.z, -data.position.x, data.position.y),
            (q1.w, q1.x, q1.y, q1.z),
        )

    def move_left_cube_callback(self, data: Pose):
        if data.position.x > 3000:
            self.tracking_enabled["left"] = False
        else:
            self.tracking_enabled["left"] = True

        q1 = pyq.Quaternion(x=data.orientation.x, y=data.orientation.y, z=data.orientation.z, w=data.orientation.w)
        mul_rot = pyq.Quaternion(w=0.0, x=0.0, y=0.707, z=0.707)  ## Handles axis correction ## Maybe needs to be removed and other adjusted
        x_rot = pyq.Quaternion(w=0.707, x=-0.707, y=-0.0, z=0.0)  ## Handles X rotation
        y_rot = pyq.Quaternion(w=0.707, x=0.0, y=-0.707, z=0.0)  ## Handles Y rotation

        offset_rot = pyq.Quaternion(w=0.0, x=0.0, y=0.0, z=1.0)  ## Handles Changing axis

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

    def robot_run_simulation(self):
        self.robot.motion_policy.update_world()
        self.robot_2.motion_policy.update_world()
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
                    self.left_cube.set_world_pose(*self.left_cube_pose)
                self.robot.arm.send_end_effector(target_pose=PosePq(*self.left_cube.get_world_pose()))


            if (not self.gripper_bool) and self.tracking_enabled["right"]:
                if self.trigger["right"] and self.robot_2.gripper.is_open():
                    ## Passing None need to make sure the self.command = None is commented out in the cortex file
                    self.robot_2.gripper.close(speed= None)
                    # print("Closing gripper")
                elif not (self.robot_2.gripper.is_open() and self.trigger["right"]):
                    ## Passing None need to make sure the self.command = None is commented out in the cortex file
                    self.robot_2.gripper.open(speed= None)
                    # print("Opening gripper")

            if self.tracking_enabled["right"]:
                if self.right_cube_pose is not None:
                    self.right_cube.set_world_pose(*self.right_cube_pose)
                self.robot_2.arm.send_end_effector(target_pose=PosePq(*self.right_cube.get_world_pose()))
            


    def setup_robot_scene(self):
        # self.robot_pos = (np.array([0.55, 0, 0]),np.array([0.707, 0.0, 0.0, 0.707]))
        self.robot_pos = (np.array([0.55, 0, 0]),np.array([1.0, 0.0, 0.0, 0.0]))
        self.robot = self.ros_world.add_robot(CortexUR(name="ur", urdf_path=self.urdf_path, rmp_path=self.rmp_path,position=self.robot_pos[0], orientation=self.robot_pos[1]))
        self.robot.set_world_pose(*self.robot_pos)
        self.robot.set_default_state(*self.robot_pos)
        
        self.robot_2_pos = (np.array([-0.55, 0, 0]),np.array([0.707, 0.0, 0.0, -0.707]))
        self.robot_2 = self.ros_world.add_robot(CortexUR(name="ur_2", urdf_path=self.urdf_path, rmp_path=self.rmp_path, position=self.robot_2_pos[0], orientation=self.robot_2_pos[1]))
        self.robot_2.set_world_pose(*self.robot_2_pos)
        self.robot_2.set_default_state(*self.robot_2_pos)
        self._robot_2 = self.robot_2
        add_reference_to_stage(
            usd_path=self.assets_root_path + "/Isaac/Environments/Office/Props/SM_TableC.usd",
            prim_path=f"/World/Tables/table",
        )
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


if __name__ == "__main__":
    rclpy.init()
    subscriber = UR_World()
    subscriber.run_simulation()
