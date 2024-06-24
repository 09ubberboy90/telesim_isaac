from collections import defaultdict
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core.objects.cuboid import DynamicCuboid, FixedCuboid, VisualCuboid
from omni.isaac.core.prims.xform_prim import XFormPrim
from omni.isaac.core.utils.extensions import disable_extension, enable_extension
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
from omni.isaac.core.utils.stage import add_reference_to_stage
from general_world import TeleopWorld
from ur3.ur_robotiq import CortexUR

class UR_World(TeleopWorld):
    def __init__(self):
        self.urdf_path = "/home/ubb/Documents/Baxter_isaac/ROS2/src/ur_robotiq/ur_robotiq_isaac/urdfs/ur5e_robotiq.urdf"
        self.rmp_path = "/home/ubb/Documents/Baxter_isaac/ROS2/src/ur_robotiq/ur_robotiq_isaac/config/ur5_rmp_config.json"

        super().__init__(simulation_app)
        self.ros_sub = self.create_subscription(
            Pose, "right_hand/pose", self.move_right_cube_callback, 10
        )
        self.ros_sub2 = self.create_subscription(
            Pose, "left_hand/pose", self.move_left_cube_callback, 10
        )
        self.ros_sub = self.create_subscription(
            Pose, "bp/right_hand/pose", self.move_right_cube_unreal_callback, 10
        )
        self.ros_sub2 = self.create_subscription(
            Pose, "bp/left_hand/pose", self.move_left_cube_unreal_callback, 10
        )
        self.gripper_bool = False
        self.controller_sub = self.create_subscription(
            Bool, "controller_switch", self.controller_switch, 10
        )
        self.trigger_sub = self.create_subscription(
            Bool, "right_hand/trigger", self.right_trigger_callback, 10
        )
        self.trigger_sub2 = self.create_subscription(
            Bool, "left_hand/trigger", self.left_trigger_callback, 10
        )
        self.trigger = defaultdict(bool)
        self.once = True

    def controller_switch(self, data: Bool):
        self.gripper_bool = data.data
        self.robot.gripper.direct_control = data.data
        self.right_robot.gripper.direct_control = data.data

    def move_right_cube_callback(self, data: Pose):
        if data.position.x > 3000:
            self.tracking_enabled["right"] = False
        else:
            self.tracking_enabled["right"] = True

        q1 = pyq.Quaternion(
            x=data.orientation.x,
            y=data.orientation.y,
            z=data.orientation.z,
            w=data.orientation.w,
        )
        mul_rot = pyq.Quaternion(
            w=0.0, x=0.0, y=0.707, z=0.707
        )  ## Handles axis correction ## Maybe needs to be removed and other adjusted
        z_rot = pyq.Quaternion(w=0.0, x=-0.0, y=-0.0, z=1.0)  ## Handles Z rotation

        offset_rot = pyq.Quaternion(
            w=0.707, x=0.0, y=0.0, z=-0.707
        )  ## Handles Changing axis
        x_rot = pyq.Quaternion(
            w=0.707, x=-0.707, y=0.0, z=-0.0
        )  ## Handles Changing axis

        q1 = mul_rot * q1
        q1 *= z_rot
        q1 *= x_rot

        q1 = offset_rot * q1
        self.right_cube_pose = (
            (data.position.z, data.position.x, data.position.y),
            (q1.w, q1.x, q1.y, q1.z),
        )

    def move_right_cube_unreal_callback(self, data: Pose):
        q1 = pyq.Quaternion(
            x=data.orientation.x,
            y=data.orientation.y,
            z=data.orientation.z,
            w=data.orientation.w,
        )
        offset_rot = pyq.Quaternion(
            w=0.707, x=0.0, y=0.0, z=-0.707
        )  ## Handles Changing axis
        x_offset = pyq.Quaternion(
            w=0, x=1.0, y=0.0, z=0
        )  ## 180 rotation on x
        q1 = offset_rot * q1
        q1 *= x_offset

        self.right_cube_pose = (
            (data.position.x, data.position.y, data.position.z),
            (q1.w, q1.x, q1.y, q1.z),
        )

    def move_left_cube_unreal_callback(self, data: Pose):
        q1 = pyq.Quaternion(
            x=data.orientation.x,
            y=data.orientation.y,
            z=data.orientation.z,
            w=data.orientation.w,
        )
        offset_rot = pyq.Quaternion(
            w=0.707, x=0.0, y=0.0, z=-0.707
        )  ## Handles Changing axis
        q1 = offset_rot * q1

        self.right_cube_pose = (
            (data.position.x, data.position.y, data.position.z),
            (q1.w, q1.x, q1.y, q1.z),
        )

    def move_left_cube_callback(self, data: Pose):
        if data.position.x > 3000:
            self.tracking_enabled["left"] = False
        else:
            self.tracking_enabled["left"] = True

        q1 = pyq.Quaternion(
            x=data.orientation.x,
            y=data.orientation.y,
            z=data.orientation.z,
            w=data.orientation.w,
        )
        mul_rot = pyq.Quaternion(
            w=0.0, x=0.0, y=0.707, z=0.707
        )  ## Handles axis correction ## Maybe needs to be removed and other adjusted
        x_rot = pyq.Quaternion(w=0.707, x=-0.707, y=-0.0, z=0.0)  ## Handles X rotation
        y_rot = pyq.Quaternion(w=0.707, x=0.0, y=-0.707, z=0.0)  ## Handles Y rotation

        offset_rot = pyq.Quaternion(
            w=0.707, x=0.0, y=0.0, z=0.707
        )  ## Handles Changing axis

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
        if self.once:
            self.robot.motion_policy.add_obstacle(self.cortex_table)
            self.right_robot.motion_policy.add_obstacle(self.cortex_table)
            self.robot.motion_policy.add_obstacle(self.cortex_table1)
            self.right_robot.motion_policy.add_obstacle(self.cortex_table1)
            self.robot.motion_policy.add_obstacle(self.cortex_table2)
            self.right_robot.motion_policy.add_obstacle(self.cortex_table2)
            self.robot.motion_policy.add_obstacle(self.cortex_table3)
            self.right_robot.motion_policy.add_obstacle(self.cortex_table3)
            self.robot.motion_policy.add_obstacle(self.cortex_table4)
            self.right_robot.motion_policy.add_obstacle(self.cortex_table4)
            self.robot.motion_policy.add_obstacle(self.cortex_small_table)
            self.right_robot.motion_policy.add_obstacle(self.cortex_small_table)
            print("Adding obstacles")
            self.once = False

        self.robot.motion_policy.update_world()
        self.right_robot.motion_policy.update_world()
        self.robot.motion_policy.set_robot_base_pose(*self.robot_pos)
        self.right_robot.motion_policy.set_robot_base_pose(*self.right_robot_pos)

        if self.global_tracking:  ## Make sure this is not enable when working with corte
            if (not self.gripper_bool) and self.tracking_enabled["left"]:
                if self.trigger["left"] and self.robot.gripper.is_open():
                    ## Passing None need to make sure the self.command = None is commented out in the cortex file
                    self.robot.gripper.close(speed=None)
                    # print("Closing gripper")
                elif not (self.robot.gripper.is_open() and self.trigger["left"]):
                    ## Passing None need to make sure the self.command = None is commented out in the cortex file
                    self.robot.gripper.open(speed=None)
                    # print("Opening gripper")

            if self.tracking_enabled["left"]:
                if self.left_cube_pose is not None:
                    self.left_cube.set_world_pose(*self.left_cube_pose)
                self.robot.arm.send_end_effector(
                    target_pose=PosePq(*self.left_cube.get_world_pose())
                )

            if (not self.gripper_bool) and self.tracking_enabled["right"]:
                if self.trigger["right"] and self.right_robot.gripper.is_open():
                    ## Passing None need to make sure the self.command = None is commented out in the cortex file
                    self.right_robot.gripper.close(speed=None)
                    # print("Closing gripper")
                elif not (self.right_robot.gripper.is_open() and self.trigger["right"]):
                    ## Passing None need to make sure the self.command = None is commented out in the cortex file
                    self.right_robot.gripper.open(speed=None)
                    # print("Opening gripper")

            if self.tracking_enabled["right"]:
                if self.right_cube_pose is not None:
                    self.right_cube.set_world_pose(*self.right_cube_pose)
                self.right_robot.arm.send_end_effector(
                    target_pose=PosePq(*self.right_cube.get_world_pose())
                )

    def setup_robot_scene(self):
        self.robot_pos = (np.array([0.55, 0, 0]), np.array([0.707, 0.0, 0.0, 0.707]))

        self.robot = self.ros_world.add_robot(
            CortexUR(
                name="ur",
                urdf_path=self.urdf_path,
                rmp_path=self.rmp_path,
                position=self.robot_pos[0],
                orientation=self.robot_pos[1],
            )
        )
        self.robot.set_world_pose(*self.robot_pos)
        self.robot.set_default_state(*self.robot_pos)

        self.right_robot_pos = (
            np.array([-0.55, 0, 0]),
            np.array([0.707, 0.0, 0.0, -0.707]),
        )

        self.right_robot = self.ros_world.add_robot(
            CortexUR(
                name="ur_2",
                urdf_path=self.urdf_path,
                rmp_path=self.rmp_path,
                position=self.right_robot_pos[0],
                orientation=self.right_robot_pos[1],
            )
        )

        self.right_robot.set_world_pose(*self.right_robot_pos)
        self.right_robot.set_default_state(*self.right_robot_pos)
        self._robot_2 = self.right_robot

        self.robot.motion_policy.set_robot_base_pose(*self.robot_pos)
        self.right_robot.motion_policy.set_robot_base_pose(*self.right_robot_pos)

        add_reference_to_stage(
            usd_path=self.assets_root_path
            + "/Isaac/Environments/Office/Props/SM_TableC.usd",
            prim_path=f"/World/Tables/table",
        )
        add_reference_to_stage(
            usd_path=self.assets_root_path
            + "/Isaac/Environments/Office/Props/SM_TableC.usd",
            prim_path=f"/World/Tables/smalltable",
        )
        add_reference_to_stage(
            # usd_path=self.assets_root_path + "/../../../../Projects/YCBV/Cracker/textured_obj.usd",
            # usd_path=self.assets_root_path + "/../../../../Projects/YCBV/Banana/textured_obj.usd",
            usd_path=self.assets_root_path
            + "/../../../../Projects/YCBV/Wood/textured_obj.usd",
            prim_path=f"/World/obj",
        )

        self.table = XFormPrim(
            prim_path=f"/World/Tables/table",
            name="table",
            position=np.array([0.4, 0.11, -0.77]),
            orientation=np.array([0.707, 0, 0, 0.707]),
            scale=[1.8, 1.8, 1.7],
        )  # w,x,y,z
        self.smalltable = XFormPrim(
            prim_path=f"/World/Tables/smalltable",
            name="table",
            position=np.array([-0.98, 0.11, -0.8]),
            orientation=np.array([0.707, 0, 0, 0.707]),
            scale=[1.8, .5, 1.7],
        )  # w,x,y,z

        self.cortex_small_table = FixedCuboid(
            "/World/Tables/cortex_small_table",
            position=np.array([-0.42, 0.09, -0.13]),
            orientation=np.array([1, 0, 0, 0]),
            color=np.array([0, 0.2196, 0.3961]),
            size=0.8,
            scale=[2.7, 1.9, 0.1],
            visible=False
        )
        self.cortex_table = FixedCuboid(
            "/World/Tables/cortex_table",
            position=np.array([0.39, 0.09, -0.096]),
            orientation=np.array([1, 0, 0, 0]),
            color=np.array([0, 0.2196, 0.3961]),
            size=0.8,
            scale=[2.7, 1.9, 0.1],
            visible=False
        )
        self.cortex_table1 = FixedCuboid(
            "/World/Tables/cortex_table1",
            position=np.array([-1.53,.09,.9]),
            orientation=np.array([.707, 0, .707, 0]),
            color=np.array([0, 0.2196, 0.3961]),
            size=0.8,
            scale=[2.7, 1.9, 0.1],
            visible=False
        )
        self.cortex_table2 = FixedCuboid(
            "/World/Tables/cortex_table2",
            position=np.array([.23,.09,.9]),
            orientation=np.array([.707, 0, .707, 0]),
            color=np.array([0, 0.2196, 0.3961]),
            size=0.8,
            scale=[2.7, 1.9, 0.1],
            visible=False
        )
        self.cortex_table3 = FixedCuboid(
            "/World/Tables/cortex_table3",
            position=np.array([-0.65, -0.62, 0.9]),
            orientation=np.array([.5,-.5,.5,.5]),
            color=np.array([0, 0.2196, 0.3961]),
            size=0.8,
            scale=[2.7, 2.1, 0.1],
            visible=False

        )
        self.cortex_table4 = FixedCuboid(
            "/World/Tables/cortex_table4",
            position=np.array([-0.65, .81, 0.9]),
            orientation=np.array([-.5,.5,.5,.5]),
            color=np.array([0, 0.2196, 0.3961]),
            size=0.8,
            scale=[2.7, 2.1, 0.1],
            visible=False

        )

        # Create a cuboid to visualize where the ee frame is according to the kinematics"
        self.right_cube = VisualCuboid(
            "/World/Control/right_cube",
            position=np.array([-1, 0.21, 0.215]),
            orientation=np.array([0, -.707,.707, 0]),
            size=0.005,
            color=np.array([0, 0, 1]),
        )

        self.left_cube = VisualCuboid(
            "/World/Control/left_cube",
            position=np.array([1, 0.3, 0.4]),
            orientation=np.array([0, 0, 1, 0]),
            size=0.005,
            color=np.array([0, 0, 1]),
        )

if __name__ == "__main__":
    rclpy.init()
    subscriber = UR_World()
    subscriber.run_simulation()
